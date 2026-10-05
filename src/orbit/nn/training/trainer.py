import time

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from orbit.core import DataLoader, Tensor
from orbit.nn import Module
from orbit.nn.losses import Loss
from orbit.nn.optimizers import Optimizer
from orbit.ui import console, is_tty


def _gradient_norm(model: Module) -> float:
    """
    L2 norm of every parameter's gradient, flattened and concatenated - the
    standard signal for spotting vanishing/exploding gradients. Parameters
    whose grad is None (never involved in the loss) are skipped.
    """
    squared_sum = 0.0
    for param in model.parameters():
        if param.grad is not None:
            squared_sum += float(np.sum(param.grad ** 2))
    return float(np.sqrt(squared_sum))


def _layer_gradient_norms(model: Module) -> dict:
    """
    L2 norm of each weight matrix's gradient, keyed by parameter name (e.g.
    "0.weight", "2.weight"). One global norm can look healthy while the
    first layers get almost nothing - per layer is what shows vanishing or
    exploding gradients *where* they happen. Biases are left out: they're
    the same signal, minus the dependence on the layer's input.
    """
    return {
        name: float(np.sqrt(np.sum(param.grad ** 2)))
        for name, param in model.named_parameters()
        if name.split(".")[-1] == "weight" and param.grad is not None
    }


def _clip_gradients(model: Module, grad_norm: float, max_norm: float) -> float:
    """
    Gradient clipping by global norm: if the gradient of all parameters,
    seen as one long vector g, is longer than max_norm, shrink it to length
    max_norm - every parameter's gradient multiplied by the same
    scale = max_norm / ||g||. Returns that scale (1.0 when no clipping
    was needed).

    Scaling everything by one factor keeps the gradient's *direction* -
    the step still goes downhill the same way - and only caps its *size*.
    That's what stops the runaway behind divergence: an oversized step
    overshoots, which makes the next gradient bigger, which makes the next
    step bigger still; with a cap on the step's length that feedback loop
    can't grow without bound. Because it acts on param.grad before
    optimizer.step(), it works the same for every optimizer.
    """
    if grad_norm <= max_norm:
        return 1.0
    scale = max_norm / grad_norm
    for param in model.parameters():
        if param.grad is not None:
            param.grad = param.grad * scale
    return scale


def _whole_pass_metric(accuracy_fn, predictions: list, targets: list):
    """
    Apply accuracy_fn once to every sample seen in an epoch/evaluation pass,
    rather than averaging per-batch values. For a per-sample average like
    classification accuracy both give the same number, but R^2 isn't one -
    averaging per-batch R^2 is wrong, and a 1-row batch has no R^2 at all.
    """
    if accuracy_fn is None:
        return None
    return accuracy_fn(Tensor(np.concatenate(predictions)), Tensor(np.concatenate(targets)))


class Trainer:
    def __init__(self):
        self.history = []
        self.duration_seconds = None
        self.gradient_norm_history = []
        self.layer_gradient_norm_history = {}
        self.accuracy_history = None
        # The epoch whose loss or gradients became inf/NaN, if training
        # diverged (see _run_epoch) - None for a run that finished normally.
        self.diverged_at_epoch = None
        # Max global gradient norm per batch (see _clip_gradients), set by
        # fit(); None/0 = no clipping.
        self.grad_clip = None
        # The last batch's clipping scale, so _record_epoch can report the
        # gradient as it was *before* clipping.
        self._last_clip_scale = 1.0
        # Validation (see fit's val_dataloader/patience): per-epoch loss and
        # accuracy on the validation set, None when there is none.
        self.val_loss_history = None
        self.val_accuracy_history = None
        # Early stopping: the epoch with the lowest validation loss (whose
        # weights the model ends with) and the epoch training stopped at.
        self.best_epoch = None
        self.stopped_early_at_epoch = None
        self._val_dataloader = None
        self._patience = None
        self._best_val_loss = None
        self._best_weights = None

    def _run_epoch(
        self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader, accuracy_fn=None
    ):
        """
        One pass over dataloader, updating the model after every batch.
        Returns (avg_loss, grad_norm, avg_accuracy), or (None, None, None)
        if training diverged during the epoch.

        Divergence: once steps are too large for the loss surface, each
        update overshoots further than the last and the numbers grow
        geometrically until they overflow float64 (~1e308) into inf, and
        then into NaN (inf - inf, 0 * inf). From there every later value is
        NaN too, so there is nothing left to learn - the epoch is abandoned
        at the first non-finite loss (checked before backward()) or
        gradient norm (checked before optimizer.step()). Stopping before
        the update keeps the weights themselves finite, so the checkpoint
        and the visualizations still work on a diverged run.
        """
        total_loss = 0.0
        total_samples = 0
        predictions, targets = [], []
        grad_norm = 0.0
        for X_batch, Y_batch in dataloader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch) # loss is a Tensor
            if not np.all(np.isfinite(loss.data)):
                return None, None, None
            batch_size = X_batch.shape[0]
            total_loss += loss.data * batch_size
            total_samples += batch_size
            if accuracy_fn is not None:
                predictions.append(y_pred.data)
                targets.append(Y_batch.data)
            model.zero_grad()
            loss.backward()
            grad_norm = _gradient_norm(model)
            if not np.isfinite(grad_norm):
                return None, None, None
            self._last_clip_scale = (
                _clip_gradients(model, grad_norm, self.grad_clip) if self.grad_clip else 1.0
            )
            optimizer.step()

        # zero_grad() runs at the START of each batch, not the end of the
        # epoch, so grad_norm (and the model's grads, which _record_epoch
        # reads for the per-layer norms) hold the last batch's values - not
        # a running average across the epoch. grad_norm is measured before
        # any clipping, since the unclipped size is what shows a run
        # heading for an explosion.
        avg_loss = total_loss / total_samples
        avg_accuracy = _whole_pass_metric(accuracy_fn, predictions, targets)
        return avg_loss, grad_norm, avg_accuracy

    def _record_epoch(self, model: Module, avg_loss, grad_norm, avg_accuracy) -> None:
        """
        Append one epoch to every history. The model's grads still hold the
        epoch's last batch, same as grad_norm (see _run_epoch). If that batch
        was clipped, every grad was multiplied by the same scale, so dividing
        each layer's norm by it recovers the exact pre-clip value - matching
        grad_norm, which is measured before clipping.
        """
        self.history.append(avg_loss)
        self.gradient_norm_history.append(grad_norm)
        for name, norm in _layer_gradient_norms(model).items():
            self.layer_gradient_norm_history.setdefault(name, []).append(norm / self._last_clip_scale)
        if avg_accuracy is not None:
            self.accuracy_history.append(avg_accuracy)

    def evaluate(self, model: Module, loss_fn: Loss, dataloader: DataLoader, accuracy_fn=None):
        """
        Forward-pass-only pass over dataloader: no zero_grad()/backward()/
        optimizer.step(), so it never mutates the model. Mirrors
        _run_epoch's batch-weighted loss averaging and whole-pass accuracy
        metric. Used for a post-training
        test-set pass (orbit run, once test_split is set) and by the
        standalone `orbit test` command.
        """
        if is_tty():
            return self._evaluate_with_progress_bar(model, loss_fn, dataloader, accuracy_fn=accuracy_fn)
        return self._evaluate_pass(model, loss_fn, dataloader, accuracy_fn)

    def _evaluate_pass(self, model: Module, loss_fn: Loss, dataloader: DataLoader, accuracy_fn=None,
                       on_batch=None):
        """
        The loop behind evaluate() and the per-epoch validation pass.
        on_batch(running_loss) is called after every batch (the progress bar).
        """
        model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions, targets = [], []
        for X_batch, Y_batch in dataloader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)
            batch_size = X_batch.shape[0]
            total_loss += loss.data * batch_size
            total_samples += batch_size
            if accuracy_fn is not None:
                predictions.append(y_pred.data)
                targets.append(Y_batch.data)
            if on_batch is not None:
                on_batch(total_loss / total_samples)
        model.train()

        avg_loss = total_loss / total_samples
        avg_accuracy = _whole_pass_metric(accuracy_fn, predictions, targets)
        return avg_loss, avg_accuracy

    def _evaluate_with_progress_bar(
        self, model: Module, loss_fn: Loss, dataloader: DataLoader, accuracy_fn=None
    ):
        console.print()
        with Progress(
            TextColumn("[bold #ffeab0]Evaluating[/bold #ffeab0]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("batch {task.completed}/{task.total}"),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("evaluate", total=len(dataloader), loss=float("nan"))
            result = self._evaluate_pass(
                model, loss_fn, dataloader, accuracy_fn,
                on_batch=lambda loss: progress.update(task, advance=1, loss=loss),
            )
        console.print()
        return result

    def _validate(self, model: Module, loss_fn: Loss, accuracy_fn, epoch: int) -> bool:
        """
        Measure the epoch's model on the validation set and, with patience,
        apply early stopping. Returns True when training should stop.

        Early stopping: training loss keeps falling as a model starts to
        memorize its training rows, but the loss on rows it never trains on
        (the validation set) stops improving, then rises - the model is
        overfitting. So keep a copy of the weights from the epoch with the
        lowest validation loss so far, and stop once `patience` epochs in a
        row have failed to beat it; fit() then puts those best weights back.
        """
        val_loss, val_accuracy = self._evaluate_pass(model, loss_fn, self._val_dataloader, accuracy_fn)
        self.val_loss_history.append(val_loss)
        if val_accuracy is not None:
            self.val_accuracy_history.append(val_accuracy)

        if not self._patience:
            return False
        if self._best_val_loss is None or val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self.best_epoch = epoch
            self._best_weights = [param.data.copy() for param in model.parameters()]
            return False
        if epoch - self.best_epoch >= self._patience:
            self.stopped_early_at_epoch = epoch
            return True
        return False

    def _end_of_epoch(self, model: Module, loss_fn: Loss, accuracy_fn, epoch: int, avg_loss,
                      on_epoch_end) -> bool:
        """Validation and the on_epoch_end callback, shared by both fit paths. True = stop."""
        stop = self._validate(model, loss_fn, accuracy_fn, epoch) if self._val_dataloader is not None else False
        if on_epoch_end is not None:
            on_epoch_end(epoch, model, avg_loss)
        return stop

    def fit(self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader, epochs: int,
            verbose: bool = False, log_every: int = 100, accuracy_fn=None, on_epoch_end=None,
            grad_clip=None, val_dataloader: DataLoader = None, patience: int = None):
        """
        grad_clip: optional max global gradient norm per batch (see
        _clip_gradients); None or 0 trains without clipping.

        val_dataloader: optional validation set, measured after every epoch
        into val_loss_history/val_accuracy_history. patience (needs a
        val_dataloader): stop once the validation loss hasn't improved for
        that many epochs, and end with the best epoch's weights (see
        _validate). Without them, training is exactly as before.

        on_epoch_end: optional callable(epoch, model, avg_loss), called after
        every epoch (epochs count from 1) - how weight snapshots and the live
        terminal view watch training without the Trainer knowing about either.

        Returns the last epoch's average loss, or NaN if training diverged -
        then self.diverged_at_epoch says when, and the histories hold only
        the epochs before it (see _run_epoch).
        """

        if patience and val_dataloader is None:
            raise ValueError("Early stopping (patience) needs a validation set (val_dataloader)")

        start = time.time()
        self.grad_clip = grad_clip
        self._val_dataloader = val_dataloader
        self._patience = patience
        self.best_epoch = self.stopped_early_at_epoch = None
        self._best_val_loss = self._best_weights = None
        if accuracy_fn is not None:
            self.accuracy_history = []
        if val_dataloader is not None:
            self.val_loss_history = []
            if accuracy_fn is not None:
                self.val_accuracy_history = []

        # Overflow/invalid-value warnings are what a diverging run produces
        # on its way to inf/NaN. _run_epoch detects that and stops training,
        # and the CLI reports it in one line - instead of numpy printing a
        # wall of RuntimeWarnings.
        with np.errstate(over="ignore", invalid="ignore"):
            if verbose and is_tty():
                avg_loss = self._fit_with_progress_bar(
                    model, loss_fn, optimizer, dataloader, epochs, accuracy_fn=accuracy_fn,
                    on_epoch_end=on_epoch_end,
                )
            else:
                avg_loss = self._fit_plain(
                    model, loss_fn, optimizer, dataloader, epochs, verbose, log_every,
                    accuracy_fn=accuracy_fn, on_epoch_end=on_epoch_end,
                )

        if self._best_weights is not None:
            for param, best in zip(model.parameters(), self._best_weights):
                param.data = best

        self.duration_seconds = time.time() - start
        return float("nan") if self.diverged_at_epoch is not None else avg_loss

    def _fit_plain(
        self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader, epochs: int,
        verbose: bool, log_every: int, accuracy_fn=None, on_epoch_end=None
    ):
        avg_loss = None
        for e in range(1, epochs+1):
            avg_loss, grad_norm, avg_accuracy = self._run_epoch(
                model, loss_fn, optimizer, dataloader, accuracy_fn=accuracy_fn
            )
            if avg_loss is None:
                self.diverged_at_epoch = e
                break

            if verbose and e % log_every == 0:
                print(f"{e} | {avg_loss}")

            self._record_epoch(model, avg_loss, grad_norm, avg_accuracy)
            if self._end_of_epoch(model, loss_fn, accuracy_fn, e, avg_loss, on_epoch_end):
                break

        return avg_loss

    def _fit_with_progress_bar(
        self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader, epochs: int,
        accuracy_fn=None, on_epoch_end=None
    ) -> float:
        avg_loss = None
        console.print()
        with Progress(
            TextColumn("[bold #ffeab0]Training[/bold #ffeab0]"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("epoch {task.completed}/{task.total}"),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("train", total=epochs, loss=float("nan"))
            for e in range(1, epochs + 1):
                avg_loss, grad_norm, avg_accuracy = self._run_epoch(
                    model, loss_fn, optimizer, dataloader, accuracy_fn=accuracy_fn
                )
                if avg_loss is None:
                    self.diverged_at_epoch = e
                    break
                self._record_epoch(model, avg_loss, grad_norm, avg_accuracy)
                stop = self._end_of_epoch(model, loss_fn, accuracy_fn, e, avg_loss, on_epoch_end)
                progress.update(task, advance=1, loss=avg_loss)
                if stop:
                    break
        console.print()

        return avg_loss
