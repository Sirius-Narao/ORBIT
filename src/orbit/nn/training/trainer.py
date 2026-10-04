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
            optimizer.step()

        # zero_grad() runs at the START of each batch, not the end of the
        # epoch, so grad_norm (and the model's grads, which _record_epoch
        # reads for the per-layer norms) hold the last batch's values - not
        # a running average across the epoch.
        avg_loss = total_loss / total_samples
        avg_accuracy = _whole_pass_metric(accuracy_fn, predictions, targets)
        return avg_loss, grad_norm, avg_accuracy

    def _record_epoch(self, model: Module, avg_loss, grad_norm, avg_accuracy) -> None:
        """
        Append one epoch to every history. The model's grads still hold the
        epoch's last batch, same as grad_norm (see _run_epoch).
        """
        self.history.append(avg_loss)
        self.gradient_norm_history.append(grad_norm)
        for name, norm in _layer_gradient_norms(model).items():
            self.layer_gradient_norm_history.setdefault(name, []).append(norm)
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
        model.train()

        avg_loss = total_loss / total_samples
        avg_accuracy = _whole_pass_metric(accuracy_fn, predictions, targets)
        return avg_loss, avg_accuracy

    def _evaluate_with_progress_bar(
        self, model: Module, loss_fn: Loss, dataloader: DataLoader, accuracy_fn=None
    ):
        model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions, targets = [], []
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
            for X_batch, Y_batch in dataloader:
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, Y_batch)
                batch_size = X_batch.shape[0]
                total_loss += loss.data * batch_size
                total_samples += batch_size
                if accuracy_fn is not None:
                    predictions.append(y_pred.data)
                    targets.append(Y_batch.data)
                progress.update(task, advance=1, loss=total_loss / total_samples)
        console.print()
        model.train()

        avg_loss = total_loss / total_samples
        avg_accuracy = _whole_pass_metric(accuracy_fn, predictions, targets)
        return avg_loss, avg_accuracy

    def fit(self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader, epochs: int,
            verbose: bool = False, log_every: int = 100, accuracy_fn=None, on_epoch_end=None):
        """
        on_epoch_end: optional callable(epoch, model, avg_loss), called after
        every epoch (epochs count from 1) - how weight snapshots and the live
        terminal view watch training without the Trainer knowing about either.

        Returns the last epoch's average loss, or NaN if training diverged -
        then self.diverged_at_epoch says when, and the histories hold only
        the epochs before it (see _run_epoch).
        """

        start = time.time()
        if accuracy_fn is not None:
            self.accuracy_history = []

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
            if on_epoch_end is not None:
                on_epoch_end(e, model, avg_loss)

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
                if on_epoch_end is not None:
                    on_epoch_end(e, model, avg_loss)
                progress.update(task, advance=1, loss=avg_loss)
        console.print()

        return avg_loss
