from typing import Optional
from orbit.nn import Module
from orbit.nn.losses import Loss
from orbit.nn.optimizers import Optimizer
from orbit.nn.training import Trainer
from orbit.nn.training.snapshots import MAX_SNAPSHOT_PARAMETERS, SnapshotRecorder, parameter_count
from orbit.core import DataLoader, Results

class Experiment:
    def __init__(
        self, 
        model: Module, 
        loss_fn: Loss, 
        optimizer: Optimizer, 
        dataloader: DataLoader, 
        epochs: int, 
        name: Optional[str] = None,
        verbose: bool = False,
        log_every: int = 100,
        accuracy_fn=None,
        test_dataloader: Optional[DataLoader] = None,
        task: Optional[str] = None,
        accuracy_tolerance: Optional[float] = None,
        grad_clip: Optional[float] = None,
        val_dataloader: Optional[DataLoader] = None,
        patience: Optional[int] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.epochs = epochs
        self.name = name
        self.log_every = log_every
        self.verbose = verbose
        self.accuracy_fn = accuracy_fn
        self.test_dataloader = test_dataloader
        self.task = task
        self.accuracy_tolerance = accuracy_tolerance
        self.grad_clip = grad_clip
        self.val_dataloader = val_dataloader
        self.patience = patience

        self.trainer = Trainer()
        # Weight snapshots from the last run() (a SnapshotRecorder), or None
        # if it hasn't run or the model was too big to record - see
        # _start_snapshots. Kept off Results so results.json stays small;
        # the CLI saves them next to the checkpoint instead.
        self.snapshots = None

    def _start_snapshots(self) -> Optional[SnapshotRecorder]:
        count = parameter_count(self.model)
        if count > MAX_SNAPSHOT_PARAMETERS:
            from orbit.ui import warning
            warning(
                f"Model has {count:,} parameters - not recording weight snapshots "
                f"(limit {MAX_SNAPSHOT_PARAMETERS:,}), so training can't be animated."
            )
            return None
        recorder = SnapshotRecorder(self.epochs)
        recorder.record(0, self.model, float("nan"))
        return recorder

    def run(self, skip_test: bool = False, on_epoch_end=None) -> Results:
        """
        on_epoch_end: optional extra callable(epoch, model, avg_loss) run
        after every epoch, alongside the snapshot recorder (e.g. the live
        terminal view).
        """
        self.snapshots = self._start_snapshots()
        callbacks = [c for c in (self.snapshots, on_epoch_end) if c is not None]

        def after_epoch(epoch, model, avg_loss):
            for callback in callbacks:
                callback(epoch, model, avg_loss)

        final_loss = self.trainer.fit(
            self.model,
            self.loss_fn,
            self.optimizer,
            self.dataloader,
            self.epochs,
            verbose=self.verbose,
            log_every=self.log_every,
            accuracy_fn=self.accuracy_fn,
            on_epoch_end=after_epoch if callbacks else None,
            grad_clip=self.grad_clip,
            val_dataloader=self.val_dataloader,
            patience=self.patience,
        )

        # A diverged model's test loss would just be NaN too - leave the test
        # metrics unset (None) instead, so rankings list the run as
        # incomplete rather than comparing a meaningless number.
        diverged = self.trainer.diverged_at_epoch is not None
        test_loss, test_accuracy = None, None
        if not skip_test and not diverged and self.test_dataloader is not None:
            test_loss, test_accuracy = self.trainer.evaluate(
                self.model, self.loss_fn, self.test_dataloader, accuracy_fn=self.accuracy_fn
            )

        hyperparams = {"epochs": self.epochs, "lr": self.optimizer.lr, "batch_size": self.dataloader.batch_size, "loss": self.loss_fn.name}
        # Recorded so displays know whether accuracy_history/test_accuracy
        # hold a percentage-style accuracy or an R^2 (see metrics.format_metric).
        if self.task is not None:
            hyperparams["task"] = self.task
        if self.accuracy_tolerance is not None:
            hyperparams["accuracy_tolerance"] = self.accuracy_tolerance
        if self.grad_clip:
            hyperparams["grad_clip"] = self.grad_clip
        if self.patience:
            hyperparams["patience"] = self.patience

        return Results(
            final_loss = final_loss,
            loss_history = self.trainer.history,
            hyperparams=hyperparams,
            name = self.name,
            duration_seconds = self.trainer.duration_seconds,
            gradient_norm_history = self.trainer.gradient_norm_history,
            accuracy_history = self.trainer.accuracy_history,
            test_loss = test_loss,
            test_accuracy = test_accuracy,
            layer_gradient_norm_history = self.trainer.layer_gradient_norm_history,
            diverged_at_epoch = self.trainer.diverged_at_epoch,
            val_loss_history = self.trainer.val_loss_history,
            val_accuracy_history = self.trainer.val_accuracy_history,
            best_epoch = self.trainer.best_epoch,
            stopped_early_at_epoch = self.trainer.stopped_early_at_epoch,
            )
