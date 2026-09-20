from typing import Optional
from orbit.nn import Module
from orbit.nn.losses import Loss
from orbit.nn.optimizers import Optimizer
from orbit.nn.training import Trainer
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

        self.trainer = Trainer()

    def run(self) -> Results:
        final_loss = self.trainer.fit(
            self.model,
            self.loss_fn,
            self.optimizer,
            self.dataloader,
            self.epochs,
            verbose=self.verbose,
            log_every=self.log_every,
            accuracy_fn=self.accuracy_fn,
        )

        test_loss, test_accuracy = None, None
        if self.test_dataloader is not None:
            test_loss, test_accuracy = self.trainer.evaluate(
                self.model, self.loss_fn, self.test_dataloader, accuracy_fn=self.accuracy_fn
            )

        return Results(
            final_loss = final_loss,
            loss_history = self.trainer.history,
            hyperparams={"epochs": self.epochs, "lr": self.optimizer.lr, "batch_size": self.dataloader.batch_size, "loss": self.loss_fn.name},
            name = self.name,
            duration_seconds = self.trainer.duration_seconds,
            gradient_norm_history = self.trainer.gradient_norm_history,
            accuracy_history = self.trainer.accuracy_history,
            test_loss = test_loss,
            test_accuracy = test_accuracy,
            )
