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
        log_every: int = 100
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.epochs = epochs
        self.name = name
        self.log_every = log_every
        self.verbose = verbose

        self.trainer = Trainer()
        
    def run(self) -> Results:
        final_loss = self.trainer.fit(
            self.model,
            self.loss_fn,
            self.optimizer,
            self.dataloader,
            self.epochs,
            verbose=self.verbose,
            log_every=self.log_every
        )

        return Results(
            final_loss = final_loss,
            loss_history = self.trainer.history,
            hyperparams={"epochs": self.epochs, "lr": self.optimizer.lr, "batch_size": self.dataloader.batch_size, "loss": self.loss_fn.name},
            name = self.name
            )
