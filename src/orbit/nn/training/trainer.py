import time

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from orbit.core import DataLoader
from orbit.nn import Module
from orbit.nn.losses import Loss
from orbit.nn.optimizers import Optimizer
from orbit.ui import console, is_tty

class Trainer:
    def __init__(self):
        self.history = []
        self.duration_seconds = None

    def _run_epoch(self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader) -> float:
        total_loss = 0.0
        total_samples = 0
        for X_batch, Y_batch in dataloader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch) # loss is a Tensor
            batch_size = X_batch.shape[0]
            total_loss += loss.data * batch_size
            total_samples += batch_size
            model.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss / total_samples

    def fit(self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader, epochs: int,
            verbose: bool = False, log_every: int = 100):

        start = time.time()

        if verbose and is_tty():
            avg_loss = self._fit_with_progress_bar(model, loss_fn, optimizer, dataloader, epochs)
            self.duration_seconds = time.time() - start
            return avg_loss

        avg_loss = None
        for e in range(1, epochs+1):
            avg_loss = self._run_epoch(model, loss_fn, optimizer, dataloader)

            if verbose and e % log_every == 0:
                print(f"{e} | {avg_loss}")

            self.history.append(avg_loss)

        self.duration_seconds = time.time() - start
        return avg_loss

    def _fit_with_progress_bar(self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader, epochs: int) -> float:
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
            for _ in range(epochs):
                avg_loss = self._run_epoch(model, loss_fn, optimizer, dataloader)
                self.history.append(avg_loss)
                progress.update(task, advance=1, loss=avg_loss)
        console.print()

        return avg_loss
