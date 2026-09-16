from orbit.core import DataLoader
from orbit.nn import Module
from orbit.nn.losses import Loss
from orbit.nn.optimizers import Optimizer

class Trainer:
    def __init__(self):
        self.history = []

    def fit(self, model: Module, loss_fn: Loss, optimizer: Optimizer, dataloader: DataLoader, epochs: int,
            verbose: bool = False, log_every: int = 100):

        
        for e in range(1, epochs+1):
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

            avg_loss = total_loss / total_samples

            if verbose and e % log_every == 0:
                print(f"{e} | {avg_loss}")

            self.history.append(avg_loss)

        return avg_loss
