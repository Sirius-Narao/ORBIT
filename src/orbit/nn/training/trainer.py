from orbit.core import Tensor
from orbit.nn.module import Module
from orbit.nn.losses.loss import Loss
from orbit.nn.optimizers.optimizer import Optimizer

class Trainer:
    def __init__(self):
        pass

    def fit(self, model: Module, loss_fn: Loss, optimizer: Optimizer, X: Tensor, Y: Tensor, epochs: int,
            verbose: bool = False, log_every: int = 100):

        for e in range(1, epochs+1):
            y_pred = model(X)
            loss = loss_fn(y_pred, Y) # loss is a Tensor

            model.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and e % log_every == 0:
                print(f"{e} | {loss.data}")

        return loss
