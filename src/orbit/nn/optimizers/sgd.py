from orbit.nn import Parameter
import numpy as np
try:
    from .optimizer import Optimizer
except ImportError:
    from orbit.nn.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.data -= self.lr * param.grad
