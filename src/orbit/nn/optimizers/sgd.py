from orbit.nn import Parameter
import numpy as np
try:
    from .optimizer import Optimizer
except ImportError:
    from orbit.nn.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic gradient descent, with optional (heavy-ball) momentum.

    momentum=0 is plain SGD: param -= lr * grad.

    momentum>0 keeps a per-parameter velocity, PyTorch's convention:
        v     = momentum * v + grad
        param = param - lr * v
    so a gradient that keeps pointing the same way accumulates speed
    (up to 1 / (1 - momentum) times the plain step), while one that flips
    sign every step mostly cancels itself out.
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        # One velocity buffer per parameter, created lazily on its first
        # non-None gradient (keyed by position in self.parameters).
        self.velocities = {}

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            if self.momentum == 0:
                # Kept as its own branch so plain SGD stays the exact same
                # arithmetic as before momentum existed.
                param.data -= self.lr * param.grad
                continue
            if i not in self.velocities:
                self.velocities[i] = np.zeros_like(param.data, dtype=float)
            self.velocities[i] = self.momentum * self.velocities[i] + param.grad
            param.data -= self.lr * self.velocities[i]
