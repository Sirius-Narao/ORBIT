import numpy as np
try:
    from .optimizer import Optimizer
except ImportError:
    from orbit.nn.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam (Kingma & Ba, 2014): SGD with a per-weight adaptive step size.

    For every parameter it keeps two exponential moving averages:
        m = beta1 * m + (1 - beta1) * g        (mean of the gradient)
        v = beta2 * v + (1 - beta2) * g**2     (mean of the squared gradient)

    Both start at zero, so early on they're biased toward zero (after one
    step, m is only (1 - beta1) * g = 0.1 * g). Dividing by (1 - beta**t)
    undoes exactly that shrinkage:
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
    (after step 1: m_hat = g, v_hat = g**2 exactly).

    The update is then
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
    m_hat / sqrt(v_hat) is roughly "gradient divided by its own typical
    size", so every weight moves about lr per step regardless of how large
    or small its raw gradient is - which is why Adam needs far less
    learning-rate tuning than plain SGD. eps only guards against dividing
    by zero.
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.betas = tuple(betas)
        self.eps = eps
        # Per-parameter state keyed by position in self.parameters. The step
        # count t is per parameter too, so a parameter skipped this step
        # (grad is None) doesn't get its bias correction advanced.
        self.m = {}
        self.v = {}
        self.t = {}

    def step(self):
        beta1, beta2 = self.betas
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            if i not in self.t:
                self.m[i] = np.zeros_like(param.data, dtype=float)
                self.v[i] = np.zeros_like(param.data, dtype=float)
                self.t[i] = 0

            g = param.grad
            self.t[i] += 1
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g ** 2

            m_hat = self.m[i] / (1 - beta1 ** self.t[i])
            v_hat = self.v[i] / (1 - beta2 ** self.t[i])

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
