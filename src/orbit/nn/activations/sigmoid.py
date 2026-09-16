import numpy as np
try:
    from .activation import Activation
except ImportError:
    from orbit.nn.activations.activation import Activation
from orbit.core import Tensor

class Sigmoid(Activation):
    """
    Sigmoid activation function.

    Applies element-wise:
        Sigmoid(x) = 1 / (1 + exp(-x))
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.output = x.sigmoid()
        return self.output

if __name__ == "__main__":
    # Example demonstrating Sigmoid usage integrated with Autograd
    sigmoid = Sigmoid()
    x = Tensor(np.array([-1.0, 0.0, 1.0, 2.0]), requires_grad=True)
    output = sigmoid(x)
    print(f"Input:\n{x}")
    print(f"Output:\n{output}")

    # Compute scalar loss and run automatic differentiation
    loss = output.sum()
    loss.backward()
    print(f"x.grad: {x.grad}")
