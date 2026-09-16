import numpy as np
try:
    from .activation import Activation
except ImportError:
    from orbit.nn.activations.activation import Activation
from orbit.core import Tensor

class Softmax(Activation):
    """
    Softmax activation function.

    Applies softmax along the specified axis:
        Softmax(x_i) = exp(x_i) / sum(exp(x))
    """
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.output = x.softmax(axis=self.axis)
        return self.output

if __name__ == "__main__":
    # Example demonstrating Softmax usage integrated with Autograd
    softmax = Softmax(axis=-1)
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    output = softmax(x)
    print(f"Input:\n{x}")
    print(f"Output:\n{output}")

    # Compute scalar loss and run automatic differentiation
    loss = output.sum()
    loss.backward()
    print(f"x.grad: {x.grad}")
