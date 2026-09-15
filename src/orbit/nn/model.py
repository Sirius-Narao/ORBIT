from orbit.nn.module import Module
from orbit.core import Tensor
class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()

        for i, layer in enumerate(layers):
            self.register_module(str(i), layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x


if __name__ == "__main__":
    from orbit.nn.layers.linear import Linear
    from orbit.nn.activations import Tanh, Sigmoid
    import numpy as np

    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1), Sigmoid())
    x = Tensor(np.array([[0.0, 1.0], [1.0, 0.0]]))

    print(model(x).data)
