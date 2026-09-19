from orbit.nn import Module, Parameter
from orbit.core import Tensor
import numpy as np

class Linear(Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Xavier/Glorot normal init: without this, a Linear's output variance
        # grows with in_features (unscaled randn), which saturates Tanh/
        # Sigmoid on wide layers and kills gradient flow before training can
        # even start. Scaling by fan-in/fan-out keeps pre-activation variance
        # roughly independent of layer width, regardless of which activation
        # follows.
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = self.register_parameter(
            "weight", Parameter(scale * np.random.randn(in_features, out_features))
        )
        self.bias = self.register_parameter(
            "bias", Parameter(np.zeros(out_features))
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

if __name__ == "__main__":
    for _ in range(1):
        linear = Linear(5, 3)
        x = Tensor(np.array([[10,20,30,40,50]]))

        print(x.shape)
        print(linear.weight.shape)
        print(linear.bias.shape)
        print(linear.forward(x).shape)
        # print(linear.forward(x).data)
        # print("\n")
        # print(f"mean: {linear.weight.data.mean()}")
        # print(f"std: {linear.weight.data.std()}")
        # print("\n")


