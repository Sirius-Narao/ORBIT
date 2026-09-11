from orbit.nn.module import Module
from orbit.nn.parameter import Parameter
from orbit.core.tensor import Tensor
import numpy as np

class Linear(Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Parameter(np.random.randn(in_features, out_features))
        self.bias = Parameter(np.zeros(out_features))
    
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

for _ in range(1):
    linear = Linear(5, 3)
    x = Tensor([[10,20,30,40,50]])

    print(x.shape)
    print(linear.weight.shape)
    print(linear.bias.shape)
    print(linear.forward(x).shape)
    # print(linear.forward(x).data)
    # print("\n")
    # print(f"mean: {linear.weight.data.mean()}")
    # print(f"std: {linear.weight.data.std()}")
    # print("\n")


