from orbit.nn.losses.loss import Loss
from orbit.core.tensor import Tensor

class MSE(Loss):

    def __init__(self):
        super().__init__("mse")
    
    def forward(self, y_pred: Tensor, y_true: Tensor):
        # calculate loss: mean over (y_pred - y_true)²
        # Each op (-, **, mean) builds a graph node with parents + _backward,
        # so returning `loss` directly preserves the full chain for backprop.
        loss = ((y_pred - y_true) ** 2).mean()
        # return tensor
        return loss