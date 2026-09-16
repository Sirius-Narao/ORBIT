from orbit.nn import Parameter
import numpy as np

class Optimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

if __name__ == "__main__":
    param1 = Parameter([[1, 2, 3, 4]], grad=79)
    param2 = Parameter([[1, 8, 1, 9]], grad=9)

    opt = Optimizer([param1, param2])
    opt.zero_grad()

    print(opt.parameters[0].grad)