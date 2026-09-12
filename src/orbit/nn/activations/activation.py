from orbit.nn.module import Module
import numpy as np

class Activation(Module):
    """
    Activation functions base class.
    """
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        """
        Forward pass.
        """
        raise NotImplementedError