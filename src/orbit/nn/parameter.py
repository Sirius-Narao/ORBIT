from orbit.core.tensor import Tensor
from typing import Union, Optional, List
import numpy as np

class Parameter(Tensor):
    """
    Parameters are the same as tensors but are meant to be trained. 
    Therefore, they are always differentiable and require their gradients to be computed.
    """

    def __init__(
            self, 
            data: Union[float, int, list, tuple, np.ndarray, "Tensor"], 
            requires_grad: bool = True, 
            grad: Optional[np.ndarray] = None, 
            operation: Optional[str] = None, 
            parents: Optional[List["Tensor"]] = None
        ):
        super().__init__(data, requires_grad, grad, operation, parents)
    
    # Pretty print
    def __repr__(self):
        return f"Parameter({self.data})"

    def __str__(self):
        return f"---------------\nParameter:\nData: {self.data}\nShape: {self.shape}\nComes from operation: {self.operation}\nWith following parents: {self.parents}\n---------------\n"