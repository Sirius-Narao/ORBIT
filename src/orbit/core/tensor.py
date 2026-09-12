import numpy as np
from typing import List, Optional, Union, Callable

try:
    from .autograd import (
        backward,
        backward_add,
        backward_subtract,
        backward_multiply,
        backward_divide,
        backward_matmul,
        backward_sum,
        backward_mean,
        backward_neg,
        backward_pow,
        backward_relu,
        backward_sigmoid,
        backward_tanh,
        backward_softmax,
        backward_cross_entropy,
    )
except ImportError:
    from orbit.core.autograd import (
        backward,
        backward_add,
        backward_subtract,
        backward_multiply,
        backward_divide,
        backward_matmul,
        backward_sum,
        backward_mean,
        backward_neg,
        backward_pow,
        backward_relu,
        backward_sigmoid,
        backward_tanh,
        backward_softmax,
        backward_cross_entropy,
    )

class Tensor:
    """
    The "Atom" or "Cell" of ORBIT.
    A multi-dimensional container for data, capable of performing operations and
    tracking its lineage for automatic differentiation.

    Supports:
        - Basic arithmetic operations (+, -, *, /, **)
        - Matrix multiplication (@)
        - Reduction operations (sum, mean)
        - Chain rule differentiation
    """
    
    def __init__(
            self, 
            data: Union[float, int, list, tuple, np.ndarray, "Tensor"], 
            requires_grad: bool = False, 
            grad: Optional[np.ndarray] = None, 
            operation: Optional[str] = None, 
            parents: Optional[List["Tensor"]] = None
        ):
        if isinstance(data, Tensor):
            self.data = np.array(data.data)
        else:
            self.data = np.array(data)
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        self.grad = grad
        self.operation = operation
        self.parents = parents if parents is not None else []
        self._backward: Optional[Callable[["Tensor"], None]] = None
        self.ndim = self.data.ndim
        self.size = self.data.size
        self.dtype = self.data.dtype

    # Pretty print
    def __repr__(self):
        return f"Tensor({self.data})"

    def __str__(self):
        return f"---------------\nTensor:\nData: {self.data}\nShape: {self.shape}\nComes from operation: {self.operation}\nWith following parents: {self.parents}\n---------------\n"

    # Reverse Math Operators
    def __radd__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            x.data + self.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "+",
            parents = [x, self]
        )
        out._backward = lambda res=out, left=x, right=self: backward_add(res, left, right)
        return out

    def __rmul__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            x.data * self.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "*",
            parents = [x, self]
        )
        out._backward = lambda res=out, left=x, right=self: backward_multiply(res, left, right)
        return out

    def __rsub__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            x.data - self.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "-",
            parents = [x, self]
        )
        out._backward = lambda res=out, left=x, right=self: backward_subtract(res, left, right)
        return out

    def __rpow__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            x.data ** self.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "**",
            parents = [x, self]
        )
        out._backward = lambda res=out, left=x, right=self: backward_pow(res, left, right)
        return out

    def __rtruediv__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            x.data / self.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "/",
            parents = [x, self]
        )
        out._backward = lambda res=out, left=x, right=self: backward_divide(res, left, right)
        return out

    def __rmatmul__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            np.matmul(x.data, self.data),
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "@",
            parents = [x, self]
        )
        out._backward = lambda res=out, left=x, right=self: backward_matmul(res, left, right)
        return out

    # Math Operators
    def __neg__(self):
        out = Tensor(
            -self.data,
            requires_grad = self.requires_grad,
            operation = "neg",
            parents = [self]
        )
        out._backward = lambda res=out, parent=self: backward_neg(res, parent)
        return out

    def __add__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            self.data + x.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "+",
            parents = [self, x]
        )
        out._backward = lambda res=out, left=self, right=x: backward_add(res, left, right)
        return out

    def __mul__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            self.data * x.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "*",
            parents = [self, x]
        )
        out._backward = lambda res=out, left=self, right=x: backward_multiply(res, left, right)
        return out

    def __pow__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            self.data ** x.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "**",
            parents = [self, x]
        )
        out._backward = lambda res=out, left=self, right=x: backward_pow(res, left, right)
        return out
    
    def __sub__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            self.data - x.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "-",
            parents = [self, x]
        )
        out._backward = lambda res=out, left=self, right=x: backward_subtract(res, left, right)
        return out
    
    def __truediv__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            self.data / x.data,
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "/",
            parents = [self, x]
        )
        out._backward = lambda res=out, left=self, right=x: backward_divide(res, left, right)
        return out

    def __matmul__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = Tensor(
            np.matmul(self.data, x.data),
            requires_grad = self.requires_grad or x.requires_grad,
            operation = "@",
            parents = [self, x]
        )
        out._backward = lambda res=out, left=self, right=x: backward_matmul(res, left, right)
        return out
    
    # Practical functions
    def sum(self, axis = None):
        out = Tensor(
            self.data.sum(axis = axis),
            requires_grad = self.requires_grad,
            operation = "sum",
            parents = [self]
        )
        out._backward = lambda res=out, parent=self: backward_sum(res, parent)
        return out
    
    def mean(self, axis = None):
        out = Tensor(
            self.data.mean(axis = axis),
            requires_grad = self.requires_grad,
            operation = "mean",
            parents = [self]
        )
        out._backward = lambda res=out, parent=self: backward_mean(res, parent)
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            operation="reshape",
            parents=[self]
        )

    def relu(self):
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            operation="relu",
            parents=[self]
        )
        out._backward = lambda res=out, parent=self: backward_relu(res, parent)
        return out
    
    def sigmoid(self):
        out = Tensor(
            1 / (1 + np.exp(-self.data)),
            requires_grad=self.requires_grad,
            operation="sigmoid",
            parents=[self]
        )
        out._backward = lambda res=out, parent=self: backward_sigmoid(res, parent)
        return out

    def tanh(self):
        out = Tensor(
            np.tanh(self.data),
            requires_grad=self.requires_grad,
            operation="tanh",
            parents=[self]
        )
        out._backward = lambda res=out, parent=self: backward_tanh(res, parent)
        return out

    def softmax(self, axis: int = -1):
        exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        softmax_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        out = Tensor(
            softmax_data,
            requires_grad=self.requires_grad,
            operation="softmax",
            parents=[self]
        )
        out._backward = lambda res=out, parent=self, a=axis: backward_softmax(res, parent, axis=a)
        return out

    # Slicing
    def __getitem__(self, index):
        return Tensor(
            self.data[index],
            requires_grad = self.requires_grad,
            operation = "getitem",
            parents = [self]
        )

    # Backward pass
    def backward(self, gradient: Optional[np.ndarray] = None):
        backward(self, gradient=gradient)

if __name__ == "__main__":
    X = Tensor([
        [1, 2.2, 3], 
        [4, 5, 6]
    ])

    Y = Tensor([2, 3, 4])

    Z = X @ Y
    print(getattr(Z, "_backward", None))