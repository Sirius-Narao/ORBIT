"""
ORBIT - Linear Layer Unit Tests
================================

Tests for nn.layers.linear.Linear, previously only covered indirectly via
tests/integration/test_xor.py. Follows the same philosophy as
tests/core/test_autograd.py: hand-derive the expected math in a comment,
then assert np.allclose(...).
"""

import numpy as np

from orbit.core import Tensor
from orbit.nn.layers.linear import Linear


# ===========================================================================
# 1. Forward
# ===========================================================================

def test_linear_forward_computes_xw_plus_b():
    """
    x @ W + b for:
        x = [[1, 2]]
        W = [[1, 0, 2],
             [0, 1, 3]]
        b = [1, 1, 1]

    x @ W = [1*1+2*0, 1*0+2*1, 1*2+2*3] = [1, 2, 8]
    + b    = [2, 3, 9]
    """
    linear = Linear(2, 3)
    linear.weight.data = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    linear.bias.data = np.array([1.0, 1.0, 1.0])

    x = Tensor([[1.0, 2.0]])
    out = linear(x)

    assert np.allclose(out.data, [[2.0, 3.0, 9.0]])


def test_linear_bias_initialized_to_zero():
    linear = Linear(4, 5)
    assert np.allclose(linear.bias.data, np.zeros(5))


def test_linear_weight_uses_xavier_scale():
    """
    Xavier/Glorot init scales randn by sqrt(2 / (in_features + out_features)),
    so a large weight matrix's std should land close to that scale.
    """
    np.random.seed(0)
    in_features, out_features = 200, 100
    linear = Linear(in_features, out_features)

    expected_scale = np.sqrt(2.0 / (in_features + out_features))
    assert np.isclose(linear.weight.data.std(), expected_scale, rtol=0.1)


def test_linear_registers_weight_and_bias_as_parameters():
    linear = Linear(2, 3)
    params = linear.parameters()

    assert params == [linear.weight, linear.bias]
    assert linear.weight.requires_grad
    assert linear.bias.requires_grad


# ===========================================================================
# 2. Backward
# ===========================================================================

def test_linear_backward_computes_expected_gradients():
    """
    x = [[1, 1]], W = [[1, 2], [3, 4]], b = [0.5, 0.5]
    out = x @ W + b, loss = sum(out)
    dout = [[1, 1]]  (gradient of sum wrt each output element)

    dW = x.T @ dout = [[1], [1]] @ [[1, 1]] = [[1, 1], [1, 1]]
    db = dout.sum(axis=0) = [1, 1]
    dx = dout @ W.T = [[1, 1]] @ [[1, 3], [2, 4]] = [1*1+1*2, 1*3+1*4] = [3, 7]
    """
    linear = Linear(2, 2)
    linear.weight.data = np.array([[1.0, 2.0], [3.0, 4.0]])
    linear.bias.data = np.array([0.5, 0.5])

    x = Tensor([[1.0, 1.0]], requires_grad=True)
    out = linear(x)
    loss = out.sum()
    loss.backward()

    assert np.allclose(linear.weight.grad, [[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(linear.bias.grad, [1.0, 1.0])
    assert np.allclose(x.grad, [[3.0, 7.0]])
