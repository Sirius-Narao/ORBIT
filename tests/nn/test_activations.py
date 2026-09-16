"""
ORBIT - Activation Unit Tests
=============================

Tests for activation functions and their integration with ORBIT's autograd engine.
"""

import numpy as np
from orbit.core import Tensor
from orbit.nn.activations import ReLU, Sigmoid, Tanh, Softmax


# ===========================================================================
# 1. ReLU Tests
# ===========================================================================

def test_relu_forward():
    """
    Test ReLU forward calculation for positive, zero, and negative values.
    """
    relu = ReLU()
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 3.0], requires_grad=True)
    out = relu(x)

    assert np.allclose(out.data, [0.0, 0.0, 0.0, 1.0, 3.0])
    assert out.operation == "relu"
    assert out.parents == [x]


def test_relu_backward():
    """
    Test ReLU backward pass with scalar loss.

    Mathematics:
        d(ReLU(x))/dx = 1 if x > 0 else 0
    """
    relu = ReLU()
    x = Tensor([-2.0, -0.5, 0.0, 1.5, 3.0], requires_grad=True)
    out = relu(x)
    loss = out.sum()
    loss.backward()

    assert np.allclose(x.grad, [0.0, 0.0, 0.0, 1.0, 1.0])


def test_relu_in_graph():
    """
    Test ReLU in a combined computational graph:
        z = ReLU(x * w + b)
        loss = z.sum()
    """
    x = Tensor([[1.0, -2.0]], requires_grad=False)
    w = Tensor([[2.0], [3.0]], requires_grad=True)  # x @ w = 1*2 + (-2)*3 = -4
    b = Tensor([5.0], requires_grad=True)            # -4 + 5 = 1.0 (positive)

    linear_out = x @ w + b
    relu = ReLU()
    out = relu(linear_out)

    assert np.allclose(out.data, [[1.0]])

    loss = out.sum()
    loss.backward()

    assert np.allclose(b.grad, [1.0])
    assert np.allclose(w.grad, [[1.0], [-2.0]])


# ===========================================================================
# 2. Sigmoid Tests
# ===========================================================================

def test_sigmoid_forward():
    """
    Test Sigmoid forward calculation:
        Sigmoid(0) = 0.5
    """
    sigmoid = Sigmoid()
    x = Tensor([0.0], requires_grad=True)
    out = sigmoid(x)

    assert np.allclose(out.data, [0.5])
    assert out.operation == "sigmoid"


def test_sigmoid_backward():
    """
    Test Sigmoid backward pass:
        d(Sigmoid(x))/dx = Sigmoid(x) * (1 - Sigmoid(x))
    For x = 0:
        Sigmoid(0) = 0.5
        d(Sigmoid(0))/dx = 0.5 * 0.5 = 0.25
    """
    sigmoid = Sigmoid()
    x = Tensor([0.0], requires_grad=True)
    out = sigmoid(x)
    loss = out.sum()
    loss.backward()

    assert np.allclose(x.grad, [0.25])


# ===========================================================================
# 3. Tanh Tests
# ===========================================================================

def test_tanh_forward():
    """
    Test Tanh forward calculation:
        Tanh(0) = 0.0
    """
    tanh = Tanh()
    x = Tensor([0.0], requires_grad=True)
    out = tanh(x)

    assert np.allclose(out.data, [0.0])
    assert out.operation == "tanh"


def test_tanh_backward():
    """
    Test Tanh backward pass:
        d(Tanh(x))/dx = 1 - Tanh(x)^2
    For x = 0:
        Tanh(0) = 0.0
        d(Tanh(0))/dx = 1.0
    """
    tanh = Tanh()
    x = Tensor([0.0], requires_grad=True)
    out = tanh(x)
    loss = out.sum()
    loss.backward()

    assert np.allclose(x.grad, [1.0])


# ===========================================================================
# 4. Softmax Tests
# ===========================================================================

def test_softmax_forward():
    """
    Test Softmax forward calculation. Outputs must sum to 1.
    """
    softmax = Softmax(axis=-1)
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = softmax(x)

    exp_x = np.exp(np.array([1.0, 2.0, 3.0]) - 3.0)
    expected = exp_x / np.sum(exp_x)

    assert np.allclose(out.data, expected)
    assert np.allclose(out.data.sum(), 1.0)
    assert out.operation == "softmax"


def test_softmax_backward():
    """
    Test Softmax backward pass with custom gradient / loss.
    For Softmax followed by sum(): sum(Softmax(x)) is always 1.0,
    so d(sum(Softmax(x)))/dx should be 0 everywhere.
    """
    softmax = Softmax(axis=-1)
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    out = softmax(x)
    loss = out.sum()
    loss.backward()

    # Sum of softmax probabilities is constant 1, so gradient wrt inputs is 0
    assert np.allclose(x.grad, [0.0, 0.0, 0.0])


def test_softmax_2d_backward():
    """
    Test Softmax backward pass on a 2D batch tensor.
    """
    softmax = Softmax(axis=-1)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    out = softmax(x)
    
    # Scale first element of each batch row to create non-trivial loss
    loss = (out * Tensor([[1.0, 0.0], [0.0, 1.0]])).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 2)
