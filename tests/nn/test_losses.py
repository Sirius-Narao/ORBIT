"""
ORBIT - Loss Unit Tests
========================

Tests for nn.losses.mse.MSE and nn.losses.cross_entropy.CrossEntropy,
previously only covered indirectly via tests/integration/test_xor.py.
Follows the same philosophy as tests/core/test_autograd.py: hand-derive the
expected math in a comment, then assert np.allclose(...).
"""

import numpy as np

from orbit.core import Tensor
from orbit.nn.losses.mse import MSE
from orbit.nn.losses.cross_entropy import CrossEntropy


# ===========================================================================
# 1. MSE
# ===========================================================================

def test_mse_forward_computes_mean_squared_error():
    """
    y_pred = [1, 2, 3], y_true = [1, 2, 5]
    squared diffs = [0, 0, 4], mean = 4/3
    """
    mse = MSE()
    y_pred = Tensor([1.0, 2.0, 3.0])
    y_true = Tensor([1.0, 2.0, 5.0])

    loss = mse(y_pred, y_true)

    assert np.isclose(loss.data, 4.0 / 3.0)


def test_mse_backward_gradients():
    """
    d(mean((p - t)^2))/dp = 2(p - t) / n
    p = [1, 2, 3], t = [1, 2, 5], n = 3
    -> 2 * [0, 0, -2] / 3 = [0, 0, -4/3]
    """
    mse = MSE()
    y_pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y_true = Tensor([1.0, 2.0, 5.0])

    loss = mse(y_pred, y_true)
    loss.backward()

    assert np.allclose(y_pred.grad, [0.0, 0.0, -4.0 / 3.0])


def test_mse_name_attribute():
    assert MSE().name == "mse"


# ===========================================================================
# 2. CrossEntropy
# ===========================================================================

def test_cross_entropy_forward_single_sample():
    """
    logits = [1, 2, 3], true class index = 2
    softmax(logits) = exp(logits - max) / sum(exp(logits - max))
    loss = -log(probs[2])
    """
    ce = CrossEntropy()
    logits = Tensor([[1.0, 2.0, 3.0]])
    y_true = Tensor([2])

    loss = ce(logits, y_true)

    shifted = np.array([1.0, 2.0, 3.0]) - 3.0
    exp = np.exp(shifted)
    probs = exp / exp.sum()
    expected = -np.log(probs[2])

    assert np.isclose(loss.data, expected)


def test_cross_entropy_forward_batch():
    """
    Two samples, true classes 2 and 0. Loss is the mean of each sample's
    -log(true class probability).
    """
    ce = CrossEntropy()
    logits_data = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    logits = Tensor(logits_data)
    y_true = Tensor([2, 0])

    loss = ce(logits, y_true)

    shifted = logits_data - logits_data.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    expected = np.mean([-np.log(probs[0, 2]), -np.log(probs[1, 0])])

    assert np.isclose(loss.data, expected)


def test_cross_entropy_backward_matches_softmax_minus_onehot_over_n():
    """
    Fused softmax + CCE gradient: dL/dlogits = (probs - one_hot) / N
    """
    ce = CrossEntropy()
    logits_data = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    logits = Tensor(logits_data, requires_grad=True)
    y_true = Tensor([2, 0])

    loss = ce(logits, y_true)
    loss.backward()

    shifted = logits_data - logits_data.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=-1, keepdims=True)
    one_hot = np.zeros_like(probs)
    one_hot[0, 2] = 1.0
    one_hot[1, 0] = 1.0
    expected_grad = (probs - one_hot) / 2

    assert np.allclose(logits.grad, expected_grad)


def test_cross_entropy_graph_metadata():
    """
    y_true carries no gradient, so it must not appear as a parent - only
    y_pred (the logits) does.
    """
    ce = CrossEntropy()
    logits = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y_true = Tensor([1])

    out = ce(logits, y_true)

    assert out.operation == "cross_entropy"
    assert out.parents == [logits]


def test_cross_entropy_name_attribute():
    assert CrossEntropy().name == "cross_entropy"
