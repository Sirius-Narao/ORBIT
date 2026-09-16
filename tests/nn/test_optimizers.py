import numpy as np
import pytest
from orbit.nn import Parameter
from orbit.nn.optimizers import SGD


def test_sgd_updates_parameter():
    """
    Test that SGD.step() correctly updates a parameter using the gradient.

    SGD rule:
        param.data -= lr * param.grad

    With param.data = [1.0, 2.0], param.grad = [0.1, 0.1], lr = 0.1:
        new param.data = [1.0 - 0.1*0.1, 2.0 - 0.1*0.1]
                       = [0.99, 1.99]
    """
    param = Parameter(np.array([1.0, 2.0]))
    param.grad = np.array([0.1, 0.1])

    optimizer = SGD([param], lr=0.1)
    optimizer.step()

    expected = np.array([0.99, 1.99])
    assert np.allclose(param.data, expected)


def test_zero_grad_clears_gradients():
    """
    Test that zero_grad() sets all parameter gradients to None.
    """
    param1 = Parameter(np.array([1.0, 2.0]))
    param2 = Parameter(np.array([3.0, 4.0]))

    # Manually set gradients
    param1.grad = np.array([0.5, 0.5])
    param2.grad = np.array([0.2, 0.2])

    assert param1.grad is not None
    assert param2.grad is not None

    optimizer = SGD([param1, param2], lr=0.01)
    optimizer.zero_grad()

    assert param1.grad is None
    assert param2.grad is None


def test_sgd_skips_none_gradients():
    """
    Test that SGD.step() safely skips parameters with grad=None.
    """
    param1 = Parameter(np.array([1.0]))
    param2 = Parameter(np.array([2.0]))

    param1.grad = np.array([0.1])
    param2.grad = None  # No gradient

    optimizer = SGD([param1, param2], lr=0.1)

    initial_param2 = param2.data.copy()
    optimizer.step()

    # param1 should update
    assert np.allclose(param1.data, [0.99])

    # param2 should remain unchanged
    assert np.allclose(param2.data, initial_param2)


def test_sgd_multiple_steps():
    """
    Test that repeated SGD.step() calls accumulate updates correctly.

    With param = 1.0, grad = 0.1, lr = 0.1:
        After step 1: param = 1.0 - 0.1*0.1 = 0.99
        After step 2: param = 0.99 - 0.1*0.1 = 0.98
    """
    param = Parameter(np.array([1.0]))
    optimizer = SGD([param], lr=0.1)

    param.grad = np.array([0.1])
    optimizer.step()
    assert np.allclose(param.data, [0.99])

    param.grad = np.array([0.1])
    optimizer.step()
    assert np.allclose(param.data, [0.98])
