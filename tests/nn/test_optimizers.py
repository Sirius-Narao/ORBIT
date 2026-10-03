import numpy as np
import pytest
from orbit.nn import Parameter
from orbit.nn.optimizers import SGD, Adam


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


def test_sgd_momentum_accumulates_velocity():
    """
    Momentum rule (PyTorch convention):
        v     = momentum * v + grad
        param = param - lr * v

    With param = 1.0, grad = 0.1 both steps, lr = 0.1, momentum = 0.9:
        Step 1: v = 0.9*0   + 0.1 = 0.1   -> param = 1.0  - 0.1*0.1  = 0.99
        Step 2: v = 0.9*0.1 + 0.1 = 0.19  -> param = 0.99 - 0.1*0.19 = 0.971
    """
    param = Parameter(np.array([1.0]))
    optimizer = SGD([param], lr=0.1, momentum=0.9)

    param.grad = np.array([0.1])
    optimizer.step()
    assert np.allclose(param.data, [0.99])

    param.grad = np.array([0.1])
    optimizer.step()
    assert np.allclose(param.data, [0.971])


def test_sgd_zero_momentum_matches_plain_sgd_exactly():
    grads = [np.array([0.3, -0.7]), np.array([-0.2, 0.05]), np.array([1.1, 0.4])]

    plain = Parameter(np.array([1.0, 2.0]))
    with_zero = Parameter(np.array([1.0, 2.0]))
    opt_plain = SGD([plain], lr=0.1)
    opt_zero = SGD([with_zero], lr=0.1, momentum=0.0)

    for g in grads:
        plain.grad = g.copy()
        with_zero.grad = g.copy()
        opt_plain.step()
        opt_zero.step()

    assert np.array_equal(plain.data, with_zero.data)


def test_adam_first_two_steps():
    """
    Adam with lr = 0.1, betas = (0.9, 0.999), param = 1.0, grad = 0.1:

    Step 1:
        m = 0.1 * 0.1 = 0.01          -> m_hat = 0.01 / (1 - 0.9)    = 0.1
        v = 0.001 * 0.01 = 1e-5       -> v_hat = 1e-5 / (1 - 0.999)  = 0.01
        param = 1.0 - 0.1 * 0.1 / (sqrt(0.01) + eps) ~= 1.0 - 0.1 = 0.9

    Step 2 (same grad): m and v are still unbiased estimates of g and g^2
    after bias correction (m_hat = 0.1, v_hat = 0.01), so the step is
    again ~lr:
        param ~= 0.9 - 0.1 = 0.8
    """
    param = Parameter(np.array([1.0]))
    optimizer = Adam([param], lr=0.1)

    param.grad = np.array([0.1])
    optimizer.step()
    assert np.allclose(param.data, [0.9])

    param.grad = np.array([0.1])
    optimizer.step()
    assert np.allclose(param.data, [0.8])


def test_adam_step_size_is_independent_of_gradient_scale():
    """
    m_hat / sqrt(v_hat) = g / |g| on the first step, so a tiny and a huge
    gradient move their weights by the same ~lr.
    """
    small = Parameter(np.array([0.0]))
    large = Parameter(np.array([0.0]))
    small.grad = np.array([1e-3])
    large.grad = np.array([1e3])

    Adam([small, large], lr=0.01).step()

    assert np.allclose(small.data, [-0.01])
    assert np.allclose(large.data, [-0.01])


def test_adam_skips_none_gradients_without_advancing_their_step_count():
    param1 = Parameter(np.array([1.0]))
    param2 = Parameter(np.array([1.0]))
    optimizer = Adam([param1, param2], lr=0.1)

    param1.grad = np.array([0.1])
    param2.grad = None
    optimizer.step()

    assert np.allclose(param2.data, [1.0])
    assert optimizer.t == {0: 1}

    # param2's first real step still gets full (t=1) bias correction,
    # so it moves by ~lr, same as param1's first step did.
    param1.grad = None
    param2.grad = np.array([0.1])
    optimizer.step()

    assert np.allclose(param2.data, [0.9])
    assert optimizer.t == {0: 1, 1: 1}


def test_adam_zero_grad_clears_gradients():
    param = Parameter(np.array([1.0]))
    param.grad = np.array([0.5])

    Adam([param]).zero_grad()

    assert param.grad is None
