"""
ORBIT - Autograd Tests
======================

Tests for the v0.1 automatic differentiation system.

Run this file from the ORBIT project root with:

    pytest tests/core/test_autograd.py -v

The philosophy of these tests is simple:

    1. Create Tensors.
    2. Perform an operation.
    3. Call backward().
    4. Check that the gradients are what mathematics predicts.

These tests are intentionally small. If one fails, we want to be able
to understand WHY it failed instead of staring at a 500-line test.
"""

import numpy as np

from orbit.core import Tensor


# ===========================================================================
# 1. Basic addition
# ===========================================================================

def test_addition_backward():
    """
    Test:

        z = x + y

    Mathematics:

        dz/dx = 1
        dz/dy = 1

    Therefore, after:

        z.backward()

    both gradients should be 1.
    """

    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x + y

    # Forward result first.
    assert np.allclose(z.data, 5.0)

    # Start the backward pass.
    z.backward()

    # The derivative of x + y with respect to both variables is 1.
    assert np.allclose(x.grad, 1.0)
    assert np.allclose(y.grad, 1.0)


# ===========================================================================
# 2. Basic multiplication
# ===========================================================================

def test_multiplication_backward():
    """
    Test:

        z = x * y

    Mathematics:

        dz/dx = y
        dz/dy = x

    With x = 2 and y = 3:

        dz/dx = 3
        dz/dy = 2
    """

    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x * y

    assert np.allclose(z.data, 6.0)

    z.backward()

    assert np.allclose(x.grad, 3.0)
    assert np.allclose(y.grad, 2.0)


# ===========================================================================
# 3. Subtraction
# ===========================================================================

def test_subtraction_backward():
    """
    Test:

        z = x - y

    Mathematics:

        dz/dx =  1
        dz/dy = -1
    """

    x = Tensor(5.0, requires_grad=True)
    y = Tensor(2.0, requires_grad=True)

    z = x - y

    assert np.allclose(z.data, 3.0)

    z.backward()

    assert np.allclose(x.grad, 1.0)
    assert np.allclose(y.grad, -1.0)


# ===========================================================================
# 4. Division
# ===========================================================================

def test_division_backward():
    """
    Test:

        z = x / y

    Mathematics:

        dz/dx = 1 / y

        dz/dy = -x / y²

    With x = 6 and y = 2:

        dz/dx = 1/2
        dz/dy = -6/4 = -1.5
    """

    x = Tensor(6.0, requires_grad=True)
    y = Tensor(2.0, requires_grad=True)

    z = x / y

    assert np.allclose(z.data, 3.0)

    z.backward()

    assert np.allclose(x.grad, 0.5)
    assert np.allclose(y.grad, -1.5)


# ===========================================================================
# 5. Chain rule
# ===========================================================================

def test_chain_rule():
    """
    Test a small computational graph:

        z = x * y
        loss = z + x

    Therefore:

        loss = xy + x

    Derivatives:

        dloss/dx = y + 1
        dloss/dy = x

    With x = 2 and y = 3:

        dloss/dx = 4
        dloss/dy = 2

    This test is important because it checks that ORBIT is not only
    calculating individual derivatives. It must also propagate them
    through multiple operations using the chain rule.
    """

    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x * y
    loss = z + x

    assert np.allclose(loss.data, 8.0)

    loss.backward()

    assert np.allclose(x.grad, 4.0)
    assert np.allclose(y.grad, 2.0)


# ===========================================================================
# 6. Gradient accumulation
# ===========================================================================

def test_gradient_accumulation():
    """
    Test:

        y = x * x

    Here x is used twice.

    Mathematics:

        y = x²

        dy/dx = 2x

    With x = 3:

        dy/dx = 6

    Why is this test important?

    The graph contains TWO paths from x into the multiplication node:

            x
           / \\
          /   \\
         *     <-- x * x
          \\
           y

    ORBIT must add the two gradient contributions instead of allowing
    one to overwrite the other.
    """

    x = Tensor(3.0, requires_grad=True)

    y = x * x

    assert np.allclose(y.data, 9.0)

    y.backward()

    assert np.allclose(x.grad, 6.0)


# ===========================================================================
# 7. Tensor that does not require gradients
# ===========================================================================

def test_requires_grad_false():
    """
    A Tensor with:

        requires_grad=False

    should not accumulate a gradient.

    This is important because inputs do not always need to be
    differentiated. In a neural network, for example, we normally care
    about gradients for the trainable parameters.
    """

    x = Tensor(2.0, requires_grad=False)
    y = Tensor(3.0, requires_grad=True)

    z = x * y

    z.backward()

    assert x.grad is None
    assert np.allclose(y.grad, 2.0)


# ===========================================================================
# 8. Matrix multiplication
# ===========================================================================

def test_matrix_multiplication_backward():
    """
    Test:

        Y = X @ W

    We use very small matrices so the expected gradients can be
    calculated by hand.

        X = [[1, 2]]
        W = [[3],
             [4]]

        Y = [[11]]

    For matrix multiplication:

        dL/dX = dL/dY @ W.T

        dL/dW = X.T @ dL/dY

    Because Y is a scalar-shaped (1, 1) result, dL/dY starts as 1.

    Expected:

        dL/dX = [[3, 4]]

        dL/dW = [[1],
                 [2]]
    """

    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[3.0], [4.0]], requires_grad=True)

    y = x @ w

    assert np.allclose(y.data, [[11.0]])

    y.backward()

    assert np.allclose(x.grad, [[3.0, 4.0]])
    assert np.allclose(w.grad, [[1.0], [2.0]])


# ===========================================================================
# 9. Sum backward
# ===========================================================================

def test_sum_backward():
    """
    Test:

        y = x.sum()

    For:

        x = [1, 2, 3]

    we have:

        y = 6

    Every element contributes equally to the sum, so:

        dy/dx = [1, 1, 1]
    """

    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = x.sum()

    assert np.allclose(y.data, 6.0)

    y.backward()

    assert np.allclose(x.grad, [1.0, 1.0, 1.0])


# ===========================================================================
# 10. Mean backward
# ===========================================================================

def test_mean_backward():
    """
    Test:

        y = mean(x)

    For:

        x = [1, 2, 3]

    we have:

        y = 2

    Since the mean divides by 3, every element receives:

        dy/dx_i = 1/3
    """

    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = x.mean()

    assert np.allclose(y.data, 2.0)

    y.backward()

    assert np.allclose(x.grad, [1 / 3, 1 / 3, 1 / 3])


# ===========================================================================
# 11. A slightly larger computational graph
# ===========================================================================

def test_multiple_operations():
    """
    Test a graph containing several operations:

        a = x * y
        b = a + x
        c = b * y

    With:

        x = 2
        y = 3

    Forward:

        a = 6
        b = 8
        c = 24

    Algebraically:

        c = (xy + x)y
          = xy² + xy

    Therefore:

        dc/dx = y² + y
              = 9 + 3
              = 12

        dc/dy = 2xy + x
              = 12 + 2
              = 14
    """

    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    a = x * y
    b = a + x
    c = b * y

    assert np.allclose(c.data, 24.0)

    c.backward()

    assert np.allclose(x.grad, 12.0)
    assert np.allclose(y.grad, 14.0)


# ===========================================================================
# 12. Negation backward
# ===========================================================================

def test_negation_backward():
    """
    Test:

        z = -x

    Mathematics:

        dz/dx = -1
    """

    x = Tensor(4.0, requires_grad=True)
    z = -x

    assert np.allclose(z.data, -4.0)

    z.backward()

    assert np.allclose(x.grad, -1.0)


# ===========================================================================
# 13. Broadcasting: bias vector added to a batch
# ===========================================================================

def test_add_broadcast_bias_backward():
    """
    Test:

        result = batch + bias

    where:

        batch = [[1, 2],
                 [3, 4],
                 [5, 6]]     shape (3, 2)

        bias  = [10, 20]     shape (2,)

    NumPy broadcasts `bias` across the 3 rows of `batch` during the
    forward pass. Every one of those 3 broadcasted copies of `bias`
    contributes to the result, so during backward() their gradients must
    be SUMMED to produce a single gradient with bias's own shape (2,).

    With an upstream gradient of all-ones (from `.sum()`):

        dbatch = [[1, 1],
                  [1, 1],
                  [1, 1]]     shape (3, 2) -- same shape as batch

        dbias  = [3, 3]       shape (2,)  -- summed over the 3 rows
    """

    batch = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    bias = Tensor([10.0, 20.0], requires_grad=True)

    result = batch + bias

    assert np.allclose(result.data, [[11.0, 22.0], [13.0, 24.0], [15.0, 26.0]])

    result.sum().backward()

    assert batch.grad.shape == (3, 2)
    assert np.allclose(batch.grad, [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    assert bias.grad.shape == (2,)
    assert np.allclose(bias.grad, [3.0, 3.0])
