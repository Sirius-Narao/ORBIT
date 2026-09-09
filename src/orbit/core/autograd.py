"""
ORBIT - Automatic Differentiation
=================================

v0.1

This module contains the small "engine" responsible for running the
backward pass of ORBIT's automatic differentiation system.

IMPORTANT:
----------
This file is intentionally simple.

At this stage, the goal is NOT to build something as sophisticated as
PyTorch's autograd engine. The goal is to understand the core idea:

    forward pass
        ↓
    computational graph
        ↓
    backward pass
        ↓
    gradients

The Tensor class is still responsible for storing numerical data and
information about how a Tensor was created.

This module is responsible for the algorithm that walks through that
graph backwards.

The first version supports the basic operations that ORBIT needs to
learn the foundations of backpropagation:

    +   addition
    -   subtraction
    *   multiplication
    /   division
    @   matrix multiplication

More complicated operations can be added later.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Set, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # This import is only used by type checkers.
    # It does NOT create a runtime circular import.
    from .tensor import Tensor


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A backward function receives the gradient flowing FROM the child
# Tensor and uses it to calculate gradients for the Tensor's parents.
#
# Example:
#
#       x ----\
#              multiply ----> z
#       y ----/
#
# During backward(), multiply receives dz and calculates:
#
#       dx = dz * y
#       dy = dz * x
#
BackwardFunction = Callable[["Tensor"], None]


# ---------------------------------------------------------------------------
# Small utility
# ---------------------------------------------------------------------------

def accumulate_gradient(tensor: "Tensor", gradient: np.ndarray) -> None:
    """
    Add `gradient` to a Tensor's existing gradient.

    Why do we ADD instead of simply assigning?

    Because a Tensor can influence the final result through multiple
    paths in the computational graph.

    Example:

        y = x * x

    Here `x` appears twice.

    Both uses of x contribute to dy/dx, so ORBIT must accumulate both
    contributions.

    This is one of the most important ideas in reverse-mode
    automatic differentiation.
    """

    if not tensor.requires_grad:
        # If the user did not ask for gradients for this Tensor,
        # there is nothing to store.
        return

    if tensor.grad is None:
        # First gradient contribution.
        tensor.grad = np.array(gradient, copy=True)
    else:
        # Another path through the graph contributed a gradient.
        tensor.grad += gradient


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

def build_topological_order(root: "Tensor") -> List["Tensor"]:
    """
    Return the computational graph in topological order.

    A computational graph is a DAG:

        Directed
        Acyclic
        Graph

    If Tensor B was created using Tensor A, then:

        A -> B

    B must be processed before A during the backward pass.

    We therefore:

        1. Start at the final Tensor.
        2. Recursively visit its parents.
        3. Add each Tensor after its parents have been visited.

    The result goes from "oldest" dependencies to the final result.

    `backward()` can then simply traverse this list in reverse.

    Example graph:

        x -> a -> b -> loss

    Topological order:

        [x, a, b, loss]

    Backward order:

        [loss, b, a, x]
    """

    visited: Set[int] = set()
    order: List["Tensor"] = []

    def visit(tensor: "Tensor") -> None:
        # id(tensor) gives us a unique identity for this particular
        # Tensor object during this traversal.
        if id(tensor) in visited:
            return

        visited.add(id(tensor))

        # First visit everything that this Tensor depends on.
        if tensor.parents:
            for parent in tensor.parents:
                visit(parent)

        # Only after its parents have been visited do we add the Tensor.
        order.append(tensor)

    visit(root)

    return order


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def backward(root: "Tensor", gradient: np.ndarray | None = None) -> None:
    """
    Run reverse-mode automatic differentiation from `root`.

    Normally the user will call:

        loss.backward()

    rather than calling this function directly.

    The important idea is that the final output starts with:

        dL/dL = 1

    If L is the loss, its derivative with respect to itself is 1.

    From there, each operation applies the chain rule:

        dL/dx = dL/dy * dy/dx

    where:

        dL/dy
            is the gradient coming FROM the operation's output

        dy/dx
            is the local derivative of that operation

        dL/dx
            is the gradient we send TO the parent Tensor.

    Parameters
    ----------
    root:
        The final Tensor from which differentiation starts.

    gradient:
        Optional initial gradient.

        For a scalar loss, this is normally simply 1.
    """

    # -----------------------------------------------------------------------
    # 1. Make sure we have an initial gradient.
    # -----------------------------------------------------------------------

    if gradient is None:
        # For a scalar output:
        #
        #       dL/dL = 1
        #
        # This is the starting point of the entire backward pass.
        if root.data.size != 1:
            raise ValueError(
                "backward() requires an initial gradient for "
                "non-scalar tensors."
            )

        gradient = np.ones_like(root.data)

    else:
        gradient = np.asarray(gradient)

    # -----------------------------------------------------------------------
    # 2. Build the graph order.
    # -----------------------------------------------------------------------

    # We need to know in which order the graph should be traversed.
    # Reverse topological order is exactly what reverse-mode
    # automatic differentiation needs.
    nodes = build_topological_order(root)

    # -----------------------------------------------------------------------
    # 3. Give the root its initial gradient.
    # -----------------------------------------------------------------------

    # This is the first gradient contribution.
    #
    # If root is L:
    #
    #       root.grad = dL/dL = 1
    #
    if root.requires_grad:
        root.grad = np.array(gradient, copy=True)

    # -----------------------------------------------------------------------
    # 4. Walk backwards through the graph.
    # -----------------------------------------------------------------------

    for node in reversed(nodes):
        # Every operation-created Tensor can optionally contain a
        # backward function.
        #
        # Leaf tensors, such as:
        #
        #       x = Tensor(3, requires_grad=True)
        #
        # do not have an operation behind them, so there is no
        # backward function to execute.
        backward_function = getattr(node, "_backward", None)

        if backward_function is not None:
            # The operation's backward function is responsible for:
            #
            #   a) looking at node.grad
            #   b) calculating local derivatives
            #   c) sending gradients to node.parents
            #
            # This is where the chain rule is actually applied.
            backward_function(node)


# ---------------------------------------------------------------------------
# Backward rules
# ---------------------------------------------------------------------------
#
# The following helpers are small mathematical rules.
#
# They are kept here so the logic is easy to understand.
#
# Your Tensor operations can create a result Tensor and attach one of
# these rules to that result.
#
# Conceptually:
#
#       result._backward = ...
#
# Then:
#
#       result.backward()
#
# eventually executes the rule.
# ---------------------------------------------------------------------------


def backward_add(
    result: "Tensor",
    left: "Tensor",
    right: "Tensor",
) -> None:
    """
    Backward rule for:

        result = left + right

    Mathematics:

        d(result)/d(left)  = 1
        d(result)/d(right) = 1

    Therefore, if the upstream gradient is:

        dL/dresult

    then:

        dL/dleft  = dL/dresult
        dL/dright = dL/dresult

    Note:
    -----
    This v0.1 rule assumes compatible shapes without trying to solve
    the full broadcasting problem yet.
    """

    if result.grad is None:
        return

    accumulate_gradient(left, result.grad)
    accumulate_gradient(right, result.grad)


def backward_subtract(
    result: "Tensor",
    left: "Tensor",
    right: "Tensor",
) -> None:
    """
    Backward rule for:

        result = left - right

    Mathematics:

        d(result)/d(left)  =  1
        d(result)/d(right) = -1
    """

    if result.grad is None:
        return

    accumulate_gradient(left, result.grad)
    accumulate_gradient(right, -result.grad)


def backward_multiply(
    result: "Tensor",
    left: "Tensor",
    right: "Tensor",
) -> None:
    """
    Backward rule for:

        result = left * right

    Mathematics:

        d(result)/d(left)  = right
        d(result)/d(right) = left

    Then the chain rule gives:

        dL/dleft  = dL/dresult * right
        dL/dright = dL/dresult * left
    """

    if result.grad is None:
        return

    accumulate_gradient(left, result.grad * right.data)
    accumulate_gradient(right, result.grad * left.data)


def backward_divide(
    result: "Tensor",
    left: "Tensor",
    right: "Tensor",
) -> None:
    """
    Backward rule for:

        result = left / right

    Rewrite division as:

        left * (1 / right)

    The derivatives are:

        d(result)/d(left)  = 1 / right

        d(result)/d(right) = -left / right²

    Then apply the chain rule.
    """

    if result.grad is None:
        return

    accumulate_gradient(
        left,
        result.grad / right.data,
    )

    accumulate_gradient(
        right,
        result.grad * (-left.data / (right.data ** 2)),
    )


def backward_matmul(
    result: "Tensor",
    left: "Tensor",
    right: "Tensor",
) -> None:
    """
    Backward rule for matrix multiplication:

        result = left @ right

    For ordinary 2-D matrices:

        dL/dleft  = dL/dresult @ right.T

        dL/dright = left.T @ dL/dresult

    Example shapes:

        left   : (2, 3)
        right  : (3, 4)
        result : (2, 4)

    Therefore:

        dleft  : (2, 3)
        dright : (3, 4)

    This is the rule that will eventually allow Linear layers to
    learn their weights.
    """

    if result.grad is None:
        return

    accumulate_gradient(
        left,
        result.grad @ right.data.T,
    )

    accumulate_gradient(
        right,
        left.data.T @ result.grad,
    )


# ---------------------------------------------------------------------------
# Reduction rules
# ---------------------------------------------------------------------------

def backward_sum(
    result: "Tensor",
    parent: "Tensor",
) -> None:
    """
    Backward rule for:

        result = parent.sum()

    Every element contributes equally to the sum.

    Therefore, if:

        parent = [a, b, c]

    then:

        result = a + b + c

    and:

        d(result)/da = 1
        d(result)/db = 1
        d(result)/dc = 1

    So the incoming gradient is distributed to every element.
    """

    if result.grad is None:
        return

    gradient = np.ones_like(parent.data) * result.grad

    accumulate_gradient(parent, gradient)


def backward_mean(
    result: "Tensor",
    parent: "Tensor",
) -> None:
    """
    Backward rule for:

        result = parent.mean()

    A mean is:

        sum(parent) / number_of_elements

    Therefore each element receives:

        upstream_gradient / number_of_elements
    """

    if result.grad is None:
        return

    number_of_elements = parent.data.size

    gradient = (
        np.ones_like(parent.data)
        * result.grad
        / number_of_elements
    )

    accumulate_gradient(parent, gradient)
 
 
def backward_neg(
    result: "Tensor",
    parent: "Tensor",
) -> None:
    """
    Backward rule for:

        result = -parent

    Mathematics:

        d(result)/d(parent) = -1

    Therefore:

        dL/dparent = -dL/dresult
    """

    if result.grad is None:
        return

    accumulate_gradient(parent, -result.grad)