from orbit.core.tensor import Tensor
from orbit.core.metrics import accuracy, accuracy_multiclass


def test_accuracy_partial_match():
    """
    predictions (threshold 0.5) = [1, 0, 1, 0]
    y_true                      = [1, 0, 1, 1]
    3 of 4 match -> 0.75
    """
    y_pred = Tensor([[0.9], [0.2], [0.6], [0.4]])
    y_true = Tensor([[1.0], [0.0], [1.0], [1.0]])

    assert accuracy(y_pred, y_true) == 0.75


def test_accuracy_all_correct():
    y_pred = Tensor([[0.9], [0.1]])
    y_true = Tensor([[1.0], [0.0]])

    assert accuracy(y_pred, y_true) == 1.0


def test_accuracy_multiclass_partial_match():
    """
    argmax(y_pred) = [1, 0, 2]
    y_true         = [1, 0, 1]
    2 of 3 match -> 2/3
    """
    y_pred = Tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
    y_true = Tensor([1, 0, 1])

    assert accuracy_multiclass(y_pred, y_true) == 2 / 3
