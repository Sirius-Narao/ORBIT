from orbit.core import Tensor, accuracy, accuracy_multiclass


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


# --- regression metrics ------------------------------------------------------

import numpy as np

from orbit.core import regression_tolerance, r2_score, metric_label, format_metric


def test_regression_tolerance_counts_predictions_within_tolerance():
    """
    |pred - true| = [0.2, 0.5, 0.7, 0.0] with tolerance 0.5: the first,
    second (exactly on the boundary - counts) and fourth are correct -> 3/4.
    """
    y_pred = Tensor([[5.2], [6.5], [2.3], [4.0]])
    y_true = Tensor([[5.0], [6.0], [3.0], [4.0]])

    assert regression_tolerance(y_pred, y_true, tolerance=0.5) == 0.75


def test_regression_tolerance_requires_every_output_within_tolerance():
    # Row 0: both outputs within 1.0 -> correct. Row 1: second output off by 2 -> wrong.
    y_pred = Tensor([[1.0, 1.0], [1.0, 3.0]])
    y_true = Tensor([[1.5, 0.5], [1.0, 1.0]])

    assert regression_tolerance(y_pred, y_true, tolerance=1.0) == 0.5


def test_r2_perfect_fit_is_one():
    y = Tensor([[1.0], [2.0], [3.0]])

    assert np.isclose(r2_score(y, y), 1.0)


def test_r2_predicting_the_mean_is_zero():
    # mean(y_true) = 2, so SS_res == SS_tot -> R^2 = 0.
    y_pred = Tensor([[2.0], [2.0], [2.0]])
    y_true = Tensor([[1.0], [2.0], [3.0]])

    assert np.isclose(r2_score(y_pred, y_true), 0.0)


def test_r2_worse_than_the_mean_is_negative():
    """
    y_true = [1, 2, 3], y_pred = [3, 2, 1]:
    SS_res = 4 + 0 + 4 = 8, SS_tot = 1 + 0 + 1 = 2 -> R^2 = 1 - 8/2 = -3.
    """
    y_pred = Tensor([[3.0], [2.0], [1.0]])
    y_true = Tensor([[1.0], [2.0], [3.0]])

    assert np.isclose(r2_score(y_pred, y_true), -3.0)


def test_r2_constant_target_does_not_divide_by_zero():
    y_true = Tensor([[5.0], [5.0]])

    assert r2_score(Tensor([[5.0], [5.0]]), y_true) == 1.0
    assert r2_score(Tensor([[4.0], [6.0]]), y_true) == 0.0


def test_r2_averages_over_output_columns():
    # Column 0 is a perfect fit (R^2 = 1), column 1 predicts its mean (R^2 = 0) -> 0.5.
    y_true = Tensor([[1.0, 1.0], [3.0, 3.0]])
    y_pred = Tensor([[1.0, 2.0], [3.0, 2.0]])

    assert np.isclose(r2_score(y_pred, y_true), 0.5)


def test_metric_label_and_format_by_task():
    assert metric_label("regression_r2") == "R²"
    assert format_metric(0.91234, "regression_r2") == "0.9123"
    assert format_metric(-1.5, "regression_r2") == "-1.5000"

    for task in ("binary_classification", "regression_tolerance", None):
        assert metric_label(task) == "Accuracy"
        assert format_metric(0.875, task) == "87.50%"
