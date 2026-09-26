import numpy as np
from orbit.core import Tensor


def accuracy(y_pred: Tensor, y_true: Tensor, threshold: float = 0.5) -> float:
    """
    Binary classification accuracy: fraction of predictions on the correct
    side of `threshold` that match y_true.
    """
    predictions = (y_pred.data > threshold).astype(float)
    return float(np.mean(predictions == y_true.data))


def accuracy_multiclass(y_pred: Tensor, y_true: Tensor) -> float:
    """
    Multiclass accuracy: fraction of samples where the predicted class
    (argmax over the last axis of y_pred) matches the true class index.
    """
    predictions = np.argmax(y_pred.data, axis=-1)
    labels = y_true.data.astype(int)
    return float(np.mean(predictions == labels))


def regression_tolerance(y_pred: Tensor, y_true: Tensor, tolerance: float = 0.5) -> float:
    """
    Regression "accuracy": fraction of samples whose every output is within
    `tolerance` of the target, i.e. |y_pred - y_true| <= tolerance. With
    integer-valued targets (e.g. a 3-9 wine-quality score) and the default
    0.5, "correct" means the prediction rounds to the right score.
    """
    within = np.abs(y_pred.data - y_true.data) <= tolerance
    return float(np.mean(np.all(within, axis=-1)))


def r2_score(y_pred: Tensor, y_true: Tensor) -> float:
    """
    Coefficient of determination, per output column, averaged over columns:

        R^2 = 1 - SS_res / SS_tot
        SS_res = sum((y_true - y_pred)^2)          (the model's squared error)
        SS_tot = sum((y_true - mean(y_true))^2)    (error of always predicting the mean)

    So 1 is a perfect fit, 0 is no better than predicting the mean, and a
    negative value is worse than that. Not a per-sample average: it must be
    computed over a whole epoch / evaluation pass at once (see Trainer), not
    averaged across batches.

    A constant-target column has SS_tot = 0; it scores 1.0 if the predictions
    are exact and 0.0 otherwise (same convention as scikit-learn), instead of
    dividing by zero.
    """
    pred = np.asarray(y_pred.data, dtype=float).reshape(len(y_pred.data), -1)
    true = np.asarray(y_true.data, dtype=float).reshape(len(y_true.data), -1)

    ss_res = np.sum((true - pred) ** 2, axis=0)
    ss_tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0)

    safe_tot = np.where(ss_tot == 0, 1.0, ss_tot)
    per_column = np.where(ss_tot == 0, np.where(ss_res == 0, 1.0, 0.0), 1.0 - ss_res / safe_tot)
    return float(np.mean(per_column))


# Tasks whose metric is not a 0-1 "fraction correct" and so must not be shown
# as a percentage. Everything else (including an unknown/absent task, e.g. an
# old results.json) is displayed as accuracy.
_METRIC_LABELS = {"regression_r2": "R²"}


def metric_label(task) -> str:
    return _METRIC_LABELS.get(task, "Accuracy")


def format_metric(value: float, task) -> str:
    if task in _METRIC_LABELS:
        return f"{value:.4f}"
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    y_pred = Tensor([[0.9], [0.2], [0.6], [0.4]])
    y_true = Tensor([[1.0], [0.0], [1.0], [1.0]])
    print(accuracy(y_pred, y_true))
