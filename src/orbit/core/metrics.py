import numpy as np
from orbit.core.tensor import Tensor


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


if __name__ == "__main__":
    y_pred = Tensor([[0.9], [0.2], [0.6], [0.4]])
    y_true = Tensor([[1.0], [0.0], [1.0], [1.0]])
    print(accuracy(y_pred, y_true))
