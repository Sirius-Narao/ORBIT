import numpy as np
import pytest
from orbit.core.tensor import Tensor
from orbit.core.dataset import TensorDataset


def test_len_returns_row_count():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
    Y = np.array([[0], [1], [1], [0], [1]])

    dataset = TensorDataset(X, Y)

    assert len(dataset) == 5


def test_getitem_returns_matching_row():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    dataset = TensorDataset(X, Y)
    x, y = dataset[2]

    assert np.array_equal(x, X[2])
    assert np.array_equal(y, Y[2])


def test_mismatched_lengths_raises():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])  # 5 rows
    Y = np.array([[0], [1], [1], [0]])  # 4 rows

    with pytest.raises(ValueError):
        TensorDataset(X, Y)


def test_accepts_tensor_input():
    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = Tensor([[0], [1], [1], [0]])

    dataset = TensorDataset(X, Y)
    x, y = dataset[1]

    assert len(dataset) == 4
    assert np.array_equal(x, X.data[1])
    assert np.array_equal(y, Y.data[1])
