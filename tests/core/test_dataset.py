import numpy as np
import pytest
from orbit.core import Tensor, TensorDataset
from orbit.core.dataset import Dataset


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


def test_input_shape_and_output_shape_report_feature_counts():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 rows, 2 features
    Y = np.array([[0], [1], [1], [0]])               # 4 rows, 1 feature

    dataset = TensorDataset(X, Y)

    assert dataset.input_shape == 2
    assert dataset.output_shape == 1


def test_input_shape_and_output_shape_with_multiple_output_features():
    X = np.array([[0, 0, 0], [1, 1, 1]])       # 2 rows, 3 features
    Y = np.array([[0, 1], [1, 0]])             # 2 rows, 2 features

    dataset = TensorDataset(X, Y)

    assert dataset.input_shape == 3
    assert dataset.output_shape == 2


def test_dataset_base_class_shape_properties_are_not_implemented():
    dataset = Dataset()

    with pytest.raises(NotImplementedError):
        dataset.input_shape

    with pytest.raises(NotImplementedError):
        dataset.output_shape
