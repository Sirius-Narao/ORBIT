import numpy as np
import pytest
from orbit.core import Tensor, TensorDataset
from orbit.core.dataset import Dataset, Subset, train_test_split


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


def make_indexable_dataset(n_rows=10):
    X = np.arange(n_rows * 2).reshape(n_rows, 2)
    Y = np.arange(n_rows).reshape(n_rows, 1)
    return TensorDataset(X, Y), X, Y


def test_subset_len_matches_index_count():
    dataset, _, _ = make_indexable_dataset(10)
    subset = Subset(dataset, [0, 3, 7])

    assert len(subset) == 3


def test_subset_getitem_delegates_through_indices():
    dataset, X, Y = make_indexable_dataset(10)
    subset = Subset(dataset, [5, 1])

    x0, y0 = subset[0]
    x1, y1 = subset[1]

    assert np.array_equal(x0, X[5]) and np.array_equal(y0, Y[5])
    assert np.array_equal(x1, X[1]) and np.array_equal(y1, Y[1])


def test_subset_shape_properties_delegate_to_wrapped_dataset():
    dataset, _, _ = make_indexable_dataset(10)
    subset = Subset(dataset, [0, 1])

    assert subset.input_shape == dataset.input_shape
    assert subset.output_shape == dataset.output_shape


def test_train_test_split_sizes_match_rounded_ratio():
    """
    4-row dataset at the documented default 0.2 ratio must give 3 train
    rows, 1 test row - round(4*0.2)==1, matching CLAUDE.md's worked example
    (and NOT int(4*0.2)==0, which would wrongly empty the test set).
    """
    dataset, _, _ = make_indexable_dataset(4)

    train, test = train_test_split(dataset, 0.2)

    assert len(train) == 3
    assert len(test) == 1


def test_train_test_split_covers_every_row_with_no_overlap():
    dataset, X, _ = make_indexable_dataset(10)

    train, test = train_test_split(dataset, 0.3)

    train_rows = np.array([train[i][0] for i in range(len(train))])
    test_rows = np.array([test[i][0] for i in range(len(test))])
    all_rows = np.concatenate([train_rows, test_rows], axis=0)

    assert len(train) + len(test) == len(dataset)
    assert np.array_equal(np.sort(all_rows, axis=0), np.sort(X, axis=0))


def test_train_test_split_is_reproducible_with_same_seed():
    dataset, _, _ = make_indexable_dataset(10)

    np.random.seed(3)
    train_a, test_a = train_test_split(dataset, 0.3)
    np.random.seed(3)
    train_b, test_b = train_test_split(dataset, 0.3)

    assert list(train_a.indices) == list(train_b.indices)
    assert list(test_a.indices) == list(test_b.indices)


def test_train_test_split_raises_when_test_set_would_be_empty():
    dataset, _, _ = make_indexable_dataset(4)

    with pytest.raises(ValueError):
        train_test_split(dataset, 0.01)


def test_train_test_split_raises_when_train_set_would_be_empty():
    dataset, _, _ = make_indexable_dataset(4)

    with pytest.raises(ValueError):
        train_test_split(dataset, 0.99)


# --- normalization -----------------------------------------------------------

from orbit.core.dataset import NormalizedDataset, fit_normalizer, apply_normalizer


def test_standard_normalizer_centers_and_scales():
    # X = [1, 3]: mean = 2, population std = sqrt(((1-2)^2 + (3-2)^2) / 2) = 1
    # so (1 - 2) / 1 = -1 and (3 - 2) / 1 = 1.
    X = np.array([[1.0], [3.0]])

    stats = fit_normalizer(X, "standard")

    assert np.allclose(stats["mean"], [2.0])
    assert np.allclose(stats["std"], [1.0])
    assert np.allclose(apply_normalizer(X, stats), [[-1.0], [1.0]])


def test_minmax_normalizer_scales_to_unit_range():
    # X = [2, 4, 6]: min = 2, range = 4, so (x - 2) / 4 = [0, 0.5, 1].
    X = np.array([[2.0], [4.0], [6.0]])

    stats = fit_normalizer(X, "minmax")

    assert np.allclose(apply_normalizer(X, stats), [[0.0], [0.5], [1.0]])


def test_normalizers_scale_each_column_independently():
    X = np.array([[0.0, 100.0], [10.0, 300.0]])

    out = apply_normalizer(X, fit_normalizer(X, "minmax"))

    assert np.allclose(out, [[0.0, 0.0], [1.0, 1.0]])


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_constant_column_does_not_produce_nan(method):
    # std / range of a constant column is 0 - replaced by 1, so the column
    # maps to all zeros instead of 0/0 = NaN.
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])

    out = apply_normalizer(X, fit_normalizer(X, method))

    assert np.all(np.isfinite(out))
    assert np.allclose(out[:, 0], 0.0)


def test_fit_normalizer_unknown_method_raises():
    with pytest.raises(ValueError):
        fit_normalizer(np.array([[1.0]]), "bogus")


def test_normalized_dataset_rescales_x_and_leaves_y_untouched():
    X = np.array([[1.0], [3.0]])
    Y = np.array([[10.0], [20.0]])
    base = TensorDataset(X, Y)

    dataset = NormalizedDataset(base, fit_normalizer(X, "standard"))

    assert len(dataset) == 2
    assert dataset.input_shape == 1
    assert dataset.output_shape == 1
    x0, y0 = dataset[0]
    assert np.allclose(x0, [-1.0])
    assert np.allclose(y0, [10.0])


def test_normalized_dataset_works_over_a_subset():
    X = np.array([[1.0], [3.0], [100.0]])
    Y = np.array([[0.0], [1.0], [0.0]])
    subset = Subset(TensorDataset(X, Y), [0, 1])

    dataset = NormalizedDataset(subset, fit_normalizer(X[[0, 1]], "standard"))

    assert len(dataset) == 2
    assert np.allclose(dataset[1][0], [1.0])
