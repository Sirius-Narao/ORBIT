import numpy as np
from orbit.core.tensor import Tensor
from orbit.core.dataset import TensorDataset
from orbit.core.dataloader import DataLoader


def make_dataset(n_rows=5):
    X = np.arange(n_rows * 2).reshape(n_rows, 2)
    Y = np.arange(n_rows).reshape(n_rows, 1)
    return TensorDataset(X, Y), X, Y


def test_batch_count():
    """
    5 rows, batch_size=2 -> batches of sizes [2, 2, 1] -> ceil(5/2) = 3 batches.
    """
    dataset, _, _ = make_dataset(5)
    dataloader = DataLoader(dataset, batch_size=2)

    assert len(dataloader) == 3
    assert sum(1 for _ in dataloader) == 3


def test_last_batch_is_partial():
    dataset, _, _ = make_dataset(5)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    batches = list(dataloader)

    assert batches[0][0].shape == (2, 2)
    assert batches[1][0].shape == (2, 2)
    assert batches[2][0].shape == (1, 2)


def test_full_coverage_no_duplicates():
    """
    Regardless of shuffling, one full pass must yield every row exactly once,
    with no rows dropped or duplicated.
    """
    dataset, X, Y = make_dataset(5)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    X_seen = np.concatenate([xb.data for xb, _ in dataloader], axis=0)
    Y_seen = np.concatenate([yb.data for _, yb in dataloader], axis=0)

    assert np.array_equal(np.sort(X_seen, axis=0), np.sort(X, axis=0))
    assert np.array_equal(np.sort(Y_seen, axis=0), np.sort(Y, axis=0))


def test_shuffle_false_preserves_order():
    dataset, X, Y = make_dataset(5)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    X_seen = np.concatenate([xb.data for xb, _ in dataloader], axis=0)
    Y_seen = np.concatenate([yb.data for _, yb in dataloader], axis=0)

    assert np.array_equal(X_seen, X)
    assert np.array_equal(Y_seen, Y)


def test_batches_are_tensors():
    dataset, _, _ = make_dataset(4)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    X_batch, Y_batch = next(iter(dataloader))

    assert isinstance(X_batch, Tensor)
    assert isinstance(Y_batch, Tensor)


def test_iter_reshuffles_each_call():
    """
    Calling __iter__ again (e.g. a new epoch) must reshuffle rather than
    replaying the same order every time - otherwise every "epoch" a Trainer
    runs would see identical batches. With 5! = 120 possible orderings,
    getting the same order twice by chance is unlikely enough not to flake.
    """
    dataset, _, _ = make_dataset(5)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    first_pass = next(iter(dataloader))[0].data
    second_pass = next(iter(dataloader))[0].data

    assert not np.array_equal(first_pass, second_pass)
