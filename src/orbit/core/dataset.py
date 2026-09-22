from orbit.core import Tensor
import numpy as np
from typing import Union, Tuple

class Dataset:
    """
    Data access layer, no Tensor implementation here.
    """
    def __init__(self):
        pass

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        """
        Return a single (x, y) pair for one sample, as raw values (scalar / np arrays), not yet wrapped in Tensor.
        """
        raise NotImplementedError

class TensorDataset(Dataset):
    def __init__(self, X: Union[np.ndarray, "Tensor"], Y: Union[np.ndarray, "Tensor"]):
        super().__init__()
        self.X = X
        self.Y = Y

        # Type coercion only (Tensor/list -> ndarray) - no statistical
        # rescaling happens here; see fit_normalizer/NormalizedDataset below.
        if isinstance(self.X, Tensor):
            self.X = self.X.data
        if isinstance(self.Y, Tensor):
            self.Y = self.Y.data

        if isinstance(self.X, list):
            self.X = np.array(self.X)
        if isinstance(self.Y, list):
            self.Y = np.array(self.Y)

        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError(f"Error: X number of rows ({self.X.shape[0]}) and Y number rows ({self.Y.shape[0]}) differ!")

    @property
    def input_shape(self):
        return self.X.shape[-1]

    @property
    def output_shape(self):
        return self.Y.shape[-1]


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


class Subset(Dataset):
    """
    A view over another Dataset restricted to a list of indices - backs
    train_test_split() below. Delegates everything to the wrapped dataset
    rather than copying data, so it works for any Dataset implementation
    (TensorDataset today, whatever else later) without assuming internals.
    """
    def __init__(self, dataset: Dataset, indices):
        self.dataset = dataset
        self.indices = indices

    @property
    def input_shape(self):
        return self.dataset.input_shape

    @property
    def output_shape(self):
        return self.dataset.output_shape

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def train_test_split(dataset: Dataset, test_split: float) -> Tuple[Subset, Subset]:
    """
    Split a Dataset into (train, test) Subsets. Draws from the global numpy
    RNG, same convention as DataLoader's shuffle - calling this after
    load_experiment's np.random.seed(seed) makes the split reproducible
    without persisting indices anywhere.

    test_split is rounded (not truncated) to a row count, so a small dataset
    like the 4-row "xor" fixture at the default 0.2 still gets a non-empty
    test set (round(4 * 0.2) == 1, int(4 * 0.2) == 0).
    """
    n = len(dataset)
    n_test = round(n * test_split)
    if n_test == 0 or n_test == n:
        raise ValueError(
            f"test_split={test_split} on a dataset of {n} rows leaves an "
            "empty train or test set - use a larger dataset or a less "
            "extreme split."
        )
    indices = np.random.permutation(n)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    return Subset(dataset, train_indices), Subset(dataset, test_indices)


NORMALIZE_METHODS = ("standard", "minmax")


def fit_normalizer(X: np.ndarray, method: str) -> dict:
    """
    Compute per-feature (per-column) scaling stats from X.

    - "standard" (z-score): x' = (x - mean) / std, so each feature ends up
      with mean 0 and std 1 over X.
    - "minmax": x' = (x - min) / (max - min), so each feature spans [0, 1]
      over X.

    A zero std / zero range (a constant feature) is replaced by 1: the
    feature then just becomes x - mean (or x - min), i.e. all zeros, instead
    of a 0/0 = NaN that would poison every gradient downstream.

    X must be the TRAINING rows only. The same stats are then reused to
    transform the test rows - fitting on train+test would leak test-set
    information (its mean/spread) into training.
    """
    X = np.asarray(X, dtype=float)
    if method == "standard":
        std = X.std(axis=0)
        return {"method": method, "mean": X.mean(axis=0), "std": np.where(std == 0, 1.0, std)}
    if method == "minmax":
        x_min = X.min(axis=0)
        x_range = X.max(axis=0) - x_min
        return {"method": method, "min": x_min, "range": np.where(x_range == 0, 1.0, x_range)}
    raise ValueError(f"Unknown normalize method: {method!r} (valid: {list(NORMALIZE_METHODS)})")


def apply_normalizer(X: np.ndarray, stats: dict) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if stats["method"] == "standard":
        return (X - stats["mean"]) / stats["std"]
    return (X - stats["min"]) / stats["range"]


class NormalizedDataset(Dataset):
    """
    A view over another Dataset (including a Subset) that rescales each
    sample's inputs with precomputed stats from fit_normalizer. Targets are
    left untouched, so losses stay in the target's original units.
    """
    def __init__(self, dataset: Dataset, stats: dict):
        self.dataset = dataset
        self.stats = stats

    @property
    def input_shape(self):
        return self.dataset.input_shape

    @property
    def output_shape(self):
        return self.dataset.output_shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return apply_normalizer(x, self.stats), y


if __name__ == "__main__":
    # test for XOR
    dataset = TensorDataset(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]))
    print(len(dataset))
    print(dataset[0])

    print(dataset.X.shape[-1])
    print(dataset.Y.shape[-1])