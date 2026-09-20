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

        # Normalizing inputs
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


if __name__ == "__main__":
    # test for XOR
    dataset = TensorDataset(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]))
    print(len(dataset))
    print(dataset[0])

    print(dataset.X.shape[-1])
    print(dataset.Y.shape[-1])