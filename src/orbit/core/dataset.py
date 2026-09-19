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

if __name__ == "__main__":
    # test for XOR
    dataset = TensorDataset(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]]))
    print(len(dataset))
    print(dataset[0])

    print(dataset.X.shape[-1])
    print(dataset.Y.shape[-1])