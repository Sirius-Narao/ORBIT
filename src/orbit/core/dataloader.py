from .dataset import Dataset
from .tensor import Tensor
import numpy as np
import math

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]

            xs, ys = zip(*(self.dataset[i] for i in batch_indices))
            X_batch = np.stack(xs)
            Y_batch = np.stack(ys)

            yield Tensor(X_batch), Tensor(Y_batch)
