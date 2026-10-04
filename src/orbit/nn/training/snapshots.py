from typing import Dict, List

import numpy as np

from orbit.nn import Module

# About this many weight snapshots per run, however many epochs it has -
# enough frames for a smooth animation, few enough to stay small on disk.
MAX_SNAPSHOTS = 100
# Above this many parameters, snapshots aren't recorded at all: 100 copies
# of e.g. a 784x128 MNIST layer would be ~80 MB per run, which isn't worth
# it for a diagram that can only show a slice of such a layer anyway.
MAX_SNAPSHOT_PARAMETERS = 200_000


def snapshot_interval(epochs: int) -> int:
    return max(1, epochs // MAX_SNAPSHOTS)


def parameter_count(model: Module) -> int:
    return int(sum(param.data.size for param in model.parameters()))


class SnapshotRecorder:
    """
    Copies a model's parameters every few epochs during training, for the
    training animation. Used as (part of) Trainer.fit's on_epoch_end
    callback: record(epoch, model, loss).

    Epoch 0 - the initial weights, before any update - is recorded by
    calling record(0, model, nan) before fit() starts. After that it records
    every snapshot_interval(epochs) epochs, plus always the last one.
    Copying weights draws nothing from the RNG, so recording never changes
    a seeded run's results.
    """

    def __init__(self, epochs: int):
        self.total_epochs = epochs
        self.every = snapshot_interval(epochs)
        self.epochs: List[int] = []
        self.losses: List[float] = []
        self.params: Dict[str, List[np.ndarray]] = {}

    def __call__(self, epoch: int, model: Module, loss: float) -> None:
        self.record(epoch, model, loss)

    def record(self, epoch: int, model: Module, loss: float) -> None:
        if epoch % self.every != 0 and epoch != self.total_epochs:
            return
        self.epochs.append(int(epoch))
        self.losses.append(float(loss))
        for name, param in model.named_parameters():
            self.params.setdefault(name, []).append(np.array(param.data, dtype=float, copy=True))

    def __len__(self) -> int:
        return len(self.epochs)
