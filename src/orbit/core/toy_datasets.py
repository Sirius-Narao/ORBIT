"""
Small built-in 2-D datasets, picked by name like "xor" ("dataset": "moons").
Two input features means orbit boundary can draw what the model learned.

  moons    - two interleaving half circles          binary (0/1 target)
  circles  - a ring around a disc                    binary (0/1 target)
  spirals  - three interleaved spiral arms           3 classes (class index)
  blobs    - three Gaussian clusters                 3 classes (class index)

moons and circles are linearly inseparable binary problems (a model needs a
hidden layer); spirals is hard and needs depth or width; blobs is the easy
multiclass sanity check.

Each generator draws from its own np.random.default_rng(TOY_DATASET_SEED),
never the global numpy RNG. So a name always means exactly the same data,
and the global RNG stream that an experiment's "seed" controls (weight init,
the train/test split, shuffling) is left untouched by building the dataset.
"""
import numpy as np

from orbit.core.dataset import TensorDataset

TOY_DATASET_SEED = 0


def _rng():
    return np.random.default_rng(TOY_DATASET_SEED)


def moons(n_samples: int = 200, noise: float = 0.1) -> TensorDataset:
    rng = _rng()
    half = n_samples // 2
    angles = rng.uniform(0, np.pi, size=n_samples)
    upper = np.stack([np.cos(angles[:half]), np.sin(angles[:half])], axis=1)
    lower = np.stack([1 - np.cos(angles[half:]), 0.5 - np.sin(angles[half:])], axis=1)
    X = np.concatenate([upper, lower]) + rng.normal(0, noise, size=(n_samples, 2))
    Y = np.concatenate([np.zeros(half), np.ones(n_samples - half)]).reshape(-1, 1)
    return TensorDataset(X, Y)


def circles(n_samples: int = 200, noise: float = 0.08, factor: float = 0.5) -> TensorDataset:
    rng = _rng()
    half = n_samples // 2
    angles = rng.uniform(0, 2 * np.pi, size=n_samples)
    radii = np.concatenate([np.ones(half), np.full(n_samples - half, factor)])
    X = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    X += rng.normal(0, noise, size=X.shape)
    # 1 = the inner disc, 0 = the outer ring.
    Y = np.concatenate([np.zeros(half), np.ones(n_samples - half)]).reshape(-1, 1)
    return TensorDataset(X, Y)


def spirals(n_samples: int = 300, noise: float = 0.2, n_classes: int = 3) -> TensorDataset:
    rng = _rng()
    per_class = n_samples // n_classes
    X, Y = [], []
    for k in range(n_classes):
        radius = np.linspace(0.0, 1.0, per_class)
        angle = np.linspace(k * 4.0, (k + 1) * 4.0, per_class) + rng.normal(0, noise, per_class)
        X.append(np.stack([radius * np.sin(angle), radius * np.cos(angle)], axis=1))
        Y.append(np.full(per_class, k))
    return TensorDataset(np.concatenate(X), np.concatenate(Y), num_classes=n_classes)


BLOB_CENTERS = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])


def blobs(n_samples: int = 150, std: float = 0.7) -> TensorDataset:
    rng = _rng()
    n_classes = len(BLOB_CENTERS)
    per_class = n_samples // n_classes
    X = np.concatenate([center + rng.normal(0, std, size=(per_class, 2)) for center in BLOB_CENTERS])
    Y = np.repeat(np.arange(n_classes), per_class)
    return TensorDataset(X, Y, num_classes=n_classes)


TOY_DATASETS = {
    "moons": moons,
    "circles": circles,
    "spirals": spirals,
    "blobs": blobs,
}
