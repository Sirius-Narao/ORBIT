import pathlib

import numpy as np

from orbit.nn import Module
from orbit.storage.experiments import EXPERIMENTS_ROOT, experiment_dir


def checkpoint_path(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> pathlib.Path:
    return experiment_dir(name, root=root) / "checkpoints" / "model.npz"


def checkpoint_exists(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> bool:
    return checkpoint_path(name, root=root).exists()


def save_checkpoint(name: str, model: Module, root: pathlib.Path = EXPERIMENTS_ROOT) -> pathlib.Path:
    """
    Save a Module's (e.g. a Sequential's) trained parameters, keyed by their
    named_parameters() name (e.g. "0.weight", "0.bias"), to a single .npz
    file. Only the weights are saved - the architecture itself is already
    recorded in experiment.json's "model" list and is rebuilt by
    core.config.build_model before a checkpoint is loaded back in.
    """
    path = checkpoint_path(name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {param_name: param.data for param_name, param in model.named_parameters()}
    np.savez(path, **arrays)

    return path


def load_checkpoint(name: str, model: Module, root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
    """
    Load a previously saved checkpoint's weights into `model` in place,
    matching each array back to its parameter by named_parameters() name.
    `model` must already have the same architecture the checkpoint was
    saved from (i.e. built via build_model() from the same experiment.json).
    """
    path = checkpoint_path(name, root=root)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found for {name!r} at {path}")

    with np.load(path) as data:
        for param_name, param in model.named_parameters():
            if param_name not in data:
                raise KeyError(f"Checkpoint for {name!r} is missing parameter {param_name!r}")
            param.data = data[param_name]
