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


# --- training history (weight snapshots) -----------------------------------

def snapshots_path(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> pathlib.Path:
    return experiment_dir(name, root=root) / "checkpoints" / "history.npz"


def snapshots_exist(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> bool:
    return snapshots_path(name, root=root).exists()


def save_snapshots(name: str, recorder, root: pathlib.Path = EXPERIMENTS_ROOT) -> pathlib.Path:
    """
    Save a SnapshotRecorder's weight history to one .npz: each parameter
    name (e.g. "0.weight") maps to its snapshots stacked along a new first
    axis, shape (n_snapshots, *param_shape), plus "_epochs" and "_loss".
    The leading underscore can't clash with a named_parameters() name.
    """
    path = snapshots_path(name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {param_name: np.stack(values) for param_name, values in recorder.params.items()}
    arrays["_epochs"] = np.array(recorder.epochs, dtype=int)
    arrays["_loss"] = np.array(recorder.losses, dtype=float)
    np.savez(path, **arrays)

    return path


def load_snapshots(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> dict:
    """
    {"epochs": (S,), "loss": (S,), "params": {param_name: (S, *shape)}} -
    loss is NaN for epoch 0, recorded before any training.
    """
    path = snapshots_path(name, root=root)
    if not path.exists():
        raise FileNotFoundError(f"No training history found for {name!r} at {path}")

    with np.load(path) as data:
        return {
            "epochs": data["_epochs"],
            "loss": data["_loss"],
            "params": {key: data[key] for key in data.files if not key.startswith("_")},
        }


def delete_snapshots(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
    """Remove a stale history, e.g. after a re-run that couldn't record one."""
    snapshots_path(name, root=root).unlink(missing_ok=True)


def save_training_artifacts(name: str, experiment, root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
    """
    Everything orbit run/train persist besides results.json: the trained
    weights, and the weight history when one was recorded (otherwise any
    older history is removed, since it would no longer match the checkpoint).
    """
    save_checkpoint(name, experiment.model, root=root)
    if experiment.snapshots is not None:
        save_snapshots(name, experiment.snapshots, root=root)
    else:
        delete_snapshots(name, root=root)
