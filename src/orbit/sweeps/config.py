import json
import pathlib

from orbit.core.config import OPTIMIZER_REGISTRY
from orbit.core.dataset import NORMALIZE_METHODS
from typing import Optional
from orbit.storage.workspace import sweeps_root


# Top-level experiment.json fields a sweep can vary, mapped to how their
# values are entered/parsed: "float"/"int" are typed as comma-separated
# numbers, a list is a fixed set of choices. The model architecture is
# swept through the virtual architecture fields below.
SWEEPABLE_FIELDS = {
    "learning_rate": "float",
    "optimizer": list(OPTIMIZER_REGISTRY),
    "momentum": "float",
    "batch_size": "int",
    "epochs": "int",
    "seed": "int",
    "normalize": ["none"] + list(NORMALIZE_METHODS),
    "test_split": "float",
    "grad_clip": "float",
    # Architecture (see ARCHITECTURE_FIELDS below)
    "hidden_width": "int",
    "hidden_depth": "int",
    "activation": ["ReLU", "Tanh", "Sigmoid"],
}

# The SWEEPABLE_FIELDS that are virtual: not experiment.json keys, but
# instructions for rebuilding its "model" list as hidden_depth hidden
# Linear(hidden_width) layers with the given activation, followed by the
# base model's own output head (sweeps/generator.py's apply_architecture).
# Softmax isn't offered as an activation - it's an output activation.
ARCHITECTURE_FIELDS = ("hidden_width", "hidden_depth", "activation")

# Smallest allowed value per int field: a hidden layer needs at least one
# unit, while depth 0 (no hidden layer - a linear model) is valid.
FIELD_MINIMUMS = {"hidden_width": 1, "hidden_depth": 0, "batch_size": 1, "epochs": 1, "grad_clip": 0}


def sweep_dir(name: str, root: Optional[pathlib.Path] = None) -> pathlib.Path:
    root = sweeps_root(root)
    return root / name


def sweep_exists(name: str, root: Optional[pathlib.Path] = None) -> bool:
    root = sweeps_root(root)
    return (sweep_dir(name, root) / "sweep.json").exists()


def save_sweep(sweep: dict, root: Optional[pathlib.Path] = None) -> pathlib.Path:
    root = sweeps_root(root)
    path = sweep_dir(sweep["name"], root) / "sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sweep, f, indent=2)
    return path


def load_sweep(name: str, root: Optional[pathlib.Path] = None) -> dict:
    root = sweeps_root(root)
    path = sweep_dir(name, root) / "sweep.json"
    with open(path) as f:
        return json.load(f)


def list_sweep_names(root: Optional[pathlib.Path] = None) -> list:
    """Sweeps on disk: directories under root that hold a sweep.json."""
    root = sweeps_root(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "sweep.json").exists())


def sweep_membership(root: Optional[pathlib.Path] = None) -> dict:
    """{run_name: sweep_name} for every run listed in any sweep manifest."""
    root = sweeps_root(root)
    membership = {}
    for name in list_sweep_names(root):
        for run in load_sweep(name, root)["runs"]:
            membership[run["name"]] = name
    return membership


def sweeps_root_for(experiments_root: pathlib.Path) -> pathlib.Path:
    """
    The sweeps directory that sits next to an experiments directory:
    .orbits/experiments -> .orbits/sweeps (== the workspace's sweeps dir in normal use).

    Commands that take only an experiments root (orbit list, orbit delete)
    derive their sweeps root from it instead of defaulting to the workspace's
    sweeps dir, so a call pointed at some other experiments directory (a
    test's tmp_path) can never read - or, for `orbit delete --all`, wipe -
    the real project's .orbits/sweeps/.
    """
    return pathlib.Path(experiments_root).parent / "sweeps"
