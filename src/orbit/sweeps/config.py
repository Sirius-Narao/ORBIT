import json
import pathlib

from orbit.core.config import OPTIMIZER_REGISTRY
from orbit.core.dataset import NORMALIZE_METHODS

SWEEPS_ROOT = pathlib.Path(".orbits/sweeps")

# Top-level experiment.json fields a sweep can vary, mapped to how their
# values are entered/parsed: "float"/"int" are typed as comma-separated
# numbers, a list is a fixed set of choices. The model architecture is
# deliberately not sweepable in v1.
SWEEPABLE_FIELDS = {
    "learning_rate": "float",
    "optimizer": list(OPTIMIZER_REGISTRY),
    "momentum": "float",
    "batch_size": "int",
    "epochs": "int",
    "seed": "int",
    "normalize": ["none"] + list(NORMALIZE_METHODS),
    "test_split": "float",
}


def sweep_dir(name: str, root: pathlib.Path = SWEEPS_ROOT) -> pathlib.Path:
    return root / name


def sweep_exists(name: str, root: pathlib.Path = SWEEPS_ROOT) -> bool:
    return (sweep_dir(name, root) / "sweep.json").exists()


def save_sweep(sweep: dict, root: pathlib.Path = SWEEPS_ROOT) -> pathlib.Path:
    path = sweep_dir(sweep["name"], root) / "sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sweep, f, indent=2)
    return path


def load_sweep(name: str, root: pathlib.Path = SWEEPS_ROOT) -> dict:
    path = sweep_dir(name, root) / "sweep.json"
    with open(path) as f:
        return json.load(f)
