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


def list_sweep_names(root: pathlib.Path = SWEEPS_ROOT) -> list:
    """Sweeps on disk: directories under root that hold a sweep.json."""
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "sweep.json").exists())


def sweep_membership(root: pathlib.Path = SWEEPS_ROOT) -> dict:
    """{run_name: sweep_name} for every run listed in any sweep manifest."""
    membership = {}
    for name in list_sweep_names(root):
        for run in load_sweep(name, root)["runs"]:
            membership[run["name"]] = name
    return membership


def sweeps_root_for(experiments_root: pathlib.Path) -> pathlib.Path:
    """
    The sweeps directory that sits next to an experiments directory:
    .orbits/experiments -> .orbits/sweeps (== SWEEPS_ROOT in normal use).

    Commands that take only an experiments root (orbit list, orbit delete)
    derive their sweeps root from it instead of defaulting to the fixed
    SWEEPS_ROOT, so a call pointed at some other experiments directory (a
    test's tmp_path) can never read - or, for `orbit delete --all`, wipe -
    the real project's .orbits/sweeps/.
    """
    return pathlib.Path(experiments_root).parent / "sweeps"
