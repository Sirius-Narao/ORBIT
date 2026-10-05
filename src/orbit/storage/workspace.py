"""
Where ORBIT's on-disk state lives. Every storage/command function that takes
a root defaults it to None and resolves it through one of these, so the
location is read from the settings file at call time rather than frozen into
a module constant at import time (which orbit init/config set, or a test,
could never change afterwards).

    root = experiments_root(root)   # an explicit root wins, None = workspace
"""
import pathlib
from typing import Optional

from orbit.settings import workspace_path


def workspace_root() -> pathlib.Path:
    return workspace_path()


def experiments_root(root: Optional[pathlib.Path] = None) -> pathlib.Path:
    return pathlib.Path(root) if root is not None else workspace_root() / "experiments"


def datasets_root(root: Optional[pathlib.Path] = None) -> pathlib.Path:
    return pathlib.Path(root) if root is not None else workspace_root() / "datasets"


def sweeps_root(root: Optional[pathlib.Path] = None) -> pathlib.Path:
    return pathlib.Path(root) if root is not None else workspace_root() / "sweeps"


def comparisons_root(root: Optional[pathlib.Path] = None) -> pathlib.Path:
    return pathlib.Path(root) if root is not None else workspace_root() / "comparisons"
