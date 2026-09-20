import pathlib

from orbit.storage import EXPERIMENTS_ROOT, DATASETS_ROOT
from orbit.ui import console, success, warning


def init_project(
    experiments_root: pathlib.Path = EXPERIMENTS_ROOT,
    datasets_root: pathlib.Path = DATASETS_ROOT,
) -> None:
    """
    Scaffold .orbits/experiments/ and .orbits/datasets/ up front. Idempotent
    and non-destructive - mkdir(..., exist_ok=True) always runs, so this is
    safe to call on a project that already has experiments/datasets from
    before init existed; it just backfills whichever directory is missing.
    """
    project_root = experiments_root.parent
    already_initialized = project_root.exists()

    experiments_root.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)

    console.print()
    if already_initialized:
        warning(f"ORBIT project already initialized in {project_root.resolve()}")
    else:
        success(f"Initialized ORBIT project in {project_root.resolve()}")
    console.print()
