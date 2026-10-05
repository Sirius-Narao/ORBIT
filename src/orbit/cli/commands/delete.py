from orbit.sweeps.config import list_sweep_names, sweep_dir, sweep_membership, sweeps_root_for
from orbit.ui import console, success, warning
import pathlib
import shutil
from typing import Optional
from orbit.storage.workspace import experiments_root


def delete_experiments(
    name: str = None,
    is_all: bool = False,
    root: Optional[pathlib.Path] = None,
    sweeps_root: Optional[pathlib.Path] = None,
) -> None:
    """
    Delete one experiment, or with is_all every experiment plus every sweep
    manifest - once all runs are gone a sweep would only list "missing"
    runs. Deleting a single run of a sweep is allowed, with a warning that
    the sweep will now show it as missing.

    sweeps_root defaults to the sweeps dir next to root (sweeps_root_for),
    never the configured sweeps directory, so pointing root elsewhere (e.g. a test's
    tmp_path) can't wipe the real project's sweeps.
    """
    root = experiments_root(root)
    if sweeps_root is None:
        sweeps_root = sweeps_root_for(root)

    names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    sweep_names = list_sweep_names(sweeps_root) if is_all else []

    if not names and not sweep_names:
        console.print()
        warning("No experiments found.")
        console.print()
        return

    if is_all:
        for existing_name in names:
            shutil.rmtree(root / existing_name)
        for existing_sweep in sweep_names:
            shutil.rmtree(sweep_dir(existing_sweep, sweeps_root))
        console.print()
        message = f"Deleted {len(names)} experiment(s)."
        if sweep_names:
            message += f" Deleted {len(sweep_names)} sweep(s)."
        success(message)
        console.print()
        return

    if name not in names:
        console.print()
        warning(f"{name} was not found.")
        console.print()
        return

    sweep = sweep_membership(sweeps_root).get(name)
    shutil.rmtree(root / name)

    console.print()
    success(f"{name} was successfully deleted.")
    if sweep is not None:
        warning(
            f"{name} belongs to sweep {sweep} - it will show as missing there "
            f"(use `orbit sweep delete {sweep}` to remove the whole sweep)."
        )
    console.print()
