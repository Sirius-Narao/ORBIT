from orbit.storage import EXPERIMENTS_ROOT
from orbit.ui import console, success, warning
import pathlib
import shutil


def delete_experiments(name: str = None, is_all: bool = False, root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
    if not root.exists():
        console.print()
        warning("No experiments found.")
        console.print()
        return

    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not names:
        console.print()
        warning("No experiments found.")
        console.print()
        return

    if is_all:
        for existing_name in names:
            shutil.rmtree(root / existing_name)
        console.print()
        success(f"Deleted {len(names)} experiment(s).")
        console.print()
        return

    if name not in names:
        console.print()
        warning(f"{name} was not found.")
        console.print()
        return

    shutil.rmtree(root / name)

    console.print()
    success(f"{name} was successfully deleted.")
    console.print()