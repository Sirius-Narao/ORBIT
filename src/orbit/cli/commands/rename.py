import json
import pathlib

from orbit.storage import EXPERIMENTS_ROOT, experiment_dir
from orbit.ui import console, success


def rename_experiment(old_name: str, new_name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
    if old_name == new_name:
        console.print("Old and new names are the same - nothing to do.", style="yellow")
        return

    old_dir = experiment_dir(old_name, root=root)
    if not (old_dir / "experiment.json").exists():
        console.print(f"{old_name} was not found.", style="yellow")
        return

    new_dir = experiment_dir(new_name, root=root)
    if new_dir.exists():
        console.print(f"{new_name} already exists.", style="yellow")
        return

    old_dir.rename(new_dir)

    # Keep experiment.json's own "name" field (and results.json's, if the
    # experiment has been run) in sync with the directory it now lives in -
    # otherwise inspect/list would show the old name pulled back out of the
    # file even though the experiment is now looked up under the new one.
    config_path = new_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["name"] = new_name
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    results_path = new_dir / "results" / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results_data = json.load(f)
        results_data["name"] = new_name
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)

    success(f"Renamed {old_name!r} to {new_name!r}.")
    console.print()
