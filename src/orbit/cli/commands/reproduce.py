from orbit.storage import EXPERIMENTS_ROOT, experiment_dir, load_results
from orbit.cli.commands.run import run_experiment
from orbit.ui import console, success, warning
import json
import pathlib


def reproduce_experiment(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
    exp_dir = experiment_dir(name, root=root)
    results_path = exp_dir / "results" / "results.json"

    if not results_path.exists():
        warning(f"{name} has no recorded results to reproduce (run it first).")
        return

    with open(exp_dir / "experiment.json") as f:
        config = json.load(f)

    if "seed" not in config:
        console.print()
        warning(
            f"{name} has no recorded seed - re-running uses fresh random "
            "initialization/shuffling, so the result is not expected to match."
        )

    previous = load_results(results_path)
    new_results = run_experiment(name, root=root)

    if abs(new_results.final_loss - previous.final_loss) < 1e-9:
        success(f"Reproduced: final loss matches ({new_results.final_loss:.4f}).")
    else:
        warning(
            f"Final loss differs: previous {previous.final_loss:.4f}, "
            f"now {new_results.final_loss:.4f}."
        )
    console.print()