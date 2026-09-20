import json
import pathlib

from orbit.cli.commands.new import _print_config_summary
from orbit.storage import EXPERIMENTS_ROOT, experiment_dir, load_results
from orbit.ui import console, info, warning


def inspect_experiment(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
    exp_dir = experiment_dir(name, root=root)
    config_path = exp_dir / "experiment.json"

    if not config_path.exists():
        console.print()
        warning(f"{name} was not found.")
        console.print()
        return

    with open(config_path) as f:
        config = json.load(f)

    _print_config_summary(config)

    results_path = exp_dir / "results" / "results.json"
    if results_path.exists():
        results = load_results(results_path)
        message = (
            f"Results: final loss {results.final_loss:.4f} over "
            f"{len(results.loss_history)} epoch(s)"
        )
        if results.duration_seconds is not None:
            message += f" in {results.duration_seconds:.2f}s"
        info(message)
    else:
        info("Not run yet.")
    console.print()