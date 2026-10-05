import json
import pathlib

from orbit.cli.commands.new import _print_config_summary
from orbit.core.metrics import format_metric, metric_label
from orbit.storage import experiment_dir, load_results
from orbit.ui import console, info, warning
from typing import Optional
from orbit.storage.workspace import experiments_root


def inspect_experiment(name: str, root: Optional[pathlib.Path] = None) -> None:
    root = experiments_root(root)
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
        if results.gradient_norm_history:
            message += f", final gradient norm {results.gradient_norm_history[-1]:.4f}"
        task = results.hyperparams.get("task")
        label = metric_label(task)
        name = "accuracy" if label == "Accuracy" else label
        if results.accuracy_history:
            message += f", final {name} {format_metric(results.accuracy_history[-1], task)}"
        if results.test_loss is not None:
            message += f", test loss {results.test_loss:.4f}"
        if results.test_accuracy is not None:
            message += f", test {name} {format_metric(results.test_accuracy, task)}"
        info(message)
        if results.diverged_at_epoch is not None:
            warning(
                f"Diverged at epoch {results.diverged_at_epoch} - the loss became inf/NaN and "
                "training stopped early (the results above cover the epochs before it)."
            )
    else:
        info("Not run yet.")
    console.print()