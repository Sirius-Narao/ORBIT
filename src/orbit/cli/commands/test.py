from typing import Optional
import json
import pathlib

from orbit.storage import (
    EXPERIMENTS_ROOT,
    experiment_dir,
    load_results,
    save_results,
    checkpoint_exists,
    load_checkpoint,
)
from orbit.core import load_experiment, Results
from orbit.ui import console, warning


def test_experiment(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> Optional[Results]:
    exp_dir = experiment_dir(name, root=root)

    try:
        with open(exp_dir / "experiment.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No experiment.json found for {name!r} at {exp_dir}")

    if not checkpoint_exists(name, root=root):
        warning(
            f"{name} has not been trained yet - run 'orbit train {name}' or "
            f"'orbit run {name}' first."
        )
        return None

    if "test_split" not in config:
        warning(
            f"{name} has no test split configured (\"test_split\" not set) - "
            "nothing to test against."
        )
        return None

    if "seed" not in config:
        console.print()
        warning(
            f"{name} has no recorded seed - the reconstructed test split may "
            "not match the one used during training."
        )

    experiment = load_experiment(config)
    load_checkpoint(name, experiment.model, root=root)

    test_loss, test_accuracy = experiment.trainer.evaluate(
        experiment.model,
        experiment.loss_fn,
        experiment.test_dataloader,
        accuracy_fn=experiment.accuracy_fn,
    )

    results_path = exp_dir / "results" / "results.json"
    results = load_results(results_path)
    results.test_loss = test_loss
    results.test_accuracy = test_accuracy
    save_results(results=results, path=results_path)

    return results
