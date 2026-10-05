from orbit.storage import experiment_dir, save_results, save_training_artifacts
from orbit.core import Results
from orbit.core.config import load_experiment
from orbit.cli.commands.watch import run_with_optional_watch
import json
import pathlib
from typing import Optional
from orbit.storage.workspace import experiments_root

# What to try when a run diverges (its loss became inf/NaN) - shown by
# orbit run/train and orbit sweep start.
DIVERGENCE_HINT = 'Try a lower learning rate, the Adam optimizer, or "grad_clip" (e.g. 1.0).'


def run_experiment(name: str, root: Optional[pathlib.Path] = None, watch: bool = False) -> Results:
    root = experiments_root(root)
    exp_dir = experiment_dir(name, root = root)

    try:
        with open(exp_dir / "experiment.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No experiment.json found for {name!r} at {exp_dir}")

    experiment = load_experiment(config)

    results = run_with_optional_watch(experiment, name, watch, skip_test=False)
    save_results(results = results, path = exp_dir/"results"/"results.json")
    save_training_artifacts(name, experiment, root=root)

    return results


