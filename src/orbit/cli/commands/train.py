from orbit.storage import EXPERIMENTS_ROOT, experiment_dir, save_results, save_training_artifacts
from orbit.core import Results
from orbit.core.config import load_experiment
import json
import pathlib

def train_experiment(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> Results:
    exp_dir = experiment_dir(name, root=root)

    try:
        with open(exp_dir / "experiment.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No experiment.json found for {name!r} at {exp_dir}")

    experiment = load_experiment(config)

    results = experiment.run(skip_test=True)
    save_results(results=results, path=exp_dir/"results"/"results.json")
    save_training_artifacts(name, experiment, root=root)

    return results
