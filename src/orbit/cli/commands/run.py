from orbit.storage import EXPERIMENTS_ROOT, experiment_dir, save_results
from orbit.core import load_experiment
import json

def run_experiment(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> Results:
    exp_dir = experiment_dir(name, root = root)
    config = {}

    try: 
        config = json.load(exp_dir/"experiment.json")
    except: 
        raise FileNotFoundError

    experiment = load_experiment(config)

    results = experiment.run()
    save_results(results = results, path = exp_dir/"results"/"results.json")

    return results
