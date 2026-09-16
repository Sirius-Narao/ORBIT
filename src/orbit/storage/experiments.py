from orbit.core import Results
import pathlib
import json

EXPERIMENTS_ROOT = pathlib.Path(".orbits/experiments")

def save_results(results: Results, path: str) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)

def load_results(path: str) -> Results:
    path = pathlib.Path(path)
    with open(path) as f:
        data = json.load(f)

    return Results.from_dict(data)

def experiment_dir(name: str, root: pathlib.Path = EXPERIMENTS_ROOT) -> pathlib.Path:
    return root/name

