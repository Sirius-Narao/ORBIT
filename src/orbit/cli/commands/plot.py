import pathlib
from typing import Optional

from orbit.storage import EXPERIMENTS_ROOT, experiment_dir, load_results
from orbit.ui import warning, success
from orbit.visualization import plot_loss


def plot_experiment(
    name: str, root: pathlib.Path = EXPERIMENTS_ROOT, log_scale: bool = False
) -> Optional[pathlib.Path]:
    exp_dir = experiment_dir(name, root=root)
    config_path = exp_dir / "experiment.json"

    if not config_path.exists():
        warning(f"{name} was not found.")
        return None

    results_path = exp_dir / "results" / "results.json"
    if not results_path.exists():
        warning(f"{name} has not been run yet - nothing to plot (run it first).")
        return None

    extension = ""
    if log_scale:
        extension = extension + "_log_scale"
    
    results = load_results(results_path)
    output_path = plot_loss(results, exp_dir / "results" / f"loss{extension}.png", log_scale=log_scale)

    success(f"Saved loss plot to {output_path}")
    return output_path
