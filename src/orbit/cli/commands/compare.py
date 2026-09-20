import json
import pathlib

from rich.table import Table

from orbit.storage import EXPERIMENTS_ROOT, experiment_dir, load_results
from orbit.ui import console, success, warning
from orbit.visualization import plot_loss_comparison

COMPARISONS_ROOT = pathlib.Path(".orbits/comparisons")


def _comparison_filename(names: list, log_scale: bool) -> str:
    extension = "_log_scale" if log_scale else ""

    joined = "_vs_".join(sorted(names))
    if len(joined) > 100:
        return f"comparison_{len(names)}_experiments{extension}.png"
    return f"{joined}{extension}.png"


def compare_experiments(
    names: list = None,
    is_all: bool = False,
    plot_loss: bool = False,
    log_scale: bool = False,
    root: pathlib.Path = EXPERIMENTS_ROOT,
    comparisons_root: pathlib.Path = COMPARISONS_ROOT,
) -> None:
    if is_all:
        if not root.exists():
            console.print()
            warning("No experiments found.")
            console.print()
            return
        names = sorted(p.name for p in root.iterdir() if p.is_dir())

    if not names:
        console.print()
        warning("No experiments found.")
        console.print()
        return

    table = Table(title="Experiment Comparison")
    table.add_column("Name")
    table.add_column("Dataset")
    table.add_column("Loss")
    table.add_column("Optimizer")
    table.add_column("LR")
    table.add_column("Seed")
    table.add_column("Batch Size")
    table.add_column("Epochs")
    table.add_column("Final Loss")

    found_any = False
    plot_candidates = []
    for name in names:
        exp_dir = experiment_dir(name, root=root)
        config_path = exp_dir / "experiment.json"

        if not config_path.exists():
            warning(f"{name} was not found, skipping.")
            continue

        with open(config_path) as f:
            config = json.load(f)

        results_path = exp_dir / "results" / "results.json"
        if results_path.exists():
            results = load_results(results_path)
            final_loss = f"{results.final_loss:.4f}"
            plot_candidates.append(results)
        else:
            final_loss = "not run"

        found_any = True
        table.add_row(
            name,
            config["dataset"],
            config["loss"],
            config["optimizer"],
            str(config["learning_rate"]),
            str(config.get("seed", "none")),
            str(config.get("batch_size", "32 (default)")),
            str(config["epochs"]),
            final_loss,
        )

    if not found_any:
        console.print()
        warning("No experiments to compare.")
        console.print()
        return

    console.print()
    console.print(table)
    console.print()

    if plot_loss:
        if not plot_candidates:
            warning("No experiments with results to plot.")
            console.print()
            return

        output_path = comparisons_root / _comparison_filename([r.name for r in plot_candidates], log_scale=log_scale)
        output_path = plot_loss_comparison(plot_candidates, output_path, log_scale=log_scale)
        success(f"Saved comparison plot to {output_path}")
        console.print()