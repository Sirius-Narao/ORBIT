import json
import pathlib

from rich.table import Table

from orbit.storage import EXPERIMENTS_ROOT, experiment_dir, load_results
from orbit.ui import console


def compare_experiments(names: list = None, is_all: bool = False, root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
    if is_all:
        if not root.exists():
            console.print()
            console.print("No experiments found.", style="yellow")
            console.print()
            return
        names = sorted(p.name for p in root.iterdir() if p.is_dir())

    if not names:
        console.print()
        console.print("No experiments found.", style="yellow")
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
    for name in names:
        exp_dir = experiment_dir(name, root=root)
        config_path = exp_dir / "experiment.json"

        if not config_path.exists():
            console.print(f"{name} was not found, skipping.", style="yellow")
            continue

        with open(config_path) as f:
            config = json.load(f)

        results_path = exp_dir / "results" / "results.json"
        if results_path.exists():
            results = load_results(results_path)
            final_loss = f"{results.final_loss:.4f}"
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
        console.print("No experiments to compare.", style="yellow")
        console.print()
        return

    console.print()
    console.print(table)
    console.print()