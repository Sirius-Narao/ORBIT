from orbit.storage import EXPERIMENTS_ROOT, load_results
from orbit.ui import console, warning
from rich.table import Table
import pathlib


def list_experiments(root: pathlib.Path = EXPERIMENTS_ROOT) -> None:
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

    table = Table(title="Experiments")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Final Loss")
    table.add_column("Epochs")
    # table.add_column("Seed")

    for name in names:
        results_path = root / name / "results" / "results.json"
        if results_path.exists():
            results = load_results(results_path)
            table.add_row(name, "done", f"{results.final_loss:.4f}", str(len(results.loss_history)))
        else:
            table.add_row(name, "not run", "-", "-")

    console.print()
    console.print(table)
    console.print()