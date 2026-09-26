from orbit.core.metrics import format_metric
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
    table.add_column("Test Loss")
    table.add_column("Epochs")
    table.add_column("Duration")
    table.add_column("Grad Norm")
    table.add_column("Final Accuracy / R²")
    # table.add_column("Seed")

    for name in names:
        results_path = root / name / "results" / "results.json"
        if results_path.exists():
            results = load_results(results_path)
            duration = f"{results.duration_seconds:.2f}s" if results.duration_seconds is not None else "-"
            grad_norm = f"{results.gradient_norm_history[-1]:.4f}" if results.gradient_norm_history else "-"
            accuracy = (
                format_metric(results.accuracy_history[-1], results.hyperparams.get("task"))
                if results.accuracy_history
                else "-"
            )
            test_loss = f"{results.test_loss:.4f}" if results.test_loss is not None else "-"
            table.add_row(
                name, "done", f"{results.final_loss:.4f}", test_loss, str(len(results.loss_history)),
                duration, grad_norm, accuracy,
            )
        else:
            table.add_row(name, "not run", "-", "-", "-", "-", "-", "-")

    console.print()
    console.print(table)
    console.print()