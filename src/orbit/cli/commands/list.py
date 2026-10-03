from orbit.cli.commands.ranking_display import format_value
from orbit.core.metrics import format_metric
from orbit.core.ranking import overall_ranking
from orbit.storage import EXPERIMENTS_ROOT, load_results
from orbit.sweeps.aggregator import collect_rows, default_rank_metric
from orbit.sweeps.config import list_sweep_names, load_sweep, sweep_membership, sweeps_root_for
from orbit.ui import console, warning
from rich.table import Table
import pathlib
from typing import Optional


def list_experiments(
    root: pathlib.Path = EXPERIMENTS_ROOT,
    sweeps_root: Optional[pathlib.Path] = None,
    show_runs: bool = False,
) -> None:
    """
    Print the experiments table, then (if any sweeps exist) a separate
    Sweeps table with one summary row per sweep. Sweep runs are ordinary
    experiments, so without hiding them a single sweep would flood the
    experiments table with dozens of rows; they're left out of it unless
    show_runs (`orbit list --runs`). `orbit sweep status` has per-run detail.
    sweeps_root defaults to the sweeps dir next to root (sweeps_root_for).
    """
    if sweeps_root is None:
        sweeps_root = sweeps_root_for(root)
    names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    sweep_names = list_sweep_names(sweeps_root)
    if not show_runs:
        membership = sweep_membership(sweeps_root)
        names = [name for name in names if name not in membership]

    if not names and not sweep_names:
        console.print()
        warning("No experiments found.")
        console.print()
        return

    if names:
        _print_experiments_table(names, root)
    if sweep_names:
        _print_sweeps_table(sweep_names, root, sweeps_root)
    console.print()


def _print_experiments_table(names: list, root: pathlib.Path) -> None:
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


def _print_sweeps_table(sweep_names: list, root: pathlib.Path, sweeps_root: pathlib.Path) -> None:
    table = Table(title="Sweeps")
    table.add_column("Sweep")
    table.add_column("Base")
    table.add_column("Runs")
    table.add_column("Ranked by")
    table.add_column("Best Run")
    table.add_column("Best Value")

    for name in sweep_names:
        sweep = load_sweep(name, sweeps_root)
        rows = collect_rows(sweep, root)
        done = sum(row["status"] == "done" for row in rows)
        metric = default_rank_metric(sweep)

        best_run, best_value = "-", "-"
        if any(row[metric] is not None for row in rows):
            best = overall_ranking(rows, [metric])[0]["row"]
            best_run = best["run"]
            best_value = format_value(best[metric], metric, best["task"])

        table.add_row(name, sweep["base"], f"{done}/{len(rows)} done", metric, best_run, best_value)

    console.print()
    console.print(table)
