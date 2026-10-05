import json
import pathlib
from typing import Optional

from rich.table import Table

from orbit.cli.commands.new import _format_optimizer
from orbit.cli.commands.ranking_display import (
    METRIC_HEADERS,
    OVERALL_BEST_STYLE,
    best_for_metric,
    choose_metrics,
    drop_unusable,
    metric_cell,
    print_bests,
    ranking_title,
    test_metrics_for,
)
from orbit.core.ranking import RANK_METRICS, overall_ranking
from orbit.storage import experiment_dir, load_results
from orbit.ui import console, success, warning
from orbit.visualization import plot_loss_comparison
from orbit.storage import workspace
from orbit.storage.workspace import experiments_root



def _comparison_filename(names: list, log_scale: bool, metric: str = None) -> str:
    extension = ""
    if metric:
        extension += f"_{metric}"
    if log_scale:
        extension += "_log_scale"

    joined = "_vs_".join(sorted(names))
    if len(joined) > 100:
        return f"comparison_{len(names)}_experiments{extension}.png"
    return f"{joined}{extension}.png"


def compare_experiments(
    names: list = None,
    is_all: bool = False,
    plot_loss: bool = False,
    log_scale: bool = False,
    root: Optional[pathlib.Path] = None,
    comparisons_root: Optional[pathlib.Path] = None,
    by: Optional[list] = None,
    test_only: bool = False,
) -> Optional[list]:
    """
    Print a table of the experiments' configs and results. With by (a list
    of metrics, or [] to pick them interactively), the experiments are also
    ranked: several metrics combine by average rank (core/ranking.py), the
    overall best row is green, and each metric's best value gets its own
    color (see ranking_display.py). Without by, no ranking - the table is
    in the order given, as before ranking existed. test_only (--test) ranks
    by the test metrics alone (ranking_display.TEST_METRICS).

    Returns the ranked rows (best first) when ranking, else None.
    """
    root = experiments_root(root)
    comparisons_root = workspace.comparisons_root(comparisons_root)
    if test_only and by is not None:
        raise ValueError("Pass either by or test_only, not both")

    if is_all:
        if not root.exists():
            console.print()
            warning("No experiments found.")
            console.print()
            return None
        names = sorted(p.name for p in root.iterdir() if p.is_dir())

    if not names:
        console.print()
        warning("No experiments found.")
        console.print()
        return None

    rows = []
    plot_candidates = []
    for name in names:
        exp_dir = experiment_dir(name, root=root)
        config_path = exp_dir / "experiment.json"

        if not config_path.exists():
            warning(f"{name} was not found, skipping.")
            continue

        with open(config_path) as f:
            config = json.load(f)

        row = {"name": name, "config": config, "results": None, "task": config.get("task")}
        row.update({m: None for m in RANK_METRICS})
        results_path = exp_dir / "results" / "results.json"
        if results_path.exists():
            results = load_results(results_path)
            row.update(
                results=results,
                final_loss=results.final_loss,
                test_loss=results.test_loss,
                accuracy=results.accuracy_history[-1] if results.accuracy_history else None,
                test_accuracy=results.test_accuracy,
                duration_seconds=results.duration_seconds,
                task=results.hyperparams.get("task", row["task"]),
            )
            plot_candidates.append(results)
        rows.append(row)

    if not rows:
        console.print()
        warning("No experiments to compare.")
        console.print()
        return None

    if test_only:
        metrics = test_metrics_for(rows)
    else:
        metrics = choose_metrics(by, default=None)
        if by is not None and not metrics:
            warning("No metrics selected - showing the comparison without ranking.")
        if metrics:
            metrics = drop_unusable(rows, metrics)

    ranked_rows = None
    if metrics:
        ranking = overall_ranking(rows, metrics)
        _print_ranked_table(ranking, metrics)
        ranked_rows = [entry["row"] for entry in ranking]
    else:
        _print_table(rows)

    if plot_loss:
        if not plot_candidates:
            warning("No experiments with results to plot.")
            console.print()
            return ranked_rows

        output_path = comparisons_root / _comparison_filename([r.name for r in plot_candidates], log_scale=log_scale)
        output_path = plot_loss_comparison(plot_candidates, output_path, log_scale=log_scale)
        success(f"Saved comparison plot to {output_path}")
        console.print()

    return ranked_rows


def _add_config_columns(table: Table) -> None:
    for column in ("Name", "Dataset", "Loss", "Optimizer", "LR", "Seed", "Batch Size", "Epochs"):
        table.add_column(column)


def _config_cells(row: dict) -> list:
    config = row["config"]
    return [
        row["name"],
        config["dataset"],
        config["loss"],
        _format_optimizer(config),
        str(config["learning_rate"]),
        str(config.get("seed", "none")),
        str(config.get("batch_size", "32 (default)")),
        str(config["epochs"]),
    ]


def _print_table(rows: list) -> None:
    table = Table(title="Experiment Comparison")
    _add_config_columns(table)
    table.add_column("Final Loss")
    table.add_column("Test Loss")

    for row in rows:
        if row["results"] is not None:
            final_loss = f"{row['final_loss']:.4f}"
            test_loss = f"{row['test_loss']:.4f}" if row["test_loss"] is not None else "not trained"
        else:
            final_loss = "not run"
            test_loss = "not run"
        table.add_row(*_config_cells(row), final_loss, test_loss)

    console.print()
    console.print(table)
    console.print()


def _print_ranked_table(ranking: list, metrics: list) -> None:
    bests = {m: best_for_metric(ranking, m) for m in metrics}
    multi = len(metrics) > 1

    table = Table(title=f"Experiment Comparison {ranking_title(metrics)}")
    table.add_column("#")
    _add_config_columns(table)
    for metric in metrics:
        table.add_column(METRIC_HEADERS[metric])
    if multi:
        table.add_column("Avg Rank")

    for position, entry in enumerate(ranking, start=1):
        row = entry["row"]
        cells = [str(position), *_config_cells(row)]
        for metric in metrics:
            if row["results"] is None:
                cells.append("not run")
            else:
                cells.append(metric_cell(row[metric], metric, row["task"], entry["index"] in bests[metric]))
        if multi:
            avg = entry["avg_rank"]
            text = "-" if avg is None else f"{avg:.3g}"
            cells.append(text if entry["complete"] or avg is None else f"{text} (incomplete)")
        table.add_row(*cells, style=OVERALL_BEST_STYLE if position == 1 else None)

    console.print()
    console.print(table)
    print_bests(ranking, metrics, "name")
    console.print()
