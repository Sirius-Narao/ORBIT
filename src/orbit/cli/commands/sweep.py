import csv
import json
import pathlib
import shutil
from typing import List, Optional

import questionary
from rich.table import Table
from rich.text import Text

from orbit.cli.commands.plot import plot_experiments
from orbit.cli.commands.run import DIVERGENCE_HINT, run_experiment
from orbit.storage import experiment_dir
from orbit.cli.commands.ranking_display import (
    BEST_MARK,
    METRIC_COLORS,
    METRIC_HEADERS,
    OVERALL_BEST_STYLE,
    best_for_metric,
    choose_metrics,
    drop_unusable,
    format_value,
    metric_cell,
    metric_name,
    print_bests,
    ranking_title,
    test_metrics_for,
)
from orbit.core.ranking import RANK_METRICS, overall_ranking, usable_metrics
from orbit.sweeps.aggregator import (
    collect_rows,
    default_rank_metric,
    group_by_seed,
    varies_seed,
)
from orbit.sweeps.config import (
    ARCHITECTURE_FIELDS,
    FIELD_MINIMUMS,
    SWEEPABLE_FIELDS,
    load_sweep,
    save_sweep,
    sweep_dir,
    sweep_exists,
)
from orbit.sweeps.generator import describe_architecture, expand_grid
from orbit.ui import PROMPT_STYLE, console, info, success, warning
from orbit.storage import workspace
from orbit.storage.workspace import experiments_root

_EXPORT_METRICS = ["final_loss", "test_loss", "test_accuracy", "duration_seconds"]
# Shown in `sweep compare` for context even when not ranked by.
_CONTEXT_METRICS = ["final_loss", "test_loss", "test_accuracy"]
# What `sweep compare --all` ranks by: every quality metric, but not
# duration_seconds. On small/fast runs durations differ only by timing
# noise (milliseconds), and under average-rank aggregation that noise gets
# an equal vote - it pushed the lowest-loss run of a test sweep from 1st to
# 5th. Duration stays available explicitly via --by ... duration_seconds.
_ALL_METRICS = [m for m in RANK_METRICS if m != "duration_seconds"]


# --- helpers -------------------------------------------------------------------

def _parse_number_list(text: str, kind: str) -> list:
    cast = int if kind == "int" else float
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def _base_value(field: str, base_config: dict):
    """
    The base experiment's own value for a field - pre-fills its prompt.
    Architecture fields aren't config keys, so they're read off the model.
    """
    if field in ARCHITECTURE_FIELDS:
        return describe_architecture(base_config["model"])[field]
    return base_config.get(field, "none" if field == "normalize" else None)


def _ask_values(field: str, kind, base_config: dict) -> list:
    default = _base_value(field, base_config)
    if isinstance(kind, list):
        choices = [questionary.Choice(c, checked=(c == default)) for c in kind]
        return questionary.checkbox(f"{field} values:", choices=choices, style=PROMPT_STYLE).ask()

    minimum = FIELD_MINIMUMS.get(field)

    def validate(text):
        try:
            values = _parse_number_list(text, kind)
        except ValueError:
            return f"Please enter comma-separated {'whole numbers' if kind == 'int' else 'numbers'}"
        if not values:
            return "Please enter at least one value"
        if minimum is not None and any(v < minimum for v in values):
            return f"{field} values must be at least {minimum}"
        return True

    default = "" if default is None else str(default)
    answer = questionary.text(
        f"{field} values (comma-separated):", default=default, validate=validate, style=PROMPT_STYLE
    ).ask()
    return _parse_number_list(answer, kind)


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _load_or_warn(name: str, sweeps_root: pathlib.Path) -> Optional[dict]:
    if not sweep_exists(name, sweeps_root):
        console.print()
        warning(f"Sweep {name} was not found.")
        console.print()
        return None
    return load_sweep(name, sweeps_root)


# --- commands -------------------------------------------------------------------

def create_sweep(
    name: str,
    base: str,
    root: Optional[pathlib.Path] = None,
    sweeps_root: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    """
    Interactively build a grid over the base experiment's hyperparameters,
    then save .orbits/sweeps/<name>/sweep.json plus one ordinary experiment
    per grid combination (<name>_001, <name>_002, ...), so every existing
    per-experiment command (inspect/test/reproduce/plot/...) works on a
    single run unchanged. Nothing is trained here - see start_sweep.
    """
    root = experiments_root(root)
    sweeps_root = workspace.sweeps_root(sweeps_root)
    if sweep_exists(name, sweeps_root):
        warning(f"Sweep {name} already exists.")
        return None

    base_config_path = experiment_dir(base, root=root) / "experiment.json"
    if not base_config_path.exists():
        warning(f"Base experiment {base} was not found.")
        return None
    with open(base_config_path) as f:
        base_config = json.load(f)

    fields = questionary.checkbox(
        "Which fields to vary?", choices=list(SWEEPABLE_FIELDS), style=PROMPT_STYLE
    ).ask()
    if not fields:
        warning("No fields selected - nothing to sweep.")
        return None

    grid = {}
    for field in fields:
        values = _ask_values(field, SWEEPABLE_FIELDS[field], base_config)
        if not values:
            warning(f"No values given for {field} - nothing to sweep.")
            return None
        grid[field] = values

    try:
        runs = expand_grid(name, base_config, grid)
    except ValueError as e:
        warning(f"Can't build this grid: {e}")
        return None

    table = Table(title=f"Sweep {name!r} (base: {base})", show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Values")
    for field, values in grid.items():
        table.add_row(field, ", ".join(str(v) for v in values))
    table.add_row("Runs", str(len(runs)))
    console.print()
    console.print(table)
    if "seed" not in grid:
        info(
            f"All runs share seed {base_config.get('seed', 'none')} - differences come only "
            "from the varied fields."
        )
    varied_architecture = [f for f in ARCHITECTURE_FIELDS if f in grid]
    if varied_architecture:
        for field in ARCHITECTURE_FIELDS:
            if field not in grid:
                info(f"{field} not varied: all runs use the base model's {field} ({_base_value(field, base_config)}).")
    console.print()

    taken = [run_name for run_name, _, _ in runs if experiment_dir(run_name, root=root).exists()]
    if taken:
        warning(f"Experiment(s) already exist with run names: {', '.join(taken)}. Pick another sweep name.")
        return None

    if not questionary.confirm(f"Create {len(runs)} runs?", style=PROMPT_STYLE).ask():
        warning("Cancelled.")
        return None

    for run_name, _, config in runs:
        exp_dir = experiment_dir(run_name, root=root)
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(config, f, indent=2)

    path = save_sweep(
        {
            "name": name,
            "base": base,
            "base_config": base_config,
            "grid": grid,
            "runs": [{"name": run_name, "params": params} for run_name, params, _ in runs],
        },
        sweeps_root,
    )
    success(f"Saved sweep to {path} ({len(runs)} runs). Start it with: orbit sweep start {name}")
    console.print()
    return path


def start_sweep(
    name: str,
    root: Optional[pathlib.Path] = None,
    sweeps_root: Optional[pathlib.Path] = None,
) -> Optional[dict]:
    """
    Train every run that isn't done yet. Runs with results are skipped, so
    re-running this after an interruption (Ctrl-C, a crash) resumes where it
    left off. A run that raises is reported and the sweep moves on.
    """
    root = experiments_root(root)
    sweeps_root = workspace.sweeps_root(sweeps_root)
    sweep = _load_or_warn(name, sweeps_root)
    if sweep is None:
        return None

    total = len(sweep["runs"])
    counts = {"done": 0, "diverged": 0, "failed": 0, "skipped": 0, "missing": 0}
    console.print()
    for i, run in enumerate(sweep["runs"], start=1):
        exp_dir = experiment_dir(run["name"], root=root)
        if (exp_dir / "results" / "results.json").exists():
            counts["skipped"] += 1
            continue
        if not (exp_dir / "experiment.json").exists():
            warning(f"Run {i}/{total}: {run['name']} is missing (deleted?), skipping.")
            counts["missing"] += 1
            continue

        info(f"Run {i}/{total}: {run['name']} ({_format_params(run['params'])})")
        try:
            results = run_experiment(run["name"], root=root)
        except Exception as e:
            warning(f"{run['name']} failed: {e}")
            counts["failed"] += 1
            continue
        if results.diverged_at_epoch is not None:
            warning(f"{run['name']}: diverged at epoch {results.diverged_at_epoch} (loss became inf/NaN).")
            counts["diverged"] += 1
            continue
        success(f"{run['name']}: final loss {results.final_loss:.4f}")
        counts["done"] += 1

    console.print()
    summary = (
        f"Sweep {name}: {counts['done']} trained, {counts['skipped']} already done, "
        f"{counts['failed']} failed"
    )
    if counts["diverged"]:
        summary += f", {counts['diverged']} diverged"
    if counts["missing"]:
        summary += f", {counts['missing']} missing"
    (warning if counts["failed"] or counts["missing"] or counts["diverged"] else success)(summary)
    if counts["diverged"]:
        info(f"Diverged runs stopped early and rank last. {DIVERGENCE_HINT}")
    console.print()
    return counts


def sweep_status(
    name: str,
    root: Optional[pathlib.Path] = None,
    sweeps_root: Optional[pathlib.Path] = None,
) -> Optional[list]:
    root = experiments_root(root)
    sweeps_root = workspace.sweeps_root(sweeps_root)
    sweep = _load_or_warn(name, sweeps_root)
    if sweep is None:
        return None

    rows = collect_rows(sweep, root)
    fields = list(sweep["grid"])
    table = Table(title=f"Sweep {name!r} status")
    table.add_column("Run")
    for field in fields:
        table.add_column(field)
    table.add_column("Status")
    table.add_column("Final Loss")
    for row in rows:
        table.add_row(
            row["run"],
            *(str(row["params"].get(f, "-")) for f in fields),
            row["status"] if row["diverged_at_epoch"] is None else f"diverged (epoch {row['diverged_at_epoch']})",
            format_value(row["final_loss"], "final_loss", row["task"]),
        )

    done = sum(row["status"] == "done" for row in rows)
    console.print()
    console.print(table)
    (success if done == len(rows) else info)(f"{done}/{len(rows)} done")
    console.print()
    return rows


def compare_sweep(
    name: str,
    by: Optional[list] = None,
    all_metrics: bool = False,
    test_only: bool = False,
    root: Optional[pathlib.Path] = None,
    sweeps_root: Optional[pathlib.Path] = None,
) -> Optional[list]:
    """
    Rank the sweep's runs by one or several metrics (default: test loss
    when the sweep has a test split, else final loss; a bare --by prompts;
    all_metrics / --all ranks by every quality metric the runs recorded -
    everything but duration, see _ALL_METRICS; test_only / --test ranks by
    the test metrics alone, ranking_display.TEST_METRICS).
    Several metrics are combined by average rank (core/ranking.py): the
    overall best row is green, and each metric's best value gets its own
    color (see ranking_display.py). When the grid varies the seed, groups
    of runs differing only by seed are also ranked on their mean metrics,
    shown as mean +- std - a single seed's win can just be init/shuffle luck.

    Returns the ranked rows, best first.
    """
    root = experiments_root(root)
    sweeps_root = workspace.sweeps_root(sweeps_root)
    if sum([by is not None, all_metrics, test_only]) > 1:
        raise ValueError("Pass only one of by, all_metrics and test_only")

    sweep = _load_or_warn(name, sweeps_root)
    if sweep is None:
        return None

    rows = collect_rows(sweep, root)
    if test_only:
        metrics = test_metrics_for(rows)
        if not metrics:
            console.print()
            return [row for row in rows]
    elif all_metrics:
        # Every quality metric any run recorded (see _ALL_METRICS). Ones
        # nobody has (e.g. test metrics without a test split, accuracy
        # without a task) are skipped quietly - they weren't asked for by
        # name, so there's nothing to warn about.
        metrics = usable_metrics(rows, _ALL_METRICS)[0]
        if metrics:
            info(f"Ranking by every recorded quality metric: {', '.join(metrics)}")
    else:
        metrics = choose_metrics(by, default=[default_rank_metric(sweep)])
        if not metrics:
            warning("No metrics selected.")
            return None
        metrics = drop_unusable(rows, metrics)
    if not metrics:
        console.print()
        warning(f"No runs have results for those metrics yet - run `orbit sweep start {name}` first.")
        console.print()
        return [row for row in rows]

    ranking = overall_ranking(rows, metrics)
    bests = {m: best_for_metric(ranking, m) for m in metrics}
    # Selected metrics first (the ones highlighted), then other recorded ones for context.
    shown = metrics + [m for m in _CONTEXT_METRICS if m not in metrics and usable_metrics(rows, [m])[0]]
    multi = len(metrics) > 1

    fields = list(sweep["grid"])
    table = Table(title=f"Sweep {name!r} {ranking_title(metrics)}")
    table.add_column("#")
    table.add_column("Run")
    for field in fields:
        table.add_column(field)
    for metric in shown:
        table.add_column(METRIC_HEADERS[metric])
    if multi:
        table.add_column("Avg Rank")
    for position, entry in enumerate(ranking, start=1):
        row = entry["row"]
        table.add_row(
            str(position),
            row["run"],
            *(str(row["params"].get(f, "-")) for f in fields),
            *(metric_cell(row[m], m, row["task"], m in bests and entry["index"] in bests[m]) for m in shown),
            *([_format_avg_rank(entry)] if multi else []),
            style=OVERALL_BEST_STYLE if position == 1 else None,
        )
    console.print()
    console.print(table)
    print_bests(ranking, metrics, "run", describe=lambda row: _format_params(row["params"]))

    if varies_seed(sweep):
        groups = group_by_seed(rows, metrics)
        group_bests = {m: _best_groups(groups, m) for m in metrics}
        other_fields = [f for f in fields if f != "seed"]
        grouped = Table(title=f"Grouped over seeds ({', '.join(metrics)}, mean ± std)")
        for field in other_fields:
            grouped.add_column(field)
        grouped.add_column("Seeds")
        for metric in metrics:
            grouped.add_column(METRIC_HEADERS[metric])
        if multi:
            grouped.add_column("Avg Rank")
        for i, group in enumerate(groups):
            grouped.add_row(
                *(str(group["params"].get(f, "-")) for f in other_fields),
                str(group["n"]),
                *(_mean_std_cell(group, m, i in group_bests[m]) for m in metrics),
                *([_format_avg_rank(group)] if multi else []),
                style=OVERALL_BEST_STYLE if i == 0 else None,
            )
        console.print()
        console.print(grouped)
        best_group = groups[0]
        detail = ", ".join(
            f"mean {m} {format_value(best_group['mean'][m], m, best_group['task'])}" for m in metrics
        )
        success(f"Best configuration over seeds: {_format_params(best_group['params']) or '(base)'} - {detail}")
        if multi:
            for metric in metrics:
                winners = [g for i, g in enumerate(groups) if i in group_bests[metric]]
                if winners:
                    console.print(
                        f"Best {metric_name(metric, winners[0]['task'])} over seeds{BEST_MARK}: "
                        + "; ".join(_format_params(g["params"]) or "(base)" for g in winners)
                        + f" - mean {format_value(winners[0]['mean'][metric], metric, winners[0]['task'])}",
                        style=METRIC_COLORS[metric],
                    )

    console.print()
    return [entry["row"] for entry in ranking]


def _format_avg_rank(entry: dict) -> str:
    if entry["avg_rank"] is None:
        return "-"
    text = f"{entry['avg_rank']:.3g}"
    return text if entry["complete"] else f"{text} (incomplete)"


def _best_groups(groups: list, metric: str) -> set:
    present = [g["ranks"][metric] for g in groups if g["ranks"][metric] is not None]
    if not present:
        return set()
    top = min(present)
    return {i for i, g in enumerate(groups) if g["ranks"][metric] == top}


def _mean_std_cell(group: dict, metric: str, is_best: bool):
    mean, std = group["mean"][metric], group["std"][metric]
    if mean is None:
        return "-"
    text = f"{format_value(mean, metric, group['task'])} ± {format_value(std, metric, group['task'])}"
    if is_best:
        return Text(text + BEST_MARK, style=f"bold {METRIC_COLORS[metric]}")
    return text


def export_sweep(
    name: str,
    output: Optional[str] = None,
    root: Optional[pathlib.Path] = None,
    sweeps_root: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    root = experiments_root(root)
    sweeps_root = workspace.sweeps_root(sweeps_root)
    sweep = _load_or_warn(name, sweeps_root)
    if sweep is None:
        return None

    path = pathlib.Path(output) if output else sweep_dir(name, sweeps_root) / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(sweep["grid"])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", *fields, "status", *_EXPORT_METRICS])
        for row in collect_rows(sweep, root):
            writer.writerow([
                row["run"],
                *(row["params"].get(field, "") for field in fields),
                row["status"],
                *("" if row[m] is None else row[m] for m in _EXPORT_METRICS),
            ])

    console.print()
    success(f"Exported {len(sweep['runs'])} runs to {path}")
    console.print()
    return path


def plot_sweep(
    name: str,
    log_scale: bool = False,
    metrics: Optional[list] = None,
    root: Optional[pathlib.Path] = None,
    sweeps_root: Optional[pathlib.Path] = None,
) -> List[pathlib.Path]:
    """
    Plot the finished runs via `orbit plot`'s machinery. Multi-run plots go
    under the sweep's own plots/ dir rather than .orbits/comparisons/ -
    long sweeps fall back to comparison_<n>_experiments.png there, which
    two same-sized sweeps would overwrite.
    """
    root = experiments_root(root)
    sweeps_root = workspace.sweeps_root(sweeps_root)
    sweep = _load_or_warn(name, sweeps_root)
    if sweep is None:
        return []

    done = [row["run"] for row in collect_rows(sweep, root) if row["status"] == "done"]
    if not done:
        console.print()
        warning(f"No runs have finished yet - run `orbit sweep start {name}` first.")
        console.print()
        return []

    return plot_experiments(
        done,
        log_scale=log_scale,
        metrics=metrics,
        root=root,
        comparisons_root=sweep_dir(name, sweeps_root) / "plots",
    )


def delete_sweep(
    name: str,
    yes: bool = False,
    root: Optional[pathlib.Path] = None,
    sweeps_root: Optional[pathlib.Path] = None,
) -> Optional[int]:
    """
    Delete a sweep: every run listed in its manifest, then the sweep's own
    directory. Runs are taken from sweep.json, never matched by name prefix,
    so an unrelated experiment that happens to be named <sweep>_something
    survives. Asks first (it can be dozens of experiments) unless yes.

    Returns the number of run directories removed, or None if nothing was
    deleted.
    """
    root = experiments_root(root)
    sweeps_root = workspace.sweeps_root(sweeps_root)
    sweep = _load_or_warn(name, sweeps_root)
    if sweep is None:
        return None

    run_dirs = [
        experiment_dir(run["name"], root=root)
        for run in sweep["runs"]
        if experiment_dir(run["name"], root=root).exists()
    ]

    if not yes:
        confirmed = questionary.confirm(
            f"Delete sweep {name} and its {len(run_dirs)} run(s)?", default=False, style=PROMPT_STYLE
        ).ask()
        if not confirmed:
            console.print()
            warning("Cancelled - nothing was deleted.")
            console.print()
            return None

    for run_dir in run_dirs:
        shutil.rmtree(run_dir)
    shutil.rmtree(sweep_dir(name, sweeps_root))

    console.print()
    success(f"Deleted sweep {name} ({len(run_dirs)} runs).")
    console.print()
    return len(run_dirs)
