"""
Display side of multi-metric ranking (core/ranking.py), shared by
`orbit compare --by ...` and `orbit sweep compare --by ...`.

Colors: the best run *overall* (lowest average rank) keeps ORBIT's success
green on its whole row. Each metric's own best gets that metric's value
cell in a distinct color - never green, so it can't be confused with the
overall best - plus a star, so the highlight survives output with no color
(piped/redirected). A matching colored "Best <metric>: ..." line follows
the table.
"""
from typing import List, Optional

import questionary
from rich.text import Text

from orbit.core.metrics import format_metric, metric_label
from orbit.core.ranking import RANK_METRICS, usable_metrics
from orbit.ui import PROMPT_STYLE, console, success, warning

OVERALL_BEST_STYLE = "bold #b0ffb3"
BEST_MARK = " ★"

# More saturated than the rest of ORBIT's pastel palette on purpose: pastels
# collapse together once a terminal downgrades to 256 or 16 colors (the
# first version's gold and orange both became color 223). These stay
# distinct from each other and from the overall-best green in truecolor,
# 256-color (221/75/105/212/208 vs green's 157) and 16-color
# (11/14/12/13/9 vs green's 7) terminals - see test_ranking_display.py.
METRIC_COLORS = {
    "final_loss": "#ffd75f",        # gold
    "test_loss": "#5fafff",         # blue
    "accuracy": "#9b87ff",          # violet
    "test_accuracy": "#ff87d7",     # pink
    "duration_seconds": "#ff8700",  # orange
}

METRIC_HEADERS = {
    "final_loss": "Final Loss",
    "test_loss": "Test Loss",
    "accuracy": "Accuracy / R²",
    "test_accuracy": "Test Accuracy / R²",
    "duration_seconds": "Duration",
}


def metric_name(metric: str, task: Optional[str] = None) -> str:
    """Human name for a metric, with the accuracy slot labeled by task (R² vs Accuracy)."""
    if metric == "accuracy":
        return metric_label(task)
    if metric == "test_accuracy":
        return f"Test {metric_label(task)}"
    return METRIC_HEADERS[metric]


def format_value(value, metric: str, task: Optional[str] = None) -> str:
    if value is None:
        return "-"
    if metric in ("accuracy", "test_accuracy"):
        return format_metric(value, task)
    if metric == "duration_seconds":
        return f"{value:.2f}s"
    # Significant figures, not fixed decimals: the runs most worth telling
    # apart are the near-zero-loss ones, which .4f would all show as 0.0000.
    return f"{value:.4g}"


def metric_cell(value, metric: str, task: Optional[str], is_best: bool) -> Text:
    text = format_value(value, metric, task)
    if is_best:
        return Text(text + BEST_MARK, style=f"bold {METRIC_COLORS[metric]}")
    return Text(text)


def choose_metrics(by: Optional[list], default: Optional[list]) -> Optional[list]:
    """
    by=None -> default (None means "don't rank"); by=[] (a bare `--by`) ->
    interactive checkbox over every rank metric; otherwise by as given.
    """
    if by is None:
        return default
    if by:
        return list(dict.fromkeys(by))  # drop repeats, keep order
    labels = {METRIC_HEADERS[m]: m for m in RANK_METRICS}
    chosen = questionary.checkbox(
        "Rank by which metric(s)?", choices=list(labels), style=PROMPT_STYLE
    ).ask()
    return [labels[label] for label in chosen] if chosen else None


def drop_unusable(rows: List[dict], metrics: List[str]) -> List[str]:
    """Warn about and drop metrics that no row has a value for."""
    usable, unusable = usable_metrics(rows, metrics)
    for metric in unusable:
        warning(f"No results have {METRIC_HEADERS[metric].lower()} - not ranking by it.")
    return usable


def ranking_title(metrics: List[str]) -> str:
    if len(metrics) == 1:
        return f"ranked by {metrics[0]}"
    return f"ranked by average rank over {', '.join(metrics)}"


def print_bests(ranking: List[dict], metrics: List[str], name_key: str, describe=None) -> None:
    """
    Print the overall best (green) and each metric's best (its own color).
    ranking is core.ranking.overall_ranking's output; name_key is the row
    field holding the experiment/run name; describe(row) optionally adds
    detail (e.g. a sweep run's params) after the name.
    """
    def label(row):
        extra = describe(row) if describe else ""
        return f"{row[name_key]} ({extra})" if extra else row[name_key]

    best = ranking[0]
    row = best["row"]
    parts = ", ".join(f"{m} {format_value(row.get(m), m, row.get('task'))}" for m in metrics)
    if len(metrics) == 1:
        success(f"Best run: {label(row)} - {parts}")
    else:
        success(f"Best run overall: {label(row)} - average rank {best['avg_rank']:g} ({parts})")

    if len(metrics) > 1:
        for metric in metrics:
            # tied-for-first rows share an averaged rank (e.g. 1.5), not 1
            top = best_for_metric(ranking, metric)
            winners = [e["row"] for e in ranking if e["index"] in top]
            if not winners:
                continue
            value = format_value(winners[0].get(metric), metric, winners[0].get("task"))
            names = ", ".join(w[name_key] for w in winners)
            console.print(
                f"Best {metric_name(metric, winners[0].get('task'))}{BEST_MARK}: {names} - {value}",
                style=METRIC_COLORS[metric],
            )


def best_for_metric(ranking: List[dict], metric: str) -> set:
    """Indices (into the original rows) of the entries tied for first on metric."""
    present = [e["ranks"][metric] for e in ranking if e["ranks"][metric] is not None]
    if not present:
        return set()
    top = min(present)
    return {e["index"] for e in ranking if e["ranks"][metric] == top}
