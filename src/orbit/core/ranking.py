"""
Ranking experiments/runs by one or several metrics at once.

Metrics live on incompatible scales (a loss of 0.01, an accuracy of 0.93, a
duration of 4.2s), so they can't simply be added up. Instead every metric
is turned into a rank - 1 for the best run, 2 for the next, ... - and a
run's overall score is its average rank across the chosen metrics, the
lowest being the best overall. This is standard rank aggregation: every
metric gets an equal vote, regardless of its units or spread.

Ties share the average of the ranks they span (values 0.1, 0.1, 0.3 ->
ranks 1.5, 1.5, 3), so a tie never favors whichever row happened to come
first.

Rows are plain dicts holding each metric under its metric name. A missing
value (None or NaN - e.g. a run that wasn't evaluated on a test split)
gets no rank for that metric; a row missing any of the chosen metrics is
"incomplete" and ranked after every complete row, since an average over
fewer metrics isn't comparable to one over all of them.
"""
import math
from typing import List, Optional

# Metric name -> whether a lower or higher value is better.
RANK_METRICS = {
    "final_loss": "min",
    "test_loss": "min",
    "accuracy": "max",       # final-epoch training accuracy (or R²)
    "test_accuracy": "max",  # test-split accuracy (or R²)
    "duration_seconds": "min",
}


def _value(row: dict, metric: str) -> Optional[float]:
    value = row.get(metric)
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def metric_ranks(rows: List[dict], metric: str) -> List[Optional[float]]:
    """One rank per row (1 = best, ties averaged), None where the value is missing."""
    if metric not in RANK_METRICS:
        raise ValueError(f"Unknown rank metric: {metric!r} (valid: {list(RANK_METRICS)})")
    sign = -1 if RANK_METRICS[metric] == "max" else 1

    present = [(sign * v, i) for i, v in ((i, _value(r, metric)) for i, r in enumerate(rows)) if v is not None]
    present.sort()

    ranks = [None] * len(rows)
    position = 0
    while position < len(present):
        end = position
        while end + 1 < len(present) and present[end + 1][0] == present[position][0]:
            end += 1
        # positions position..end (0-based) tie: they share ranks position+1..end+1
        shared = (position + 1 + end + 1) / 2
        for k in range(position, end + 1):
            ranks[present[k][1]] = shared
        position = end + 1
    return ranks


def best_indices(rows: List[dict], metric: str) -> set:
    """Indices of the row(s) best on this metric (several if tied for first)."""
    ranks = metric_ranks(rows, metric)
    present = [r for r in ranks if r is not None]
    if not present:
        return set()
    top = min(present)
    return {i for i, r in enumerate(ranks) if r == top}


def usable_metrics(rows: List[dict], metrics: List[str]) -> tuple:
    """Split metrics into (usable, unusable): unusable ones have no value on any row."""
    usable = [m for m in metrics if any(_value(r, m) is not None for r in rows)]
    return usable, [m for m in metrics if m not in usable]


def overall_ranking(rows: List[dict], metrics: List[str]) -> List[dict]:
    """
    Order rows by average rank over metrics, best first. Returns one entry
    per row: {"index", "row", "ranks": {metric: rank or None}, "avg_rank",
    "complete"}.

    Sort order: complete rows before incomplete ones; then lower average
    rank; then, to break exact ties, the per-metric ranks in the order the
    metrics were given (so the first metric acts as the tie-breaker); then
    original order, keeping the sort deterministic.
    """
    per_metric = {m: metric_ranks(rows, m) for m in metrics}
    entries = []
    for i, row in enumerate(rows):
        ranks = {m: per_metric[m][i] for m in metrics}
        present = [r for r in ranks.values() if r is not None]
        entries.append({
            "index": i,
            "row": row,
            "ranks": ranks,
            "avg_rank": sum(present) / len(present) if present else None,
            "complete": len(present) == len(metrics),
        })

    def key(entry):
        avg = entry["avg_rank"]
        tie_break = tuple(math.inf if entry["ranks"][m] is None else entry["ranks"][m] for m in metrics)
        return (not entry["complete"], avg is None, avg if avg is not None else 0.0, tie_break, entry["index"])

    return sorted(entries, key=key)
