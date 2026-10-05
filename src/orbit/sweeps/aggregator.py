import pathlib

import numpy as np

from orbit.core.ranking import overall_ranking
from orbit.storage import experiment_dir, load_results
from typing import Optional
from orbit.storage.workspace import experiments_root


def run_status(run_name: str, root: Optional[pathlib.Path] = None) -> str:
    """
    Status is derived from disk, never stored in sweep.json, so it can't
    drift out of sync: results.json present -> "done"; experiment.json
    gone (e.g. `orbit delete`d) -> "missing"; otherwise "pending".
    """
    root = experiments_root(root)
    exp_dir = experiment_dir(run_name, root=root)
    if (exp_dir / "results" / "results.json").exists():
        return "done"
    if not (exp_dir / "experiment.json").exists():
        return "missing"
    return "pending"


def collect_rows(sweep: dict, root: Optional[pathlib.Path] = None) -> list:
    root = experiments_root(root)
    rows = []
    for run in sweep["runs"]:
        status = run_status(run["name"], root)
        row = {
            "run": run["name"],
            "params": run["params"],
            "status": status,
            "final_loss": None,
            "test_loss": None,
            "accuracy": None,
            "test_accuracy": None,
            "duration_seconds": None,
            "task": None,
            "diverged_at_epoch": None,
        }
        if status == "done":
            results = load_results(experiment_dir(run["name"], root=root) / "results" / "results.json")
            row.update(
                final_loss=results.final_loss,
                test_loss=results.test_loss,
                accuracy=results.accuracy_history[-1] if results.accuracy_history else None,
                test_accuracy=results.test_accuracy,
                duration_seconds=results.duration_seconds,
                task=results.hyperparams.get("task"),
                diverged_at_epoch=results.diverged_at_epoch,
            )
        rows.append(row)
    return rows


def default_rank_metric(sweep: dict) -> str:
    return "test_loss" if "test_split" in sweep["base_config"] or "test_split" in sweep["grid"] else "final_loss"


def varies_seed(sweep: dict) -> bool:
    return len(sweep["grid"].get("seed", [])) > 1


def group_by_seed(rows: list, metrics: list) -> list:
    """
    Group runs that differ only by seed and summarize each metric per group
    as mean +- std (population std, np.std's default) over the runs that
    have a value. Groups are then ranked like single runs - by average rank
    over the metrics, computed on the group means (see core/ranking.py).
    Each summary carries "ranks"/"avg_rank"/"complete" from that ranking.
    """
    groups = {}
    for row in rows:
        params = {k: v for k, v in row["params"].items() if k != "seed"}
        key = tuple(sorted((k, repr(v)) for k, v in params.items()))
        group = groups.setdefault(
            key, {"params": params, "runs": [], "values": {m: [] for m in metrics}, "task": None}
        )
        group["runs"].append(row["run"])
        for m in metrics:
            if row[m] is not None and not np.isnan(row[m]):
                group["values"][m].append(row[m])
        group["task"] = group["task"] or row["task"]

    summaries = []
    for group in groups.values():
        values = group["values"]
        summary = {
            "params": group["params"],
            "runs": group["runs"],
            "n": max((len(v) for v in values.values()), default=0),
            "mean": {m: float(np.mean(v)) if v else None for m, v in values.items()},
            "std": {m: float(np.std(v)) if v else None for m, v in values.items()},
            "task": group["task"],
        }
        # flat metric keys so the group can be ranked like a run
        summary.update(summary["mean"])
        summaries.append(summary)

    ranked = []
    for entry in overall_ranking(summaries, metrics):
        summary = entry["row"]
        summary.update(ranks=entry["ranks"], avg_rank=entry["avg_rank"], complete=entry["complete"])
        ranked.append(summary)
    return ranked
