import json

import numpy as np

from orbit.core import Results
from orbit.storage import save_results
from orbit.sweeps.aggregator import (
    collect_rows,
    default_rank_metric,
    group_by_seed,
    run_status,
    varies_seed,
)


def make_run(root, name, final_loss=None, test_loss=None, test_accuracy=None, accuracy_history=None):
    exp_dir = root / name
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(json.dumps({"name": name}))
    if final_loss is not None:
        results = Results(
            final_loss=final_loss, loss_history=[1.0, final_loss], name=name,
            test_loss=test_loss, test_accuracy=test_accuracy, accuracy_history=accuracy_history,
        )
        save_results(results, exp_dir / "results" / "results.json")


def row(run, params, **metrics):
    base = {"run": run, "params": params, "status": "done", "final_loss": None, "test_loss": None,
            "accuracy": None, "test_accuracy": None, "duration_seconds": None, "task": None}
    base.update(metrics)
    return base


def test_run_status_derived_from_disk(tmp_path):
    make_run(tmp_path, "s_001", final_loss=0.1)
    make_run(tmp_path, "s_002")

    assert run_status("s_001", tmp_path) == "done"
    assert run_status("s_002", tmp_path) == "pending"
    assert run_status("s_003", tmp_path) == "missing"


def test_collect_rows_reads_metrics_for_done_runs(tmp_path):
    make_run(tmp_path, "s_001", final_loss=0.1, test_loss=0.2, accuracy_history=[0.5, 0.75])
    make_run(tmp_path, "s_002")
    sweep = {"runs": [{"name": "s_001", "params": {"seed": 1}}, {"name": "s_002", "params": {"seed": 2}}]}

    rows = collect_rows(sweep, tmp_path)

    assert rows[0]["status"] == "done"
    assert rows[0]["final_loss"] == 0.1
    assert rows[0]["test_loss"] == 0.2
    assert rows[0]["accuracy"] == 0.75  # final-epoch training accuracy
    assert rows[1]["status"] == "pending"
    assert rows[1]["final_loss"] is None


def test_default_rank_metric_uses_test_loss_only_with_a_split():
    assert default_rank_metric({"base_config": {}, "grid": {}}) == "final_loss"
    assert default_rank_metric({"base_config": {"test_split": 0.2}, "grid": {}}) == "test_loss"
    assert default_rank_metric({"base_config": {}, "grid": {"test_split": [0.2]}}) == "test_loss"


def test_varies_seed():
    assert varies_seed({"grid": {"seed": [1, 2]}})
    assert not varies_seed({"grid": {"seed": [1]}})
    assert not varies_seed({"grid": {"learning_rate": [0.1, 0.2]}})


def test_group_by_seed_mean_and_std():
    """
    lr=0.1 over seeds: losses 0.2, 0.4 -> mean 0.3, population std 0.1
    lr=0.5 over seeds: losses 0.1, 0.1 -> mean 0.1, std 0.0  (best: ranked first)
    """
    rows = [
        row("a", {"learning_rate": 0.1, "seed": 1}, final_loss=0.2),
        row("b", {"learning_rate": 0.1, "seed": 2}, final_loss=0.4),
        row("c", {"learning_rate": 0.5, "seed": 1}, final_loss=0.1),
        row("d", {"learning_rate": 0.5, "seed": 2}, final_loss=0.1),
    ]

    groups = group_by_seed(rows, ["final_loss"])

    assert [g["params"] for g in groups] == [{"learning_rate": 0.5}, {"learning_rate": 0.1}]
    assert groups[0]["n"] == 2
    assert np.isclose(groups[0]["mean"]["final_loss"], 0.1) and np.isclose(groups[0]["std"]["final_loss"], 0.0)
    assert np.isclose(groups[1]["mean"]["final_loss"], 0.3) and np.isclose(groups[1]["std"]["final_loss"], 0.1)
    assert groups[1]["runs"] == ["a", "b"]


def test_group_by_seed_counts_only_runs_with_values():
    rows = [
        row("a", {"learning_rate": 0.1, "seed": 1}, final_loss=0.2),
        row("b", {"learning_rate": 0.1, "seed": 2}, status="pending"),
    ]

    groups = group_by_seed(rows, ["final_loss"])

    assert groups[0]["n"] == 1
    assert groups[0]["mean"]["final_loss"] == 0.2


def test_group_by_seed_ranks_groups_by_average_rank_over_several_metrics():
    """
    Group means:      test_loss   test_accuracy
      lr=0.1            0.30          0.90       -> ranks 3, 1 -> avg 2.0
      lr=0.5            0.10          0.70       -> ranks 1, 3 -> avg 2.0
      lr=1.0            0.20          0.80       -> ranks 2, 2 -> avg 2.0
    All tie on average rank, so the first metric (test_loss) breaks the tie:
    lr=0.5, then lr=1.0, then lr=0.1.
    """
    rows = []
    for lr, loss, acc in [(0.1, 0.3, 0.9), (0.5, 0.1, 0.7), (1.0, 0.2, 0.8)]:
        for seed in (1, 2):
            rows.append(row(f"{lr}-{seed}", {"learning_rate": lr, "seed": seed}, test_loss=loss, test_accuracy=acc))

    groups = group_by_seed(rows, ["test_loss", "test_accuracy"])

    assert [g["params"]["learning_rate"] for g in groups] == [0.5, 1.0, 0.1]
    assert [g["avg_rank"] for g in groups] == [2.0, 2.0, 2.0]
    assert groups[2]["ranks"] == {"test_loss": 3.0, "test_accuracy": 1.0}
