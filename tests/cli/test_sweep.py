import csv
import json

import pytest
import questionary

import orbit.cli.commands.sweep as sweep_module
from orbit.cli.commands.sweep import (
    compare_sweep,
    create_sweep,
    export_sweep,
    plot_sweep,
    start_sweep,
    sweep_status,
)
from orbit.sweeps.config import load_sweep


class FakeAnswer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value


def fake_prompts(monkeypatch, *, checkboxes, texts=(), confirm=True):
    """Hand back canned answers in call order, like test_new.py/test_copy.py."""
    checkboxes = iter(checkboxes)
    texts = iter(texts)
    monkeypatch.setattr(questionary, "checkbox", lambda *a, **k: FakeAnswer(next(checkboxes)))
    monkeypatch.setattr(questionary, "text", lambda *a, **k: FakeAnswer(next(texts)))
    monkeypatch.setattr(questionary, "confirm", lambda *a, **k: FakeAnswer(confirm))


def write_base(root, name="base_exp", **overrides):
    config = {
        "name": name,
        "dataset": "xor",
        "model": [
            {"type": "Linear", "in_features": 2, "neurons": 4},
            {"type": "Tanh"},
            {"type": "Linear", "neurons": 1},
            {"type": "Sigmoid"},
        ],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": 1.0,
        "batch_size": 4,
        "epochs": 5,
        "seed": 1,
    }
    config.update(overrides)
    exp_dir = root / name
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(json.dumps(config))
    return config


@pytest.fixture
def roots(tmp_path):
    return {"root": tmp_path / "experiments", "sweeps_root": tmp_path / "sweeps"}


def create_lr_seed_sweep(monkeypatch, roots, name="sw"):
    # vary learning_rate (text) and seed (text): 2 x 2 = 4 runs
    fake_prompts(monkeypatch, checkboxes=[["learning_rate", "seed"]], texts=["0.5, 1.0", "1,2"])
    return create_sweep(name, "base_exp", **roots)


def test_create_sweep_writes_manifest_and_one_experiment_per_run(monkeypatch, roots):
    base = write_base(roots["root"])

    path = create_lr_seed_sweep(monkeypatch, roots)

    assert path == roots["sweeps_root"] / "sw" / "sweep.json"
    sweep = load_sweep("sw", roots["sweeps_root"])
    assert sweep["base"] == "base_exp"
    assert sweep["base_config"] == base
    assert sweep["grid"] == {"learning_rate": [0.5, 1.0], "seed": [1, 2]}
    assert [r["name"] for r in sweep["runs"]] == ["sw_001", "sw_002", "sw_003", "sw_004"]

    with open(roots["root"] / "sw_003" / "experiment.json") as f:
        config = json.load(f)
    assert config["name"] == "sw_003"
    assert config["learning_rate"] == 1.0
    assert config["seed"] == 1
    assert config["model"] == base["model"]


def test_create_sweep_choice_field_uses_checkbox_values(monkeypatch, roots):
    write_base(roots["root"])
    fake_prompts(monkeypatch, checkboxes=[["optimizer"], ["SGD", "Adam"]])

    create_sweep("opt", "base_exp", **roots)

    sweep = load_sweep("opt", roots["sweeps_root"])
    assert [r["params"] for r in sweep["runs"]] == [{"optimizer": "SGD"}, {"optimizer": "Adam"}]


def test_create_sweep_missing_base(roots, capsys):
    assert create_sweep("sw", "nope", **roots) is None
    assert "was not found" in capsys.readouterr().out


def test_create_sweep_refuses_existing_sweep(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)

    assert create_sweep("sw", "base_exp", **roots) is None
    assert "already exists" in capsys.readouterr().out


def test_create_sweep_refuses_run_name_collision(monkeypatch, roots, capsys):
    write_base(roots["root"])
    write_base(roots["root"], name="sw_002")

    assert create_lr_seed_sweep(monkeypatch, roots) is None
    assert "sw_002" in capsys.readouterr().out
    assert not (roots["sweeps_root"] / "sw").exists()


def test_create_sweep_cancelled_writes_nothing(monkeypatch, roots):
    write_base(roots["root"])
    fake_prompts(monkeypatch, checkboxes=[["seed"]], texts=["1,2"], confirm=False)

    assert create_sweep("sw", "base_exp", **roots) is None
    assert not (roots["sweeps_root"] / "sw").exists()
    assert not (roots["root"] / "sw_001").exists()


def test_start_sweep_runs_everything_then_resumes_by_skipping(monkeypatch, roots):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)

    counts = start_sweep("sw", **roots)
    assert counts == {"done": 4, "failed": 0, "skipped": 0, "missing": 0}
    for i in range(1, 5):
        assert (roots["root"] / f"sw_00{i}" / "results" / "results.json").exists()

    assert start_sweep("sw", **roots)["skipped"] == 4

    # Losing one run's results re-runs only that one.
    (roots["root"] / "sw_002" / "results" / "results.json").unlink()
    counts = start_sweep("sw", **roots)
    assert counts["done"] == 1
    assert counts["skipped"] == 3


def test_start_sweep_continues_past_a_failing_run(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)

    real_run = sweep_module.run_experiment

    def flaky_run(name, root):
        if name == "sw_002":
            raise RuntimeError("boom")
        return real_run(name, root=root)

    monkeypatch.setattr(sweep_module, "run_experiment", flaky_run)
    counts = start_sweep("sw", **roots)

    assert counts["done"] == 3
    assert counts["failed"] == 1
    assert "sw_002 failed: boom" in capsys.readouterr().out
    assert (roots["root"] / "sw_004" / "results" / "results.json").exists()


def test_start_sweep_reports_deleted_runs_as_missing(monkeypatch, roots):
    import shutil

    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    shutil.rmtree(roots["root"] / "sw_001")

    counts = start_sweep("sw", **roots)

    assert counts["missing"] == 1
    assert counts["done"] == 3


def test_start_sweep_missing_sweep(roots, capsys):
    assert start_sweep("nope", **roots) is None
    assert "was not found" in capsys.readouterr().out


def test_sweep_status_reports_progress(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    capsys.readouterr()

    rows = sweep_status("sw", **roots)
    assert [r["status"] for r in rows] == ["pending"] * 4
    assert "0/4 done" in capsys.readouterr().out

    start_sweep("sw", **roots)
    capsys.readouterr()
    sweep_status("sw", **roots)
    assert "4/4 done" in capsys.readouterr().out


def test_compare_sweep_ranks_and_groups_over_seeds(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    ranked = compare_sweep("sw", **roots)
    out = capsys.readouterr().out

    losses = [r["final_loss"] for r in ranked]
    assert losses == sorted(losses)
    assert f"Best run: {ranked[0]['run']}" in out
    assert "Grouped over seeds" in out
    assert "Best configuration over seeds" in out


def test_compare_sweep_without_seed_variation_has_no_grouped_table(monkeypatch, roots, capsys):
    write_base(roots["root"])
    fake_prompts(monkeypatch, checkboxes=[["learning_rate"]], texts=["0.5,1.0"])
    create_sweep("sw", "base_exp", **roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    compare_sweep("sw", **roots)

    assert "Grouped over seeds" not in capsys.readouterr().out


def test_compare_sweep_before_any_runs_warns(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    capsys.readouterr()

    compare_sweep("sw", **roots)

    assert "orbit sweep start sw" in capsys.readouterr().out


def test_compare_sweep_defaults_to_test_loss_with_a_split(monkeypatch, roots, capsys):
    write_base(roots["root"], test_split=0.25)
    fake_prompts(monkeypatch, checkboxes=[["learning_rate"]], texts=["0.5,1.0"])
    create_sweep("sw", "base_exp", **roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    compare_sweep("sw", **roots)

    assert "ranked by test_loss" in capsys.readouterr().out


def test_export_sweep_writes_csv(monkeypatch, roots):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)
    (roots["root"] / "sw_004" / "results" / "results.json").unlink()

    path = export_sweep("sw", **roots)

    assert path == roots["sweeps_root"] / "sw" / "results.csv"
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == [
        "run", "learning_rate", "seed", "status",
        "final_loss", "test_loss", "test_accuracy", "duration_seconds",
    ]
    assert rows[1][:4] == ["sw_001", "0.5", "1", "done"]
    assert rows[1][4] != ""
    assert rows[1][5] == ""  # no test split
    assert rows[4][:4] == ["sw_004", "1.0", "2", "pending"]
    assert rows[4][4] == ""


def test_export_sweep_custom_output(monkeypatch, roots, tmp_path):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)

    path = export_sweep("sw", output=str(tmp_path / "out" / "x.csv"), **roots)

    assert path == tmp_path / "out" / "x.csv"
    assert path.exists()


def test_plot_sweep_writes_under_the_sweeps_plots_dir(monkeypatch, roots):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)

    paths = plot_sweep("sw", metrics=["loss"], **roots)

    assert len(paths) == 1
    assert paths[0].parent == roots["sweeps_root"] / "sw" / "plots"
    assert paths[0].exists()


def test_plot_sweep_before_any_runs_warns(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    capsys.readouterr()

    assert plot_sweep("sw", metrics=["loss"], **roots) == []
    assert "orbit sweep start sw" in capsys.readouterr().out


def test_compare_sweep_by_several_metrics(monkeypatch, roots, capsys):
    write_base(roots["root"], test_split=0.25, task="binary_classification")
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    ranked = compare_sweep("sw", by=["test_loss", "accuracy", "duration_seconds"], **roots)
    out = capsys.readouterr().out

    assert "ranked by average rank over test_loss, accuracy, duration_seconds" in out
    assert f"Best run overall: {ranked[0]['run']}" in out
    assert "Best Test Loss ★" in out
    assert "Best Accuracy ★" in out
    assert "Best Duration ★" in out
    # seed groups are ranked on the same metrics
    assert "Grouped over seeds (test_loss, accuracy, duration_seconds, mean ± std)" in out
    assert "Best Test Loss over seeds ★" in out


def test_compare_sweep_returns_rows_in_average_rank_order(monkeypatch, roots):
    from orbit.core.ranking import overall_ranking

    write_base(roots["root"], test_split=0.25)
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)

    ranked = compare_sweep("sw", by=["test_loss", "final_loss"], **roots)

    expected = [e["row"]["run"] for e in overall_ranking(ranked, ["test_loss", "final_loss"])]
    assert [r["run"] for r in ranked] == expected


def test_compare_sweep_all_metrics_ranks_by_every_recorded_metric(monkeypatch, roots, capsys):
    write_base(roots["root"], test_split=0.25, task="binary_classification")
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    ranked = compare_sweep("sw", all_metrics=True, **roots)
    out = capsys.readouterr().out

    every = "final_loss, test_loss, accuracy, test_accuracy"
    assert f"Ranking by every recorded quality metric: {every}" in out
    assert f"ranked by average rank over {every}" in out
    assert f"Best run overall: {ranked[0]['run']}" in out
    assert "Best Duration" not in out  # duration is excluded from --all
    for label in ("Final Loss", "Test Loss", "Accuracy", "Test Accuracy"):
        assert f"Best {label} ★" in out


def test_compare_sweep_all_metrics_quietly_skips_metrics_nobody_recorded(monkeypatch, roots, capsys):
    # No test split and no task: of the quality metrics only final_loss exists.
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    compare_sweep("sw", all_metrics=True, **roots)
    out = capsys.readouterr().out

    assert "Ranking by every recorded quality metric: final_loss" in out
    assert "duration_seconds" not in out
    assert "not ranking by it" not in out


def test_compare_sweep_all_metrics_before_any_runs_warns(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    capsys.readouterr()

    compare_sweep("sw", all_metrics=True, **roots)

    assert "orbit sweep start sw" in capsys.readouterr().out


def test_compare_sweep_rejects_by_together_with_all_metrics(roots):
    with pytest.raises(ValueError):
        compare_sweep("sw", by=["test_loss"], all_metrics=True, **roots)


# --- delete_sweep ----------------------------------------------------------------

from orbit.cli.commands.sweep import delete_sweep


def test_delete_sweep_confirmed_removes_runs_and_manifest_only(monkeypatch, roots, capsys):
    write_base(roots["root"])
    write_base(roots["root"], name="sw_extra")  # shares the prefix, not in the sweep
    create_lr_seed_sweep(monkeypatch, roots)
    monkeypatch.setattr(questionary, "confirm", lambda *a, **k: FakeAnswer(True))

    assert delete_sweep("sw", **roots) == 4

    for i in range(1, 5):
        assert not (roots["root"] / f"sw_00{i}").exists()
    assert not (roots["sweeps_root"] / "sw").exists()
    assert (roots["root"] / "sw_extra").exists()
    assert (roots["root"] / "base_exp").exists()
    assert "Deleted sweep sw (4 runs)." in capsys.readouterr().out


def test_delete_sweep_declined_changes_nothing(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    monkeypatch.setattr(questionary, "confirm", lambda *a, **k: FakeAnswer(False))

    assert delete_sweep("sw", **roots) is None

    assert (roots["sweeps_root"] / "sw" / "sweep.json").exists()
    assert (roots["root"] / "sw_001").exists()
    assert "nothing was deleted" in capsys.readouterr().out


def test_delete_sweep_yes_skips_the_prompt(monkeypatch, roots):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)

    def no_prompt(*a, **k):
        raise AssertionError("yes=True must not prompt")

    monkeypatch.setattr(questionary, "confirm", no_prompt)

    assert delete_sweep("sw", yes=True, **roots) == 4
    assert not (roots["sweeps_root"] / "sw").exists()


def test_delete_sweep_counts_only_runs_still_on_disk(monkeypatch, roots):
    import shutil

    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    shutil.rmtree(roots["root"] / "sw_002")

    assert delete_sweep("sw", yes=True, **roots) == 3


def test_delete_sweep_missing(roots, capsys):
    assert delete_sweep("nope", yes=True, **roots) is None
    assert "was not found" in capsys.readouterr().out


def test_compare_sweep_by_duration_still_works(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    ranked = compare_sweep("sw", by=["duration_seconds"], **roots)

    durations = [r["duration_seconds"] for r in ranked]
    assert durations == sorted(durations)
    assert "ranked by duration_seconds" in capsys.readouterr().out


# --- architecture sweeps -------------------------------------------------------------

def test_create_architecture_sweep_prefills_from_the_base_model(monkeypatch, roots):
    write_base(roots["root"])  # 2 -> 4 -> Tanh -> 1 -> Sigmoid

    seen = []
    texts = iter(["2, 8", "1,2"])
    checkboxes = iter([["hidden_width", "hidden_depth", "activation"], ["Tanh", "ReLU"]])

    def fake_text(message, *a, **k):
        seen.append((message, k.get("default")))
        return FakeAnswer(next(texts))

    def fake_checkbox(message, *a, **k):
        seen.append((message, [c.title for c in k.get("choices", []) if getattr(c, "checked", False)]))
        return FakeAnswer(next(checkboxes))

    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "checkbox", fake_checkbox)
    monkeypatch.setattr(questionary, "confirm", lambda *a, **k: FakeAnswer(True))

    create_sweep("arch", "base_exp", **roots)

    assert ("hidden_width values (comma-separated):", "4") in seen
    assert ("hidden_depth values (comma-separated):", "1") in seen
    assert ("activation values:", ["Tanh"]) in seen

    sweep = load_sweep("arch", roots["sweeps_root"])
    assert len(sweep["runs"]) == 8  # 2 widths x 2 depths x 2 activations
    with open(roots["root"] / "arch_008" / "experiment.json") as f:
        config = json.load(f)
    assert config["model"] == [
        {"type": "Linear", "in_features": 2, "neurons": 8},
        {"type": "ReLU"},
        {"type": "Linear", "neurons": 8},
        {"type": "ReLU"},
        {"type": "Linear", "neurons": 1},
        {"type": "Sigmoid"},
    ]


def test_create_rejects_hidden_width_below_one(monkeypatch, roots):
    write_base(roots["root"])
    validators = []

    def fake_text(message, *a, **k):
        validators.append(k["validate"])
        return FakeAnswer("4")

    monkeypatch.setattr(questionary, "checkbox", lambda *a, **k: FakeAnswer(["hidden_width"]))
    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "confirm", lambda *a, **k: FakeAnswer(False))

    create_sweep("arch", "base_exp", **roots)

    validate = validators[0]
    assert validate("0, 4") == "hidden_width values must be at least 1"
    assert validate("1, 4") is True


def test_create_allows_hidden_depth_zero(monkeypatch, roots):
    write_base(roots["root"])
    validators = []

    def fake_text(message, *a, **k):
        validators.append(k["validate"])
        return FakeAnswer("0,1")

    monkeypatch.setattr(questionary, "checkbox", lambda *a, **k: FakeAnswer(["hidden_depth"]))
    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "confirm", lambda *a, **k: FakeAnswer(False))

    create_sweep("arch", "base_exp", **roots)

    assert validators[0]("0, 1") is True
    assert validators[0]("-1") == "hidden_depth values must be at least 0"


def test_create_reports_an_unbuildable_grid_and_writes_nothing(monkeypatch, roots, capsys):
    write_base(roots["root"], model=[{"type": "Linear", "in_features": 2, "neurons": 1}, {"type": "Sigmoid"}])
    fake_prompts(monkeypatch, checkboxes=[["hidden_depth"]], texts=["1,2"])

    assert create_sweep("arch", "base_exp", **roots) is None

    assert "also vary hidden_width" in capsys.readouterr().out
    assert not (roots["sweeps_root"] / "arch").exists()


def test_architecture_sweep_trains_and_compares(monkeypatch, roots, capsys):
    write_base(roots["root"])
    fake_prompts(monkeypatch, checkboxes=[["hidden_width", "hidden_depth"]], texts=["2,4", "0,1,2"])
    create_sweep("arch", "base_exp", **roots)
    capsys.readouterr()

    counts = start_sweep("arch", **roots)
    assert counts["done"] == 5  # depth 0 once + 2 widths x 2 depths
    assert counts["failed"] == 0

    compare_sweep("arch", **roots)
    out = capsys.readouterr().out
    assert "hidden_width" in out and "hidden_depth" in out


# --- --test ----------------------------------------------------------------------

def test_compare_sweep_test_only(monkeypatch, roots, capsys):
    write_base(roots["root"], test_split=0.25, task="binary_classification")
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    ranked = compare_sweep("sw", test_only=True, **roots)
    out = capsys.readouterr().out

    assert "Ranking by test metrics only: test_loss, test_accuracy" in out
    assert "ranked by average rank over test_loss, test_accuracy" in out
    assert "Best Final Loss" not in out
    assert "Best Test Loss ★" in out

    from orbit.core.ranking import overall_ranking
    expected = [e["row"]["run"] for e in overall_ranking(ranked, ["test_loss", "test_accuracy"])]
    assert [r["run"] for r in ranked] == expected


def test_compare_sweep_test_only_without_a_test_split_warns(monkeypatch, roots, capsys):
    write_base(roots["root"])
    create_lr_seed_sweep(monkeypatch, roots)
    start_sweep("sw", **roots)
    capsys.readouterr()

    compare_sweep("sw", test_only=True, **roots)
    out = capsys.readouterr().out

    assert "No test metrics recorded" in out
    assert "ranked by" not in out


def test_compare_sweep_rejects_combined_metric_options(roots):
    with pytest.raises(ValueError):
        compare_sweep("sw", test_only=True, all_metrics=True, **roots)
    with pytest.raises(ValueError):
        compare_sweep("sw", test_only=True, by=["final_loss"], **roots)
