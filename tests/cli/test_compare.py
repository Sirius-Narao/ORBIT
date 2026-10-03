import json

from orbit.core import Results
from orbit.cli.commands.compare import compare_experiments, _comparison_filename


def write_config(root, name, epochs=5):
    exp_dir = root / name
    exp_dir.mkdir(parents=True)

    config = {
        "name": name,
        "dataset": "xor",
        "model": [
            {"type": "Linear", "in_features": 2, "neurons": 8},
            {"type": "Tanh"},
            {"type": "Linear", "neurons": 1},
            {"type": "Sigmoid"},
        ],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": 2.0,
        "batch_size": 4,
        "epochs": epochs,
    }

    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)

    return exp_dir


def write_results(exp_dir, final_loss=0.1234, loss_history=None):
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True)

    results = Results(
        name=exp_dir.name,
        final_loss=final_loss,
        loss_history=loss_history if loss_history is not None else [0.5, 0.3, final_loss],
    )

    with open(results_dir / "results.json", "w") as f:
        json.dump(results.to_dict(), f)


def test_compare_experiments_named(tmp_path, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, final_loss=0.1234)
    write_config(tmp_path, "exp_b")

    compare_experiments(["exp_a", "exp_b"], root=tmp_path)

    out = capsys.readouterr().out
    assert "exp_a" in out
    assert "exp_b" in out
    assert "0.1234" in out
    assert "not run" in out


def test_compare_experiments_skips_missing_name(tmp_path, capsys):
    write_config(tmp_path, "exp_a")

    compare_experiments(["exp_a", "does_not_exist"], root=tmp_path)

    out = capsys.readouterr().out
    assert "exp_a" in out
    assert "does_not_exist was not found, skipping." in out


def test_compare_experiments_all(tmp_path, capsys):
    write_config(tmp_path, "exp_a")
    write_config(tmp_path, "exp_b")

    compare_experiments(is_all=True, root=tmp_path)

    out = capsys.readouterr().out
    assert "exp_a" in out
    assert "exp_b" in out


def test_compare_experiments_all_empty_dir(tmp_path, capsys):
    compare_experiments(is_all=True, root=tmp_path)

    out = capsys.readouterr().out
    assert "No experiments found" in out


def test_compare_experiments_all_missing_root(tmp_path, capsys):
    compare_experiments(is_all=True, root=tmp_path / "does_not_exist")

    out = capsys.readouterr().out
    assert "No experiments found" in out


def test_compare_experiments_no_names_and_not_all(tmp_path, capsys):
    compare_experiments([], root=tmp_path)

    out = capsys.readouterr().out
    assert "No experiments found" in out


def test_compare_experiments_plotloss_saves_comparison_png(tmp_path, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, final_loss=0.1234)
    exp_b = write_config(tmp_path, "exp_b")
    write_results(exp_b, final_loss=0.5678)

    comparisons_root = tmp_path / "comparisons"
    compare_experiments(
        ["exp_a", "exp_b"], plot_loss=True, root=tmp_path, comparisons_root=comparisons_root
    )

    saved = list(comparisons_root.glob("*.png"))
    assert len(saved) == 1
    assert saved[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    out = capsys.readouterr().out
    assert "Saved comparison plot to" in out


def test_compare_experiments_plotloss_warns_when_nothing_has_run(tmp_path, capsys):
    write_config(tmp_path, "exp_a")
    write_config(tmp_path, "exp_b")

    comparisons_root = tmp_path / "comparisons"
    compare_experiments(
        ["exp_a", "exp_b"], plot_loss=True, root=tmp_path, comparisons_root=comparisons_root
    )

    out = capsys.readouterr().out
    assert "No experiments with results to plot." in out
    assert not comparisons_root.exists()


def test_compare_experiments_plotloss_skips_unrun_experiments(tmp_path, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, final_loss=0.1234)
    write_config(tmp_path, "exp_b")  # never run

    comparisons_root = tmp_path / "comparisons"
    compare_experiments(
        ["exp_a", "exp_b"], plot_loss=True, root=tmp_path, comparisons_root=comparisons_root
    )

    saved = list(comparisons_root.glob("*.png"))
    assert len(saved) == 1

    out = capsys.readouterr().out
    assert "Saved comparison plot to" in out


def test_compare_experiments_plotloss_passes_log_scale_flag(tmp_path, monkeypatch):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, final_loss=0.1234)
    exp_b = write_config(tmp_path, "exp_b")
    write_results(exp_b, final_loss=0.5678)

    calls = []
    monkeypatch.setattr(
        "orbit.cli.commands.compare.plot_loss_comparison",
        lambda results_list, output_path, log_scale=False: calls.append(log_scale) or output_path,
    )

    compare_experiments(
        ["exp_a", "exp_b"],
        plot_loss=True,
        log_scale=True,
        root=tmp_path,
        comparisons_root=tmp_path / "comparisons",
    )

    assert calls == [True]


def test_comparison_filename_without_log_scale():
    assert _comparison_filename(["exp_a", "exp_b"], log_scale=False) == "exp_a_vs_exp_b.png"


def test_comparison_filename_with_log_scale():
    assert _comparison_filename(["exp_a", "exp_b"], log_scale=True) == "exp_a_vs_exp_b_log_scale.png"


def test_comparison_filename_with_metric():
    assert _comparison_filename(["exp_a", "exp_b"], log_scale=False, metric="loss") == "exp_a_vs_exp_b_loss.png"


def test_comparison_filename_with_metric_and_log_scale():
    assert (
        _comparison_filename(["exp_a", "exp_b"], log_scale=True, metric="loss")
        == "exp_a_vs_exp_b_loss_log_scale.png"
    )


def test_comparison_filename_long_names_with_log_scale():
    names = [f"experiment_number_{i}" for i in range(10)]  # joined name exceeds 100 chars

    filename = _comparison_filename(names, log_scale=True)

    assert filename == f"comparison_{len(names)}_experiments_log_scale.png"


def test_compare_experiments_plotloss_and_logscale_produce_separate_files(tmp_path):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, final_loss=0.1234)
    exp_b = write_config(tmp_path, "exp_b")
    write_results(exp_b, final_loss=0.5678)

    comparisons_root = tmp_path / "comparisons"
    compare_experiments(
        ["exp_a", "exp_b"], plot_loss=True, log_scale=False, root=tmp_path, comparisons_root=comparisons_root
    )
    compare_experiments(
        ["exp_a", "exp_b"], plot_loss=True, log_scale=True, root=tmp_path, comparisons_root=comparisons_root
    )

    saved = list(comparisons_root.glob("*.png"))
    assert len(saved) == 2


# --- ranking (--by) ------------------------------------------------------------

def write_ranked(root, name, final_loss, test_loss=None, test_accuracy=None, duration=None, task=None):
    exp_dir = write_config(root, name)
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True)
    results = Results(
        name=name,
        final_loss=final_loss,
        loss_history=[0.5, final_loss],
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        duration_seconds=duration,
        hyperparams={"task": task} if task else {},
    )
    with open(results_dir / "results.json", "w") as f:
        json.dump(results.to_dict(), f)


def _three_experiments(tmp_path):
    """
               test_loss  test_accuracy  duration   ranks      avg
      exp_a      0.10        0.80          5.0      1, 2, 3    2.0
      exp_b      0.20        0.90          1.0      2, 1, 1    1.33  <- best overall
      exp_c      0.30        0.70          2.0      3, 3, 2    2.67
    """
    write_ranked(tmp_path, "exp_a", 0.05, test_loss=0.10, test_accuracy=0.80, duration=5.0,
                 task="binary_classification")
    write_ranked(tmp_path, "exp_b", 0.06, test_loss=0.20, test_accuracy=0.90, duration=1.0,
                 task="binary_classification")
    write_ranked(tmp_path, "exp_c", 0.07, test_loss=0.30, test_accuracy=0.70, duration=2.0,
                 task="binary_classification")


def test_compare_without_by_does_not_rank(tmp_path, capsys):
    _three_experiments(tmp_path)

    assert compare_experiments(["exp_c", "exp_a"], root=tmp_path) is None

    out = capsys.readouterr().out
    assert "Best" not in out
    assert out.index("exp_c") < out.index("exp_a")


def test_compare_by_several_metrics_ranks_by_average_rank(tmp_path, capsys):
    _three_experiments(tmp_path)

    ranked = compare_experiments(
        is_all=True, root=tmp_path, by=["test_loss", "test_accuracy", "duration_seconds"]
    )
    out = capsys.readouterr().out

    assert [r["name"] for r in ranked] == ["exp_b", "exp_a", "exp_c"]
    assert "Best run overall: exp_b - average rank 1.333" in out
    assert "Best Test Loss ★ : exp_a - 0.1" in out
    assert "Best Test Accuracy ★ : exp_b - 90.00%" in out
    assert "Best Duration ★ : exp_b - 1.00s" in out
    assert "Avg Rank" in out


def test_compare_by_single_metric_has_no_avg_rank_column(tmp_path, capsys):
    _three_experiments(tmp_path)

    ranked = compare_experiments(is_all=True, root=tmp_path, by=["test_loss"])
    out = capsys.readouterr().out

    assert [r["name"] for r in ranked] == ["exp_a", "exp_b", "exp_c"]
    assert "Best run: exp_a - test_loss 0.1" in out
    assert "Avg Rank" not in out


def test_compare_by_puts_unrun_experiments_last(tmp_path, capsys):
    _three_experiments(tmp_path)
    write_config(tmp_path, "exp_unrun")

    ranked = compare_experiments(
        ["exp_unrun", "exp_c", "exp_b"], root=tmp_path, by=["test_loss", "test_accuracy"]
    )

    assert [r["name"] for r in ranked][-1] == "exp_unrun"
    assert "not run" in capsys.readouterr().out


def test_compare_by_drops_a_metric_nobody_has(tmp_path, capsys):
    write_ranked(tmp_path, "exp_a", 0.2)
    write_ranked(tmp_path, "exp_b", 0.1)

    ranked = compare_experiments(is_all=True, root=tmp_path, by=["final_loss", "test_loss"])
    out = capsys.readouterr().out

    assert "not ranking by it" in out
    assert [r["name"] for r in ranked] == ["exp_b", "exp_a"]


def test_compare_bare_by_prompts_for_metrics(tmp_path, monkeypatch, capsys):
    import questionary

    _three_experiments(tmp_path)

    class FakeAnswer:
        def ask(self):
            return ["Test Loss", "Duration"]

    monkeypatch.setattr(questionary, "checkbox", lambda *a, **k: FakeAnswer())

    ranked = compare_experiments(is_all=True, root=tmp_path, by=[])

    # exp_a: ranks 1, 3 -> 2.0   exp_b: 2, 1 -> 1.5   exp_c: 3, 2 -> 2.5
    assert [r["name"] for r in ranked] == ["exp_b", "exp_a", "exp_c"]


# --- --test ----------------------------------------------------------------------

def test_compare_test_only_ranks_by_test_metrics_and_ignores_training_fit(tmp_path, capsys):
    """
    exp_a fits its training data best (final_loss 0.05) but exp_b
    generalizes best; --test ranks on test metrics alone:
               test_loss  test_accuracy   ranks   avg
      exp_a      0.10        0.80         1, 2    1.5
      exp_b      0.20        0.90         2, 1    1.5   (tie -> test_loss decides: exp_a)
      exp_c      0.30        0.70         3, 3    3.0
    """
    _three_experiments(tmp_path)

    ranked = compare_experiments(is_all=True, root=tmp_path, test_only=True)
    out = capsys.readouterr().out

    assert "Ranking by test metrics only: test_loss, test_accuracy" in out
    assert "ranked by average rank over test_loss, test_accuracy" in out
    assert "Final Loss" not in out.split("Best run")[0].split("ranked by")[1]  # not a ranked column
    assert [r["name"] for r in ranked] == ["exp_a", "exp_b", "exp_c"]


def test_compare_test_only_uses_whichever_test_metrics_exist(tmp_path, capsys):
    write_ranked(tmp_path, "exp_a", 0.05, test_loss=0.3)
    write_ranked(tmp_path, "exp_b", 0.06, test_loss=0.2)

    ranked = compare_experiments(is_all=True, root=tmp_path, test_only=True)
    out = capsys.readouterr().out

    assert "Ranking by test metrics only: test_loss" in out
    assert "not ranking by it" not in out
    assert [r["name"] for r in ranked] == ["exp_b", "exp_a"]


def test_compare_test_only_without_test_metrics_warns_and_shows_plain_table(tmp_path, capsys):
    write_ranked(tmp_path, "exp_a", 0.05)

    assert compare_experiments(is_all=True, root=tmp_path, test_only=True) is None
    out = capsys.readouterr().out

    assert "No test metrics recorded" in out
    assert "orbit test" in out
    assert "exp_a" in out


def test_compare_rejects_by_together_with_test_only(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        compare_experiments(is_all=True, root=tmp_path, by=["test_loss"], test_only=True)
