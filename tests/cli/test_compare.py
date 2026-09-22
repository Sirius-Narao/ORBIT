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
