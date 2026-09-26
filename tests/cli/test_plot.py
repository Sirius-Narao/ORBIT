import json

from orbit.core import Results
from orbit.cli.commands.plot import plot_experiments


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


def write_results(
    exp_dir,
    final_loss=0.1234,
    loss_history=None,
    accuracy_history=None,
    gradient_norm_history=None,
    test_loss=None,
    test_accuracy=None,
):
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results = Results(
        name=exp_dir.name,
        final_loss=final_loss,
        loss_history=loss_history if loss_history is not None else [0.5, 0.3, final_loss],
        accuracy_history=accuracy_history,
        gradient_norm_history=gradient_norm_history,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
    )

    with open(results_dir / "results.json", "w") as f:
        json.dump(results.to_dict(), f)

    return results


def test_plot_experiments_no_names_and_not_all(tmp_path, capsys):
    result = plot_experiments([], root=tmp_path)

    assert result == []
    assert "No experiments found" in capsys.readouterr().out


def test_plot_experiments_all_with_missing_root(tmp_path, capsys):
    result = plot_experiments(is_all=True, root=tmp_path / "does_not_exist")

    assert result == []
    assert "No experiments found" in capsys.readouterr().out


def test_plot_experiments_skips_missing_and_unrun(tmp_path, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a)
    write_config(tmp_path, "exp_b")  # never run

    result = plot_experiments(
        ["exp_a", "exp_b", "does_not_exist"], metrics=["loss"], root=tmp_path
    )

    out = capsys.readouterr().out
    assert "does_not_exist was not found, skipping." in out
    assert "exp_b has not been run yet, skipping." in out
    assert len(result) == 1
    assert result[0] == exp_a / "results" / "loss.png"


def test_plot_experiments_single_experiment_single_metric(tmp_path):
    exp_a = write_config(tmp_path, "exp_a", epochs=3)
    write_results(exp_a, loss_history=[0.5, 0.3, 0.1])

    result = plot_experiments(["exp_a"], metrics=["loss"], root=tmp_path)

    assert result == [exp_a / "results" / "loss.png"]
    assert result[0].exists()
    assert result[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_experiments_single_experiment_multiple_metrics(tmp_path):
    exp_a = write_config(tmp_path, "exp_a", epochs=3)
    write_results(
        exp_a, loss_history=[0.5, 0.3, 0.1], accuracy_history=[0.6, 0.8, 0.9]
    )

    result = plot_experiments(["exp_a"], metrics=["loss", "accuracy"], root=tmp_path)

    assert set(result) == {
        exp_a / "results" / "loss.png",
        exp_a / "results" / "accuracy.png",
    }
    for path in result:
        assert path.exists()


def test_plot_experiments_missing_metric_warns_and_skips(tmp_path, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a)  # no accuracy_history

    result = plot_experiments(["exp_a"], metrics=["accuracy"], root=tmp_path)

    out = capsys.readouterr().out
    assert "exp_a has no recorded accuracy" in out
    assert "No experiments have accuracy" in out
    assert result == []


def test_plot_experiments_comparison_filename_for_two_experiments(tmp_path):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, loss_history=[0.5, 0.3, 0.1])
    exp_b = write_config(tmp_path, "exp_b")
    write_results(exp_b, loss_history=[0.6, 0.4, 0.2])

    comparisons_root = tmp_path / "comparisons"
    result = plot_experiments(
        ["exp_a", "exp_b"], metrics=["loss"], root=tmp_path, comparisons_root=comparisons_root
    )

    assert result == [comparisons_root / "exp_a_vs_exp_b_loss.png"]
    assert result[0].exists()


def test_plot_experiments_comparison_skips_experiment_missing_metric(tmp_path, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, accuracy_history=[0.5, 0.7, 0.9])
    exp_b = write_config(tmp_path, "exp_b")
    write_results(exp_b)  # no accuracy_history

    comparisons_root = tmp_path / "comparisons"
    result = plot_experiments(
        ["exp_a", "exp_b"], metrics=["accuracy"], root=tmp_path, comparisons_root=comparisons_root
    )

    out = capsys.readouterr().out
    assert "exp_b has no recorded accuracy" in out
    # exp_b lacked accuracy, but exp_a's requested plot still happens - since
    # 2 experiments were requested overall, this still uses the comparison
    # (not single-experiment) output location, even though the filename
    # itself only names the experiment(s) actually eligible for this metric.
    assert result == [comparisons_root / "exp_a_accuracy.png"]
    assert result[0].exists()


def test_plot_experiments_test_loss_bar_chart(tmp_path):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, test_loss=0.42)
    exp_b = write_config(tmp_path, "exp_b")
    write_results(exp_b, test_loss=0.24)

    comparisons_root = tmp_path / "comparisons"
    result = plot_experiments(
        ["exp_a", "exp_b"], metrics=["test_loss"], root=tmp_path, comparisons_root=comparisons_root
    )

    assert result == [comparisons_root / "exp_a_vs_exp_b_test_loss.png"]
    assert result[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_experiments_test_accuracy_skips_untested_experiment(tmp_path, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, test_accuracy=0.9)
    exp_b = write_config(tmp_path, "exp_b")
    write_results(exp_b)  # trained but never tested

    result = plot_experiments(["exp_a", "exp_b"], metrics=["test_accuracy"], root=tmp_path)

    out = capsys.readouterr().out
    assert "exp_b has no recorded test accuracy" in out
    assert len(result) == 1


def test_plot_experiments_single_experiment_test_metric_bar_chart(tmp_path):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, test_loss=0.42)

    result = plot_experiments(["exp_a"], metrics=["test_loss"], root=tmp_path)

    assert result == [exp_a / "results" / "test_loss.png"]
    assert result[0].exists()


def test_plot_experiments_interactive_prompt_maps_labels_to_keys(tmp_path, monkeypatch):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, test_accuracy=0.9)

    class FakeAsk:
        def ask(self_inner):
            return ["Training loss", "Test accuracy / R²"]

    monkeypatch.setattr(
        "orbit.cli.commands.plot.questionary.checkbox",
        lambda *args, **kwargs: FakeAsk(),
    )

    result = plot_experiments(["exp_a"], root=tmp_path)

    assert set(result) == {
        exp_a / "results" / "loss.png",
        exp_a / "results" / "test_accuracy.png",
    }


def test_plot_experiments_interactive_prompt_empty_selection(tmp_path, monkeypatch, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a)

    class FakeAsk:
        def ask(self_inner):
            return None

    monkeypatch.setattr(
        "orbit.cli.commands.plot.questionary.checkbox",
        lambda *args, **kwargs: FakeAsk(),
    )

    result = plot_experiments(["exp_a"], root=tmp_path)

    assert result == []
    assert "No metrics selected" in capsys.readouterr().out


def test_plot_experiments_rejects_unknown_metrics_key(tmp_path):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a)

    try:
        plot_experiments(["exp_a"], metrics=["bogus"], root=tmp_path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "bogus" in str(e)


def test_plot_experiments_passes_log_scale_and_suffixes_filenames(tmp_path):
    exp_a = write_config(tmp_path, "exp_a", epochs=3)
    write_results(exp_a, loss_history=[0.5, 0.3, 0.1])

    result = plot_experiments(["exp_a"], metrics=["loss"], log_scale=True, root=tmp_path)

    assert result == [exp_a / "results" / "loss_log_scale.png"]
    assert result[0].exists()
