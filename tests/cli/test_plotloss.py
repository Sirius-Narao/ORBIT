import json

from orbit.core import Results
from orbit.cli.commands.plotloss import plot_experiment


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


def test_plot_experiment_missing_name(tmp_path, capsys):
    result = plot_experiment("does_not_exist", root=tmp_path)

    out = capsys.readouterr().out
    assert "was not found" in out
    assert result is None


def test_plot_experiment_not_run_yet(tmp_path, capsys):
    write_config(tmp_path, "xor_test")

    result = plot_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "has not been run yet" in out
    assert result is None
    assert not (tmp_path / "xor_test" / "results" / "loss.png").exists()


def test_plot_experiment_saves_png_after_run(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(exp_dir, final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])

    result = plot_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "Saved" in out
    assert "xor_test" in out
    assert result == exp_dir / "results" / "loss.png"
    assert result.exists()
    assert result.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_experiment_passes_log_scale_flag(tmp_path, monkeypatch):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(exp_dir, final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])

    calls = []
    monkeypatch.setattr(
        "orbit.cli.commands.plotloss.plot_loss",
        lambda results, output_path, log_scale=False: calls.append(log_scale) or output_path,
    )

    plot_experiment("xor_test", root=tmp_path, log_scale=True)

    assert calls == [True]
