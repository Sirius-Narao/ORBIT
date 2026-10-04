import json
import pytest
from orbit.core import Results
from orbit.cli.commands.run import run_experiment
from orbit.storage import checkpoint_exists


def write_experiment_config(root, name, epochs=5):
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


def test_run_experiment_returns_results_and_saves_them(tmp_path):
    write_experiment_config(tmp_path, "xor_test")

    results = run_experiment("xor_test", root=tmp_path)

    assert isinstance(results, Results)
    assert isinstance(results.final_loss, float)
    assert (tmp_path / "xor_test" / "results" / "results.json").exists()
    assert checkpoint_exists("xor_test", root=tmp_path)


def test_run_experiment_missing_experiment_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_experiment("does_not_exist", root=tmp_path)


def test_run_experiment_saves_training_history(tmp_path):
    from orbit.storage import load_snapshots

    write_experiment_config(tmp_path, "xor_test", epochs=5)

    run_experiment("xor_test", root=tmp_path)

    history = load_snapshots("xor_test", root=tmp_path)
    assert list(history["epochs"]) == [0, 1, 2, 3, 4, 5]


def test_run_experiment_removes_a_stale_history_when_none_is_recorded(tmp_path, monkeypatch):
    from orbit.storage import snapshots_exist

    write_experiment_config(tmp_path, "xor_test")
    run_experiment("xor_test", root=tmp_path)
    assert snapshots_exist("xor_test", root=tmp_path)

    monkeypatch.setattr("orbit.core.experiment.MAX_SNAPSHOT_PARAMETERS", 0)
    run_experiment("xor_test", root=tmp_path)

    assert not snapshots_exist("xor_test", root=tmp_path)


def test_run_experiment_watch_falls_back_without_a_terminal(tmp_path, capsys):
    write_experiment_config(tmp_path, "xor_test")

    results = run_experiment("xor_test", root=tmp_path, watch=True)

    assert isinstance(results, Results)
    assert "--watch needs an interactive terminal" in capsys.readouterr().out


def test_run_experiment_watch_drives_the_live_view_in_a_terminal(tmp_path, monkeypatch):
    import orbit.visualization.terminal as terminal_module

    write_experiment_config(tmp_path, "xor_test", epochs=4)
    monkeypatch.setattr("orbit.cli.commands.watch.is_tty", lambda: True)
    seen = []

    class RecordingView(terminal_module.WatchView):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, live=False, **kwargs)
            seen.append(self)

    monkeypatch.setattr(terminal_module, "WatchView", RecordingView)

    results = run_experiment("xor_test", root=tmp_path, watch=True)

    assert len(seen) == 1
    assert seen[0].losses == results.loss_history
