import json

from orbit.cli.commands.animate import animate_experiment
from orbit.cli.commands.run import run_experiment


def write_config(root, name, epochs=4, model=None):
    exp_dir = root / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "name": name,
        "dataset": "xor",
        "model": model or [
            {"type": "Linear", "in_features": 2, "neurons": 4},
            {"type": "Tanh"},
            {"type": "Linear", "neurons": 1},
            {"type": "Sigmoid"},
        ],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": 2.0,
        "batch_size": 4,
        "epochs": epochs,
        "seed": 1,
    }
    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)
    return exp_dir


def test_animate_experiment_missing_name(tmp_path, capsys):
    assert animate_experiment("nope", root=tmp_path) is None
    assert "was not found" in capsys.readouterr().out


def test_animate_experiment_warns_without_training_history(tmp_path, capsys):
    write_config(tmp_path, "xor_test")

    assert animate_experiment("xor_test", root=tmp_path) is None
    assert "no recorded training history" in capsys.readouterr().out


def test_animate_experiment_writes_one_frame_per_snapshot(tmp_path, capsys):
    from PIL import Image

    exp_dir = write_config(tmp_path, "xor_test", epochs=4)
    run_experiment("xor_test", root=tmp_path)

    result = animate_experiment("xor_test", sample=1, root=tmp_path)

    assert result == exp_dir / "results" / "training_sample1.gif"
    with Image.open(result) as gif:
        assert gif.n_frames == 5  # epochs 0-4
    assert "Saved training animation" in capsys.readouterr().out


def test_animate_experiment_warns_when_history_no_longer_matches_the_model(tmp_path, capsys):
    write_config(tmp_path, "xor_test")
    run_experiment("xor_test", root=tmp_path)
    # Architecture edited after the run: an extra hidden layer. The checkpoint
    # no longer fits either, so it's refused before the history is read -
    # either way the command must not crash with a raw KeyError.
    write_config(tmp_path, "xor_test", model=[
        {"type": "Linear", "in_features": 2, "neurons": 4},
        {"type": "Tanh"},
        {"type": "Linear", "neurons": 4},
        {"type": "Tanh"},
        {"type": "Linear", "neurons": 1},
    ])

    result = animate_experiment("xor_test", root=tmp_path)

    assert result is None
    assert "re-run it" in capsys.readouterr().out


def test_animate_experiment_reports_missing_ffmpeg(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("orbit.visualization.animation.ffmpeg_path", lambda: None)
    write_config(tmp_path, "xor_test")
    run_experiment("xor_test", root=tmp_path)
    capsys.readouterr()

    assert animate_experiment("xor_test", video_format="mp4", root=tmp_path) is None
    assert "pip install -e .[video]" in capsys.readouterr().out
