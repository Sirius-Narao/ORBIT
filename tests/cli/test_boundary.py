import json

from orbit.cli.commands.boundary import boundary_experiment
from orbit.cli.commands.run import run_experiment


def write_config(root, name, epochs=4, test_split=None):
    exp_dir = root / name
    exp_dir.mkdir(parents=True)
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
        "learning_rate": 2.0,
        "batch_size": 4,
        "epochs": epochs,
        "seed": 3,
    }
    if test_split is not None:
        config["test_split"] = test_split
    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)
    return exp_dir


def test_boundary_experiment_missing_name(tmp_path, capsys):
    assert boundary_experiment("nope", root=tmp_path) is None
    assert "was not found" in capsys.readouterr().out


def test_boundary_experiment_saves_a_png(tmp_path):
    exp_dir = write_config(tmp_path, "xor_test", test_split=0.25)
    run_experiment("xor_test", root=tmp_path)

    result = boundary_experiment("xor_test", root=tmp_path)

    assert result == exp_dir / "results" / "boundary.png"
    assert result.exists()


def test_boundary_experiment_animates_every_snapshot(tmp_path):
    from PIL import Image

    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    run_experiment("xor_test", root=tmp_path)

    result = boundary_experiment("xor_test", animate=True, root=tmp_path)

    assert result == exp_dir / "results" / "boundary.gif"
    with Image.open(result) as gif:
        assert gif.n_frames == 4


def test_boundary_experiment_animate_needs_a_history(tmp_path, capsys):
    write_config(tmp_path, "xor_test")

    assert boundary_experiment("xor_test", animate=True, root=tmp_path) is None
    assert "no recorded training history" in capsys.readouterr().out


def test_boundary_experiment_rejects_a_model_without_two_inputs(tmp_path, capsys, monkeypatch):
    import numpy as np

    write_config(tmp_path, "xor_test")
    monkeypatch.setattr(
        "orbit.cli.commands.boundary.training_arrays", lambda experiment: (np.zeros((4, 3)), np.zeros((4, 1)))
    )

    assert boundary_experiment("xor_test", root=tmp_path) is None
    assert "exactly 2 input features" in capsys.readouterr().out
