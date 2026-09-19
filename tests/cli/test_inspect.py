import json

from orbit.core import Results
from orbit.cli.commands.inspect import inspect_experiment


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


def test_inspect_experiment_not_run_shows_config_only(tmp_path, capsys):
    write_config(tmp_path, "xor_test")

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "xor_test" in out
    assert "MSE" in out
    assert "SGD" in out
    assert "Not run yet." in out


def test_inspect_experiment_shows_results_after_run(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(exp_dir, final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "xor_test" in out
    assert "0.1234" in out
    assert "3 epoch" in out


def test_inspect_experiment_missing_name(tmp_path, capsys):
    inspect_experiment("does_not_exist", root=tmp_path)

    out = capsys.readouterr().out
    assert "was not found" in out
