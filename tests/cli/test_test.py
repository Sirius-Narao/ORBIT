import json
import pytest

from orbit.cli.commands.train import train_experiment
from orbit.cli.commands.run import run_experiment
from orbit.cli.commands.test import test_experiment as evaluate_experiment
from orbit.storage import load_results


def write_experiment_config(root, name, epochs=5, test_split=None, seed=None):
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
    if test_split is not None:
        config["test_split"] = test_split
    if seed is not None:
        config["seed"] = seed

    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)

    return exp_dir


def test_test_experiment_missing_experiment_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate_experiment("does_not_exist", root=tmp_path)


def test_test_experiment_warns_when_not_trained_yet(tmp_path, capsys):
    write_experiment_config(tmp_path, "xor_test", test_split=0.25, seed=42)

    result = evaluate_experiment("xor_test", root=tmp_path)

    assert result is None
    assert "has not been trained yet" in capsys.readouterr().out


def test_test_experiment_warns_when_no_test_split_configured(tmp_path, capsys):
    write_experiment_config(tmp_path, "xor_test", seed=42)
    train_experiment("xor_test", root=tmp_path)

    result = evaluate_experiment("xor_test", root=tmp_path)

    assert result is None
    assert "no test split configured" in capsys.readouterr().out


def test_test_experiment_warns_without_seed_but_still_proceeds(tmp_path, capsys):
    write_experiment_config(tmp_path, "xor_test", test_split=0.25)
    train_experiment("xor_test", root=tmp_path)

    result = evaluate_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "no recorded seed" in out
    assert result is not None
    assert result.test_loss is not None


def test_test_experiment_evaluates_and_persists_results_after_train(tmp_path):
    write_experiment_config(tmp_path, "xor_test", test_split=0.25, seed=42)
    train_experiment("xor_test", root=tmp_path)

    result = evaluate_experiment("xor_test", root=tmp_path)

    assert result.test_loss is not None
    assert result.test_accuracy is None  # no "task" configured

    reloaded = load_results(tmp_path / "xor_test" / "results" / "results.json")
    assert reloaded.test_loss == result.test_loss


def test_test_experiment_works_after_orbit_run_too(tmp_path):
    write_experiment_config(tmp_path, "xor_test", test_split=0.25, seed=42)
    run_experiment("xor_test", root=tmp_path)

    result = evaluate_experiment("xor_test", root=tmp_path)

    assert result.test_loss is not None
