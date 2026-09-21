import json
import pytest
from orbit.core import Results
from orbit.cli.commands.train import train_experiment
from orbit.storage import checkpoint_exists


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


def test_train_experiment_returns_results_and_saves_them(tmp_path):
    write_experiment_config(tmp_path, "xor_test")

    results = train_experiment("xor_test", root=tmp_path)

    assert isinstance(results, Results)
    assert isinstance(results.final_loss, float)
    assert (tmp_path / "xor_test" / "results" / "results.json").exists()


def test_train_experiment_saves_a_checkpoint(tmp_path):
    write_experiment_config(tmp_path, "xor_test")

    train_experiment("xor_test", root=tmp_path)

    assert checkpoint_exists("xor_test", root=tmp_path)


def test_train_experiment_skips_test_evaluation_even_with_test_split(tmp_path):
    write_experiment_config(tmp_path, "xor_test", test_split=0.25, seed=42)

    results = train_experiment("xor_test", root=tmp_path)

    assert results.test_loss is None
    assert results.test_accuracy is None


def test_train_experiment_missing_experiment_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_experiment("does_not_exist", root=tmp_path)
