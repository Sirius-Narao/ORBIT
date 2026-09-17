import json
import pytest
from orbit.core import Results
from orbit.cli.commands.run import run_experiment


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


def test_run_experiment_missing_experiment_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_experiment("does_not_exist", root=tmp_path)
