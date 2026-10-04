import json

import numpy as np

from orbit.cli.commands import network as network_module
from orbit.cli.commands.health import health_experiment


def write_config(root, name, model):
    exp_dir = root / name
    exp_dir.mkdir(parents=True)
    config = {
        "name": name, "dataset": "xor", "model": model, "loss": "MSE", "optimizer": "SGD",
        "learning_rate": 1.0, "batch_size": 4, "epochs": 3, "seed": 0,
    }
    (exp_dir / "experiment.json").write_text(json.dumps(config))
    return exp_dir


HIDDEN_RELU = [
    {"type": "Linear", "in_features": 2, "neurons": 6},
    {"type": "ReLU"},
    {"type": "Linear", "neurons": 1},
]


def test_health_experiment_missing_name(tmp_path, capsys):
    assert health_experiment("nope", root=tmp_path) is None
    assert "was not found" in capsys.readouterr().out


def test_health_experiment_prints_a_table_and_saves_histograms(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "relu_net", HIDDEN_RELU)

    result = health_experiment("relu_net", root=tmp_path)

    out = capsys.readouterr().out
    assert "Linear 6 · ReLU" in out
    assert "Dead (ReLU)" in out
    assert result == exp_dir / "results" / "health.png"
    assert result.exists()


def test_health_experiment_warns_about_a_dead_layer(tmp_path, capsys, monkeypatch):
    write_config(tmp_path, "relu_net", HIDDEN_RELU)
    real_loader = network_module.load_trained_experiment

    def loader_with_dead_units(name, root):
        # Every hidden bias far negative: all six ReLUs output 0 on XOR's inputs.
        config, experiment, trained = real_loader(name, root)
        experiment.model._modules["0"].bias.data = np.full(6, -100.0)
        return config, experiment, trained

    monkeypatch.setattr("orbit.cli.commands.health.load_trained_experiment", loader_with_dead_units)

    health_experiment("relu_net", root=tmp_path)

    assert "100% of its ReLU units are dead" in capsys.readouterr().out


def test_health_experiment_needs_a_hidden_layer(tmp_path, capsys):
    write_config(tmp_path, "linear", [{"type": "Linear", "in_features": 2, "neurons": 1}])

    assert health_experiment("linear", root=tmp_path) is None
    assert "no hidden layers" in capsys.readouterr().out
