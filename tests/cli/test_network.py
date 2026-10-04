import json

from orbit.cli.commands.network import network_experiment
from orbit.cli.commands.train import train_experiment

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


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
        "seed": 42,
    }

    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)

    return exp_dir


def test_network_experiment_missing_name(tmp_path, capsys):
    result = network_experiment("does_not_exist", root=tmp_path)

    assert result is None
    assert "was not found" in capsys.readouterr().out


def test_network_experiment_draws_initial_weights_before_a_run(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test")

    result = network_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "has not been run yet" in out
    assert result == exp_dir / "results" / "network.png"
    assert result.read_bytes()[:8] == PNG_MAGIC


def test_network_experiment_draws_the_trained_model(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test")
    train_experiment("xor_test", root=tmp_path)
    capsys.readouterr()

    result = network_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "has not been run yet" not in out
    assert "Saved network diagram" in out
    assert result == exp_dir / "results" / "network.png"


def test_network_experiment_names_the_file_after_the_sample(tmp_path):
    exp_dir = write_config(tmp_path, "xor_test")

    result = network_experiment("xor_test", sample=2, root=tmp_path)

    assert result == exp_dir / "results" / "network_sample2.png"
    assert result.exists()


def test_network_experiment_rejects_an_out_of_range_sample(tmp_path, capsys):
    write_config(tmp_path, "xor_test")

    result = network_experiment("xor_test", sample=4, root=tmp_path)

    assert result is None
    assert "out of range" in capsys.readouterr().out


def test_network_experiment_honors_output_path(tmp_path):
    write_config(tmp_path, "xor_test")
    output = tmp_path / "elsewhere" / "net.png"

    result = network_experiment("xor_test", output=str(output), root=tmp_path)

    assert result == output
    assert output.exists()
