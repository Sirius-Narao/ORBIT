import json

from orbit.cli.commands.reproduce import reproduce_experiment
from orbit.cli.commands.run import run_experiment


def write_config(root, name, seed=None, epochs=50):
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
    if seed is not None:
        config["seed"] = seed

    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)

    return exp_dir


def test_reproduce_experiment_without_prior_results(tmp_path, capsys):
    write_config(tmp_path, "xor_test", seed=1)

    reproduce_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "no recorded results" in out
    assert not (tmp_path / "xor_test" / "results" / "results.json").exists()


def test_reproduce_experiment_with_seed_matches(tmp_path, capsys):
    write_config(tmp_path, "xor_test", seed=42)

    # first run establishes the recorded results
    run_experiment("xor_test", root=tmp_path)

    reproduce_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "Reproduced: final loss matches" in out
    assert "differs" not in out


def test_reproduce_experiment_without_seed_warns(tmp_path, capsys):
    write_config(tmp_path, "xor_test", seed=None)

    run_experiment("xor_test", root=tmp_path)

    reproduce_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "no recorded seed" in out
