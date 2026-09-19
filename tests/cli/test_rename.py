import json

from orbit.cli.commands.rename import rename_experiment


def write_experiment(root, name, with_results=False):
    exp_dir = root / name
    exp_dir.mkdir(parents=True)

    config = {
        "name": name,
        "dataset": "xor",
        "model": [{"type": "Linear", "in_features": 2, "neurons": 1}],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": 1.0,
        "epochs": 1,
        "seed": 1,
    }
    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)

    if with_results:
        results_dir = exp_dir / "results"
        results_dir.mkdir()
        results = {"name": name, "final_loss": 0.1, "loss_history": [0.5, 0.1], "hyperparams": {}}
        with open(results_dir / "results.json", "w") as f:
            json.dump(results, f)

    return exp_dir


def test_rename_experiment_moves_directory_and_updates_config_name(tmp_path, capsys):
    write_experiment(tmp_path, "old_name")

    rename_experiment("old_name", "new_name", root=tmp_path)

    assert not (tmp_path / "old_name").exists()
    assert (tmp_path / "new_name").exists()
    with open(tmp_path / "new_name" / "experiment.json") as f:
        config = json.load(f)
    assert config["name"] == "new_name"
    assert "Renamed" in capsys.readouterr().out


def test_rename_experiment_updates_results_name_when_present(tmp_path):
    write_experiment(tmp_path, "old_name", with_results=True)

    rename_experiment("old_name", "new_name", root=tmp_path)

    with open(tmp_path / "new_name" / "results" / "results.json") as f:
        results = json.load(f)
    assert results["name"] == "new_name"


def test_rename_experiment_missing_source(tmp_path, capsys):
    rename_experiment("does_not_exist", "new_name", root=tmp_path)

    assert not (tmp_path / "new_name").exists()
    assert "was not found" in capsys.readouterr().out


def test_rename_experiment_refuses_to_overwrite_existing_target(tmp_path, capsys):
    write_experiment(tmp_path, "exp_a")
    write_experiment(tmp_path, "exp_b")

    rename_experiment("exp_a", "exp_b", root=tmp_path)

    assert (tmp_path / "exp_a").exists()
    with open(tmp_path / "exp_b" / "experiment.json") as f:
        config = json.load(f)
    assert config["name"] == "exp_b"
    assert "already exists" in capsys.readouterr().out


def test_rename_experiment_same_name_is_a_no_op(tmp_path, capsys):
    write_experiment(tmp_path, "exp_a")

    rename_experiment("exp_a", "exp_a", root=tmp_path)

    assert (tmp_path / "exp_a").exists()
    assert "nothing to do" in capsys.readouterr().out
