import json

from orbit.core import Results
from orbit.cli.commands.compare import compare_experiments


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


def test_compare_experiments_named(tmp_path, capsys):
    exp_a = write_config(tmp_path, "exp_a")
    write_results(exp_a, final_loss=0.1234)
    write_config(tmp_path, "exp_b")

    compare_experiments(["exp_a", "exp_b"], root=tmp_path)

    out = capsys.readouterr().out
    assert "exp_a" in out
    assert "exp_b" in out
    assert "0.1234" in out
    assert "not run" in out


def test_compare_experiments_skips_missing_name(tmp_path, capsys):
    write_config(tmp_path, "exp_a")

    compare_experiments(["exp_a", "does_not_exist"], root=tmp_path)

    out = capsys.readouterr().out
    assert "exp_a" in out
    assert "does_not_exist was not found, skipping." in out


def test_compare_experiments_all(tmp_path, capsys):
    write_config(tmp_path, "exp_a")
    write_config(tmp_path, "exp_b")

    compare_experiments(is_all=True, root=tmp_path)

    out = capsys.readouterr().out
    assert "exp_a" in out
    assert "exp_b" in out


def test_compare_experiments_all_empty_dir(tmp_path, capsys):
    compare_experiments(is_all=True, root=tmp_path)

    out = capsys.readouterr().out
    assert "No experiments found" in out


def test_compare_experiments_all_missing_root(tmp_path, capsys):
    compare_experiments(is_all=True, root=tmp_path / "does_not_exist")

    out = capsys.readouterr().out
    assert "No experiments found" in out


def test_compare_experiments_no_names_and_not_all(tmp_path, capsys):
    compare_experiments([], root=tmp_path)

    out = capsys.readouterr().out
    assert "No experiments found" in out
