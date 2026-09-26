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


def write_results(
    exp_dir, final_loss=0.1234, loss_history=None, duration_seconds=None, gradient_norm_history=None,
    accuracy_history=None, test_loss=None, test_accuracy=None,
):
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True)

    results = Results(
        name=exp_dir.name,
        final_loss=final_loss,
        loss_history=loss_history if loss_history is not None else [0.5, 0.3, final_loss],
        duration_seconds=duration_seconds,
        gradient_norm_history=gradient_norm_history,
        accuracy_history=accuracy_history,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
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


def test_inspect_experiment_shows_duration_when_recorded(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(exp_dir, final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234], duration_seconds=12.5)

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "12.50s" in out


def test_inspect_experiment_omits_duration_when_not_recorded(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(exp_dir, final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "3 epoch" in out
    assert "epoch(s) in" not in out


def test_inspect_experiment_shows_gradient_norm_when_recorded(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(
        exp_dir,
        final_loss=0.1234,
        loss_history=[0.5, 0.3, 0.1234],
        gradient_norm_history=[2.0, 0.9, 0.3456],
    )

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "final gradient norm 0.3456" in out


def test_inspect_experiment_omits_gradient_norm_when_not_recorded(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(exp_dir, final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "3 epoch" in out
    assert "gradient norm" not in out


def test_inspect_experiment_shows_accuracy_when_recorded(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(
        exp_dir,
        final_loss=0.1234,
        loss_history=[0.5, 0.3, 0.1234],
        accuracy_history=[0.5, 0.75, 0.9],
    )

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "final accuracy 90.00%" in out


def test_inspect_experiment_omits_accuracy_when_not_recorded(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(exp_dir, final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "3 epoch" in out
    assert "accuracy" not in out


def test_inspect_experiment_shows_test_loss_and_accuracy_when_recorded(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(
        exp_dir,
        final_loss=0.1234,
        loss_history=[0.5, 0.3, 0.1234],
        test_loss=0.2345,
        test_accuracy=0.875,
    )

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "test loss 0.2345" in out
    assert "test accuracy 87.50%" in out


def test_inspect_experiment_omits_test_loss_when_not_recorded(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "xor_test", epochs=3)
    write_results(exp_dir, final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])

    inspect_experiment("xor_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "test loss" not in out
    assert "test accuracy" not in out


def test_inspect_experiment_missing_name(tmp_path, capsys):
    inspect_experiment("does_not_exist", root=tmp_path)

    out = capsys.readouterr().out
    assert "was not found" in out


def test_inspect_experiment_shows_r2_as_a_plain_number(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "r2_test", epochs=3)
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True)
    results = Results(
        name="r2_test",
        final_loss=0.1,
        loss_history=[0.5, 0.3, 0.1],
        hyperparams={"task": "regression_r2"},
        accuracy_history=[0.2, 0.6, 0.8123],
        test_loss=0.2,
        test_accuracy=-0.25,
    )
    with open(results_dir / "results.json", "w") as f:
        json.dump(results.to_dict(), f)

    inspect_experiment("r2_test", root=tmp_path)

    out = capsys.readouterr().out
    assert "final R² 0.8123" in out
    assert "test R² -0.2500" in out
    assert "%" not in out.split("Results:")[1]
