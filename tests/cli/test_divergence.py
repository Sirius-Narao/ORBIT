"""
How a diverged run (loss became inf/NaN, training stopped early) shows up
across the CLI. The fixture is a plain linear model (no squashing output
activation, so MSE is unbounded) trained with an absurd learning rate.
"""
import json

import numpy as np

import orbit.cli.parser as parser_module
from orbit.cli.commands.inspect import inspect_experiment
from orbit.cli.commands.list import list_experiments
from orbit.cli.commands.reproduce import reproduce_experiment
from orbit.cli.commands.run import run_experiment
from orbit.storage import load_results


def write_config(root, name, learning_rate=1e100, epochs=20):
    exp_dir = root / name
    exp_dir.mkdir(parents=True)
    config = {
        "name": name,
        "dataset": "xor",
        "model": [{"type": "Linear", "in_features": 2, "neurons": 1}],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "batch_size": 4,
        "epochs": epochs,
        "seed": 0,
    }
    (exp_dir / "experiment.json").write_text(json.dumps(config))
    return exp_dir


def test_run_experiment_records_where_training_diverged(tmp_path):
    exp_dir = write_config(tmp_path, "boom")

    results = run_experiment("boom", root=tmp_path)

    assert results.diverged_at_epoch is not None
    assert results.diverged_at_epoch <= 20
    assert np.isnan(results.final_loss)
    assert len(results.loss_history) == results.diverged_at_epoch - 1
    assert all(np.isfinite(results.loss_history))
    saved = load_results(exp_dir / "results" / "results.json")
    assert saved.diverged_at_epoch == results.diverged_at_epoch


def test_run_command_reports_divergence_in_one_line(tmp_path, monkeypatch, capsys):
    write_config(tmp_path, "boom")
    monkeypatch.setattr(parser_module, "run_experiment", lambda name, **kwargs: run_experiment(name, root=tmp_path))
    monkeypatch.setattr("sys.argv", ["orbit", "run", "boom"])

    parser_module.main()

    out = capsys.readouterr().out
    assert "Training diverged at epoch" in out
    assert "grad_clip" in out
    assert "Final loss" not in out
    assert "RuntimeWarning" not in out


def test_report_training_handles_a_run_that_diverged_in_its_first_epoch(capsys):
    class DivergedImmediately:
        final_loss = float("nan")
        loss_history = []
        diverged_at_epoch = 1

    parser_module._report_training(DivergedImmediately())

    assert "diverged at epoch 1" in capsys.readouterr().out


def test_warn_if_stalled_ignores_an_empty_history(capsys):
    class Empty:
        final_loss = float("nan")
        loss_history = []

    parser_module._warn_if_stalled(Empty())

    assert capsys.readouterr().out == ""


def test_inspect_and_list_show_the_divergence(tmp_path, capsys):
    write_config(tmp_path, "boom")
    epoch = run_experiment("boom", root=tmp_path).diverged_at_epoch
    capsys.readouterr()

    inspect_experiment("boom", root=tmp_path)
    list_experiments(root=tmp_path)

    out = capsys.readouterr().out
    assert f"Diverged at epoch {epoch}" in out
    assert f"diverged (epoch {epoch})" in out


def test_reproduce_matches_a_run_that_diverges_at_the_same_epoch(tmp_path, capsys):
    write_config(tmp_path, "boom")
    epoch = run_experiment("boom", root=tmp_path).diverged_at_epoch
    capsys.readouterr()

    reproduce_experiment("boom", root=tmp_path)

    assert f"diverged at epoch {epoch} both times" in capsys.readouterr().out


def test_reproduce_reports_a_run_that_now_diverges(tmp_path, capsys):
    exp_dir = write_config(tmp_path, "boom", learning_rate=0.01)
    run_experiment("boom", root=tmp_path)
    config = json.loads((exp_dir / "experiment.json").read_text())
    config["learning_rate"] = 1e100
    (exp_dir / "experiment.json").write_text(json.dumps(config))
    capsys.readouterr()

    reproduce_experiment("boom", root=tmp_path)

    out = capsys.readouterr().out
    assert "Result differs: previously final loss" in out
    assert "now diverged at epoch" in out
