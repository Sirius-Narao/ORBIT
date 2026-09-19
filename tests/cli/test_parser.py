import pytest
import orbit.cli.parser as parser_module


def test_new_dispatches_to_create_experiment(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "create_experiment", lambda: calls.append("called"))
    monkeypatch.setattr("sys.argv", ["orbit", "new"])

    parser_module.main()

    assert calls == ["called"]


def test_run_dispatches_to_run_experiment_with_name(monkeypatch, capsys):
    calls = []

    class FakeResults:
        final_loss = 0.1234
        loss_history = [1.0, 0.5, 0.1234]  # a healthy, clearly-converging run

    def fake_run_experiment(name):
        calls.append(name)
        return FakeResults()

    monkeypatch.setattr(parser_module, "run_experiment", fake_run_experiment)
    monkeypatch.setattr("sys.argv", ["orbit", "run", "xor_mlp_01"])

    parser_module.main()

    assert calls == ["xor_mlp_01"]
    assert "0.1234" in capsys.readouterr().out


def test_run_warns_when_loss_barely_improves(monkeypatch, capsys):
    class FakeResults:
        final_loss = 0.2389
        loss_history = [0.2448] * 10 + [0.2389]  # ~2% improvement - a stall

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "underpowered_model"])

    parser_module.main()

    assert "Warning" in capsys.readouterr().out


def test_run_does_not_warn_on_a_healthy_run(monkeypatch, capsys):
    class FakeResults:
        final_loss = 0.05
        loss_history = [1.0, 0.5, 0.2, 0.05]  # well over the 5% threshold

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "healthy_model"])

    parser_module.main()

    assert "Warning" not in capsys.readouterr().out


def test_run_does_not_warn_on_single_epoch_run(monkeypatch, capsys):
    # Not enough history to judge "improvement" at all - must not crash on
    # a one-element loss_history (would divide fine, but there's no trend).
    class FakeResults:
        final_loss = 0.5
        loss_history = [0.5]

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "one_epoch_model"])

    parser_module.main()

    assert "Warning" not in capsys.readouterr().out


def test_import_dispatches_to_import_dataset_with_args(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "import_dataset",
        lambda csv_path, name=None, target_columns=None: calls.append(
            (csv_path, name, target_columns)
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["orbit", "import", "data.csv", "--name", "housing", "--target", "price"],
    )

    parser_module.main()

    assert calls == [("data.csv", "housing", ["price"])]


def test_import_dispatches_with_defaults_when_optional_flags_omitted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "import_dataset",
        lambda csv_path, name=None, target_columns=None: calls.append(
            (csv_path, name, target_columns)
        ),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "import", "data.csv"])

    parser_module.main()

    assert calls == [("data.csv", None, None)]


def test_missing_command_exits_with_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["orbit"])

    with pytest.raises(SystemExit):
        parser_module.main()


def test_run_without_name_exits_with_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["orbit", "run"])

    with pytest.raises(SystemExit):
        parser_module.main()
