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

    def fake_run_experiment(name):
        calls.append(name)
        return FakeResults()

    monkeypatch.setattr(parser_module, "run_experiment", fake_run_experiment)
    monkeypatch.setattr("sys.argv", ["orbit", "run", "xor_mlp_01"])

    parser_module.main()

    assert calls == ["xor_mlp_01"]
    assert "0.1234" in capsys.readouterr().out


def test_missing_command_exits_with_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["orbit"])

    with pytest.raises(SystemExit):
        parser_module.main()


def test_run_without_name_exits_with_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["orbit", "run"])

    with pytest.raises(SystemExit):
        parser_module.main()
