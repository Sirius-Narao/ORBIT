import questionary

import orbit.cli.parser as parser_module
import orbit.cli.repl as repl_module


def fake_answers(monkeypatch, answers):
    """
    questionary.text(...) returns an object with an unsafe_ask() method.
    Stub it to hand back canned lines - or raise a canned exception - in
    call order, so repl() runs the same as if a person had typed each line
    and pressed enter (or Ctrl-C/Ctrl-D).
    """
    queue = iter(answers)

    class FakeAnswer:
        def __init__(self, value):
            self.value = value

        def unsafe_ask(self):
            if isinstance(self.value, BaseException):
                raise self.value
            return self.value

    monkeypatch.setattr(questionary, "text", lambda *a, **k: FakeAnswer(next(queue)))


def test_repl_prints_banner_and_dispatches_known_command(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(parser_module, "list_experiments", lambda: calls.append("called"))
    fake_answers(monkeypatch, ["list", "exit"])

    repl_module.repl()

    assert calls == ["called"]
    assert "Toolkit" in capsys.readouterr().out


def test_repl_quit_also_ends_the_loop(monkeypatch, capsys):
    fake_answers(monkeypatch, ["quit"])

    repl_module.repl()

    assert "happened" in capsys.readouterr().out


def test_repl_unknown_command_does_not_crash_the_loop(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "list_experiments", lambda: calls.append("called"))
    fake_answers(monkeypatch, ["banana", "list", "exit"])

    repl_module.repl()

    assert calls == ["called"]


def test_repl_blank_input_is_a_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "list_experiments", lambda: calls.append("called"))
    fake_answers(monkeypatch, ["", "   ", "list", "exit"])

    repl_module.repl()

    assert calls == ["called"]


def test_repl_help_prints_usage_without_dispatching(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(parser_module, "list_experiments", lambda: calls.append("called"))
    fake_answers(monkeypatch, ["help", "exit"])

    repl_module.repl()

    assert calls == []
    assert "usage" in capsys.readouterr().out.lower()


def test_repl_catches_domain_exception_and_keeps_going(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "import_dataset",
        lambda csv_path, name=None, target_columns=None: (_ for _ in ()).throw(
            FileNotFoundError("missing.csv not found")
        ),
    )
    monkeypatch.setattr(parser_module, "list_experiments", lambda: calls.append("called"))
    fake_answers(monkeypatch, ["import missing.csv", "list", "exit"])

    repl_module.repl()

    assert calls == ["called"]
    assert "missing.csv not found" in capsys.readouterr().out


def test_repl_keyboard_interrupt_reprompts_instead_of_exiting(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "list_experiments", lambda: calls.append("called"))
    fake_answers(monkeypatch, [KeyboardInterrupt(), "list", "exit"])

    repl_module.repl()

    assert calls == ["called"]


def test_repl_eof_exits_cleanly(monkeypatch, capsys):
    fake_answers(monkeypatch, [EOFError()])

    repl_module.repl()

    assert "happened" in capsys.readouterr().out
