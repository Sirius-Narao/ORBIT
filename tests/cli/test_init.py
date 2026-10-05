import questionary

from orbit.cli.commands.init import init_project
from orbit.settings import load_settings, save_settings, set_value, workspace_path


class FakeAnswer:
    def __init__(self, value):
        self.value = value

    def unsafe_ask(self):
        return self.value


def test_init_creates_the_workspace_and_points_settings_at_it(tmp_path):
    workspace = init_project(str(tmp_path / "proj"), assume_yes=True)

    assert workspace == (tmp_path / "proj" / ".orbits").resolve()
    assert (workspace / "experiments").is_dir()
    assert (workspace / "datasets").is_dir()
    assert workspace_path() == workspace


def test_init_reports_success_on_first_run(tmp_path, capsys):
    init_project(str(tmp_path), assume_yes=True)

    assert "Initialized" in capsys.readouterr().out


def test_init_warns_when_already_initialized(tmp_path, capsys):
    init_project(str(tmp_path), assume_yes=True)
    capsys.readouterr()

    init_project(str(tmp_path), assume_yes=True)

    assert "already initialized" in capsys.readouterr().out.lower()


def test_init_does_not_clobber_existing_experiments(tmp_path):
    existing = tmp_path / ".orbits" / "experiments" / "existing_experiment"
    existing.mkdir(parents=True)

    init_project(str(tmp_path), assume_yes=True)

    assert existing.is_dir()


def test_init_with_yes_keeps_existing_settings(tmp_path):
    save_settings(set_value(load_settings(), "display.theme", "dark"))

    init_project(str(tmp_path), assume_yes=True)

    assert load_settings()["display"]["theme"] == "dark"


def test_init_saves_the_answered_settings(tmp_path, monkeypatch):
    selects = iter(["dark", "mp4", "Adam", "standard"])
    texts = {
        "Default SGD learning rate:": iter(["0.5"]),
        "Default Adam learning rate:": iter(["0.01"]),
        "Default batch size:": iter(["zero", "16"]),  # invalid first: re-asked
        "Default epochs:": iter(["500"]),
        "Default test split (0 = no split):": iter(["0"]),
    }
    monkeypatch.setattr(questionary, "select", lambda *a, **k: FakeAnswer(next(selects)))
    monkeypatch.setattr(questionary, "text", lambda message, **k: FakeAnswer(next(texts[message])))

    init_project(str(tmp_path))

    settings = load_settings()
    assert settings["display"] == {"theme": "dark", "animation_format": "mp4", "fps": 10}
    assert settings["defaults"]["optimizer"] == "Adam"
    assert settings["defaults"]["learning_rate"] == 0.5
    assert settings["defaults"]["adam_learning_rate"] == 0.01
    assert settings["defaults"]["batch_size"] == 16
    assert settings["defaults"]["epochs"] == 500
    assert settings["defaults"]["test_split"] == 0.0
    assert settings["defaults"]["normalize"] == "standard"


def test_init_asks_before_switching_workspaces_and_can_keep_the_old_one(tmp_path, monkeypatch):
    first = init_project(str(tmp_path / "a"), assume_yes=True)
    monkeypatch.setattr(questionary, "confirm", lambda *a, **k: FakeAnswer(False))

    assert init_project(str(tmp_path / "b")) is None

    assert workspace_path() == first
    assert not (tmp_path / "b" / ".orbits").exists()
