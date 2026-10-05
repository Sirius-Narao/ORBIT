import pytest

from orbit.cli.commands.config import set_config, show_config
from orbit.settings import load_settings


def test_show_config_without_a_file_points_to_init(capsys):
    show_config()

    out = capsys.readouterr().out
    assert "display.theme" in out
    assert "orbit init" in out


def test_set_config_saves_the_value(capsys):
    set_config("display.theme", "dark")

    assert load_settings()["display"]["theme"] == "dark"
    assert "display.theme = dark" in capsys.readouterr().out


def test_set_config_rejects_invalid_values_without_writing(isolated_orbit_settings):
    with pytest.raises(ValueError):
        set_config("defaults.epochs", "many")

    assert not isolated_orbit_settings.exists()
