import pathlib
import tomllib

import pytest

from orbit import settings as settings_module
from orbit.settings import (
    DEFAULT_SETTINGS,
    config_path,
    load_settings,
    save_settings,
    set_value,
    workspace_path,
)


def test_config_path_follows_the_env_var(isolated_orbit_settings):
    assert config_path() == isolated_orbit_settings


def test_missing_file_gives_the_defaults():
    assert load_settings() == DEFAULT_SETTINGS


def test_workspace_defaults_to_dot_orbits_without_a_config():
    # Exactly where ORBIT kept everything before settings existed.
    assert workspace_path() == pathlib.Path(".orbits")


def test_save_then_load_round_trips(tmp_path):
    settings = set_value(load_settings(), "workspace", str(tmp_path / "ws"))
    settings = set_value(settings, "display.theme", "dark")
    settings = set_value(settings, "defaults.batch_size", "8")
    settings = set_value(settings, "defaults.learning_rate", "0.05")

    save_settings(settings)

    assert load_settings() == settings
    assert workspace_path() == tmp_path / "ws"


def test_saved_file_is_valid_toml_with_comments():
    path = save_settings(load_settings())

    text = path.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert tomllib.loads(text)["display"]["theme"] == "light"


def test_relative_workspace_is_relative_to_the_config_file(isolated_orbit_settings):
    isolated_orbit_settings.parent.mkdir(parents=True)
    isolated_orbit_settings.write_text('workspace = "ws"\n', encoding="utf-8")

    assert workspace_path() == isolated_orbit_settings.parent / "ws"


def test_partial_file_keeps_defaults_for_missing_keys(isolated_orbit_settings):
    isolated_orbit_settings.parent.mkdir(parents=True)
    isolated_orbit_settings.write_text('[display]\ntheme = "dark"\n', encoding="utf-8")

    settings = load_settings()

    assert settings["display"]["theme"] == "dark"
    assert settings["defaults"] == DEFAULT_SETTINGS["defaults"]


@pytest.mark.parametrize("key, value", [
    ("display.theme", "neon"),
    ("defaults.batch_size", "0"),
    ("defaults.batch_size", "2.5"),
    ("defaults.learning_rate", "-1"),
    ("defaults.test_split", "1"),
    ("defaults.optimizer", "RMSprop"),
    ("defaults.normalize", "zscore"),
])
def test_invalid_values_are_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        set_value(load_settings(), key, value)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown setting"):
        set_value(load_settings(), "display.nope", "x")


def test_invalid_value_in_the_file_is_rejected(isolated_orbit_settings):
    isolated_orbit_settings.parent.mkdir(parents=True)
    isolated_orbit_settings.write_text("[defaults]\nepochs = -3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="epochs"):
        load_settings()


def test_valid_choices_match_the_registries():
    # settings.py can't import core/ (storage/ imports it), so it keeps its
    # own copies of these lists.
    from orbit.core.config import OPTIMIZER_REGISTRY
    from orbit.core.dataset import NORMALIZE_METHODS

    assert set(settings_module.OPTIMIZERS) == set(OPTIMIZER_REGISTRY)
    assert set(settings_module.NORMALIZE_CHOICES) == {"none", *NORMALIZE_METHODS}
