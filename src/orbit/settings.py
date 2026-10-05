"""
User-level ORBIT settings: where the workspace (.orbits/) lives, the defaults
orbit new pre-fills, and display preferences (plot theme, animation format).

The file is ~/.orbit/config.toml, or whatever $ORBIT_CONFIG points to (tests
point it at a temp path so a developer's real config never leaks into them).
It's written by orbit init and edited by orbit config set, or by hand.

Defaults are applied when an experiment is *created* (they pre-fill orbit new's
prompts and end up written into experiment.json), never when it is run, so a
saved experiment stays self-contained: changing this file never changes what
re-running an old experiment does. Display settings apply whenever something
is rendered.

This module deliberately imports nothing from the rest of orbit, since
storage/, visualization/ and cli/ all read it. That's why the valid optimizer
and normalize values are listed here rather than imported from core/
(tests/test_settings.py checks they stay in sync).
"""
import copy
import os
import pathlib
import tomllib
from typing import Optional

CONFIG_ENV_VAR = "ORBIT_CONFIG"

THEMES = ("dark", "light")
ANIMATION_FORMATS = ("gif", "mp4")
OPTIMIZERS = ("SGD", "Adam")
NORMALIZE_CHOICES = ("none", "standard", "minmax")

DEFAULT_SETTINGS = {
    "workspace": None,
    "defaults": {
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "adam_learning_rate": 0.001,
        "momentum": 0.0,
        "batch_size": 32,
        "epochs": 1000,
        "test_split": 0.2,
        "normalize": "none",
        "grad_clip": 0.0,
    },
    "display": {
        "theme": "light",
        "animation_format": "gif",
        "fps": 10,
    },
}

# (type, check, description) per settable key. "section.key" names match the
# TOML layout and orbit config set's argument.
_SPEC = {
    "defaults.optimizer": (str, lambda v: v in OPTIMIZERS, f"one of {', '.join(OPTIMIZERS)}"),
    "defaults.learning_rate": (float, lambda v: v > 0, "a number > 0"),
    "defaults.adam_learning_rate": (float, lambda v: v > 0, "a number > 0"),
    "defaults.momentum": (float, lambda v: 0 <= v < 1, "a number in [0, 1)"),
    "defaults.batch_size": (int, lambda v: v >= 1, "an integer >= 1"),
    "defaults.epochs": (int, lambda v: v >= 1, "an integer >= 1"),
    "defaults.test_split": (float, lambda v: 0 <= v < 1, "a number in [0, 1) (0 = no split)"),
    "defaults.normalize": (str, lambda v: v in NORMALIZE_CHOICES, f"one of {', '.join(NORMALIZE_CHOICES)}"),
    "defaults.grad_clip": (float, lambda v: v >= 0, "a number >= 0 (0 = off)"),
    "display.theme": (str, lambda v: v in THEMES, f"one of {', '.join(THEMES)}"),
    "display.animation_format": (str, lambda v: v in ANIMATION_FORMATS, f"one of {', '.join(ANIMATION_FORMATS)}"),
    "display.fps": (int, lambda v: v >= 1, "an integer >= 1"),
}

SETTABLE_KEYS = tuple(_SPEC) + ("workspace",)


def config_path() -> pathlib.Path:
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".orbit" / "config.toml"


def config_exists() -> bool:
    return config_path().is_file()


def coerce_value(key: str, value):
    """
    Convert value (a TOML value, or a string from orbit config set) to the
    key's type and validate it. Raises ValueError naming the key otherwise.
    """
    if key == "workspace":
        if value is None or str(value).strip() == "":
            raise ValueError("workspace must be a path")
        return str(value)
    if key not in _SPEC:
        raise ValueError(f"Unknown setting {key!r} (valid: {', '.join(SETTABLE_KEYS)})")

    kind, check, description = _SPEC[key]
    try:
        if kind is int:
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            converted = int(value)
        elif kind is float:
            converted = float(value)
        else:
            converted = str(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be {description}, got {value!r}")

    if isinstance(value, bool) or not check(converted):
        raise ValueError(f"{key} must be {description}, got {value!r}")
    return converted


def load_settings() -> dict:
    """
    DEFAULT_SETTINGS merged with the config file, validated. Read fresh on
    every call (no caching), so the REPL sees an orbit init / orbit config set
    from earlier in the same session. A missing file just gives the defaults.
    """
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    path = config_path()
    if not path.is_file():
        return settings

    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Could not read {path}: {e}")

    if "workspace" in data:
        workspace = pathlib.Path(coerce_value("workspace", data["workspace"]))
        # A relative workspace is relative to the config file, not to
        # wherever orbit happens to be run from.
        if not workspace.is_absolute():
            workspace = path.parent / workspace
        settings["workspace"] = str(workspace)

    for section in ("defaults", "display"):
        for key, value in data.get(section, {}).items():
            settings[section][key] = coerce_value(f"{section}.{key}", value)
    return settings


def set_value(settings: dict, key: str, value) -> dict:
    """Return a copy of settings with "section.key" (or "workspace") set, validated."""
    converted = coerce_value(key, value)
    updated = copy.deepcopy(settings)
    if key == "workspace":
        updated["workspace"] = converted
    else:
        section, name = key.split(".", 1)
        updated[section][name] = converted
    return updated


def get_value(settings: dict, key: str):
    if key == "workspace":
        return settings["workspace"]
    if key not in _SPEC:
        raise ValueError(f"Unknown setting {key!r} (valid: {', '.join(SETTABLE_KEYS)})")
    section, name = key.split(".", 1)
    return settings[section][name]


def _toml_value(value) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


# tomllib can only read, so save_settings renders this template. The schema is
# small and fixed, which keeps that simple and avoids a TOML-writer dependency.
_TEMPLATE = """\
# ORBIT settings - written by `orbit init`, edit with `orbit config set` or by hand.

# Where experiments, datasets, sweeps and comparisons are stored.
workspace = {workspace}

# Pre-filled answers for `orbit new`. They are written into each new
# experiment.json, so changing them never changes existing experiments.
[defaults]
optimizer = {optimizer}
learning_rate = {learning_rate}            # SGD
adam_learning_rate = {adam_learning_rate}
momentum = {momentum}                 # SGD only, 0 = plain SGD
batch_size = {batch_size}
epochs = {epochs}
test_split = {test_split}               # 0 = no split
normalize = {normalize}             # "none", "standard" or "minmax"
grad_clip = {grad_clip}                # 0 = off

# How plots and animations look.
[display]
theme = {theme}                 # "dark" or "light"
animation_format = {animation_format}       # "gif" or "mp4"
fps = {fps}
"""


def save_settings(settings: dict) -> pathlib.Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Forward slashes read better than escaped backslashes and work on Windows too.
    workspace = pathlib.Path(settings["workspace"]).as_posix() if settings["workspace"] else ".orbits"
    values = {"workspace": _toml_value(workspace)}
    for section in ("defaults", "display"):
        for key in DEFAULT_SETTINGS[section]:
            values[key] = _toml_value(settings[section][key])

    with open(path, "w", encoding="utf-8") as f:
        f.write(_TEMPLATE.format(**values))
    return path


def workspace_path(settings: Optional[dict] = None) -> pathlib.Path:
    """
    The workspace directory (the folder holding experiments/, datasets/, ...).
    Without a config, or with one that doesn't name a workspace, this is
    ./.orbits - exactly where ORBIT kept everything before settings existed.
    """
    if settings is None:
        settings = load_settings()
    workspace = settings.get("workspace")
    return pathlib.Path(workspace) if workspace else pathlib.Path(".orbits")
