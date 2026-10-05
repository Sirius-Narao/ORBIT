import pathlib
from typing import Optional

import questionary

from orbit.settings import (
    ANIMATION_FORMATS,
    NORMALIZE_CHOICES,
    OPTIMIZERS,
    THEMES,
    config_exists,
    load_settings,
    save_settings,
    set_value,
    workspace_path,
)
from orbit.ui import PROMPT_STYLE, console, success, warning

WORKSPACE_DIRNAME = ".orbits"


def init_project(path: Optional[str] = None, assume_yes: bool = False) -> Optional[pathlib.Path]:
    """
    Set up the ORBIT workspace at <path>/.orbits (path defaults to the current
    directory) and point the user's settings file at it, so every orbit
    command uses it from now on, whatever directory it's run from.

    Asks for the defaults orbit new pre-fills and the display preferences,
    pre-filled with the current settings; assume_yes (--yes) keeps them as
    they are. Idempotent and non-destructive: directories are only ever
    created (mkdir exist_ok), and when the settings already point at a
    different workspace it asks before switching - the old workspace is
    left untouched either way. Returns the workspace path, or None if the
    user declined to switch.
    """
    workspace = ((pathlib.Path(path) if path else pathlib.Path.cwd()) / WORKSPACE_DIRNAME).resolve()
    settings = load_settings()

    if config_exists() and settings["workspace"] and not assume_yes:
        current = workspace_path(settings).resolve()
        if current != workspace:
            switch = questionary.confirm(
                f"Your settings point to the workspace {current}. Switch to {workspace}? "
                "(the old one is kept as is)",
                default=False,
                style=PROMPT_STYLE,
            ).unsafe_ask()
            if not switch:
                warning("Keeping the current workspace.")
                return None

    already_initialized = workspace.exists()
    (workspace / "experiments").mkdir(parents=True, exist_ok=True)
    (workspace / "datasets").mkdir(parents=True, exist_ok=True)

    settings = set_value(settings, "workspace", str(workspace))
    if not assume_yes:
        settings = _ask_settings(settings)
    saved_to = save_settings(settings)

    console.print()
    if already_initialized:
        warning(f"ORBIT workspace already initialized in {workspace}")
    else:
        success(f"Initialized ORBIT workspace in {workspace}")
    success(f"Settings saved to {saved_to}")
    console.print()
    return workspace


def _ask_settings(settings: dict) -> dict:
    """Ask for each default/display setting, pre-filled with its current value."""
    questions = [
        ("display.theme", "Plot theme:", THEMES),
        ("display.animation_format", "Animation format:", ANIMATION_FORMATS),
        ("defaults.optimizer", "Default optimizer:", OPTIMIZERS),
        ("defaults.learning_rate", "Default SGD learning rate:", None),
        ("defaults.adam_learning_rate", "Default Adam learning rate:", None),
        ("defaults.batch_size", "Default batch size:", None),
        ("defaults.epochs", "Default epochs:", None),
        ("defaults.test_split", "Default test split (0 = no split):", None),
        ("defaults.normalize", "Default input normalization:", NORMALIZE_CHOICES),
    ]
    for key, message, choices in questions:
        section, name = key.split(".")
        current = settings[section][name]
        if choices is not None:
            answer = questionary.select(
                message, choices=list(choices), default=current, style=PROMPT_STYLE
            ).unsafe_ask()
            settings = set_value(settings, key, answer)
            continue
        # Re-ask until the answer is valid for this setting.
        while True:
            answer = questionary.text(message, default=str(current), style=PROMPT_STYLE).unsafe_ask()
            try:
                settings = set_value(settings, key, answer)
                break
            except ValueError as e:
                warning(str(e))
    return settings
