from rich.table import Table

from orbit.settings import (
    DEFAULT_SETTINGS,
    config_exists,
    config_path,
    get_value,
    load_settings,
    save_settings,
    set_value,
    workspace_path,
)
from orbit.ui import console, success, warning


def show_config() -> None:
    """Print every setting, the workspace in use, and where the file lives."""
    settings = load_settings()

    table = Table(title="ORBIT settings", title_style="bold #ffeab0")
    table.add_column("Setting", style="#ffeab0")
    table.add_column("Value")

    table.add_row("workspace", str(workspace_path(settings).resolve()))
    for section in ("defaults", "display"):
        for key in DEFAULT_SETTINGS[section]:
            table.add_row(f"{section}.{key}", str(get_value(settings, f"{section}.{key}")))

    console.print()
    console.print(table)
    if config_exists():
        console.print(f"Settings file: {config_path()}")
    else:
        warning(
            f"No settings file yet ({config_path()}) - these are the built-in defaults. "
            "Run `orbit init` to create one."
        )
    console.print()


def set_config(key: str, value: str) -> None:
    """
    Change one setting ("section.key", e.g. display.theme) and save. Invalid
    keys/values raise ValueError (reported by the caller) without writing.
    """
    settings = set_value(load_settings(), key, value)
    path = save_settings(settings)
    success(f"{key} = {get_value(settings, key)} (saved to {path})")
