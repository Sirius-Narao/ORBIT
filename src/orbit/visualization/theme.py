"""
Dark and light themes for every plot and animation ORBIT renders.

A theme is two things:
  - matplotlib rcParams (surface, text, axes, grid, legend colors), applied
    with matplotlib.rc_context for the duration of one plot call, so nothing
    leaks into the next call - which matters in the REPL's long-lived process;
  - role colors (accent line, highlight marker, muted text, edges, ...) for
    the colors the plotting code chooses itself, looked up with theme_color.

Which theme: an explicit theme= argument, else the CLI's --theme flag
(set_theme_override, set by cli/parser.py for one command), else the
settings file's display.theme, else "light". Data colormaps (magma, RdBu_r,
viridis, tab10) are the same in both themes - they encode values, and a
value should look the same whatever the background.

Colors follow the surfaces/ink of the dataviz reference palette. The accents
were checked for >= 3:1 contrast against their surface: #2f9e44 on the light
surface, ORBIT's #b0ffb3 on the dark one (the same green the terminal UI uses).
"""
import contextlib
import functools
from typing import Optional

THEMES = {
    "light": {
        "rc": {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "savefig.facecolor": "#fcfcfb",
            "savefig.edgecolor": "#fcfcfb",
            "axes.edgecolor": "#52514e",
            "axes.labelcolor": "#0b0b0b",
            "axes.titlecolor": "#0b0b0b",
            "text.color": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "grid.color": "#e4e3df",
            "legend.facecolor": "#fcfcfb",
            "legend.edgecolor": "#d6d5d0",
            "legend.labelcolor": "#0b0b0b",
        },
        "roles": {
            "surface": "#fcfcfb",
            "accent": "#2f9e44",
            "highlight": "#7a4fd1",
            "muted": "#6b6a65",
            "edge": "#0b0b0b",
            "unhealthy": "#d03b3b",
            "zone": "#d6b0ff",
        },
    },
    "dark": {
        "rc": {
            "figure.facecolor": "#1a1a19",
            "axes.facecolor": "#1a1a19",
            "savefig.facecolor": "#1a1a19",
            "savefig.edgecolor": "#1a1a19",
            "axes.edgecolor": "#8a8984",
            "axes.labelcolor": "#ffffff",
            "axes.titlecolor": "#ffffff",
            "text.color": "#ffffff",
            "xtick.color": "#c3c2b7",
            "ytick.color": "#c3c2b7",
            "grid.color": "#383835",
            "legend.facecolor": "#242423",
            "legend.edgecolor": "#4a4a46",
            "legend.labelcolor": "#ffffff",
        },
        "roles": {
            "surface": "#1a1a19",
            "accent": "#b0ffb3",
            "highlight": "#d6b0ff",
            "muted": "#a3a29b",
            "edge": "#ffffff",
            "unhealthy": "#ffb0b0",
            "zone": "#8f6fd0",
        },
    },
}

DEFAULT_THEME = "light"

_override: Optional[str] = None
_active: list = []  # theme names of the theme_context()s currently entered


def set_theme_override(name: Optional[str]) -> None:
    """Set (or with None, clear) the theme a CLI --theme flag asked for."""
    global _override
    if name is not None and name not in THEMES:
        raise ValueError(f"Unknown theme {name!r} (valid: {', '.join(THEMES)})")
    _override = name


def resolve_theme(name: Optional[str] = None) -> str:
    if name is None:
        name = _override
    if name is None:
        from orbit.settings import load_settings

        name = load_settings()["display"]["theme"]
    if name not in THEMES:
        raise ValueError(f"Unknown theme {name!r} (valid: {', '.join(THEMES)})")
    return name


@contextlib.contextmanager
def theme_context(name: Optional[str] = None):
    """Apply a theme's rcParams and make it the one theme_color() answers for."""
    import matplotlib

    name = resolve_theme(name)
    _active.append(name)
    try:
        with matplotlib.rc_context(THEMES[name]["rc"]):
            yield name
    finally:
        _active.pop()


def theme_color(role: str) -> str:
    """A role color ("accent", "muted", ...) of the theme currently in use."""
    name = _active[-1] if _active else resolve_theme()
    return THEMES[name]["roles"][role]


def themed(function):
    """
    Run a plot function inside theme_context(). The wrapped function also
    accepts theme="dark"/"light" to pick one explicitly (used by tests).
    """
    @functools.wraps(function)
    def wrapper(*args, theme: Optional[str] = None, **kwargs):
        with theme_context(theme):
            return function(*args, **kwargs)
    return wrapper
