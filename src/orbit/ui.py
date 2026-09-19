import sys

import questionary
from rich.console import Console
from rich.theme import Theme

# Shared styling for the CLI. Lives at the top level (not under cli/) because
# nn/training/trainer.py needs the same console/tty-detection logic, and nn/
# must never depend on cli/ (cli/ depends on core/nn/storage, never the
# reverse).

# Overrides rich's own built-in style names so widgets that reach for them
# internally - Progress's BarColumn/TaskProgressColumn/TimeElapsedColumn/
# TimeRemainingColumn in trainer.py chief among them - pick up this palette
# automatically instead of rich's defaults (magenta percentage, pink/green
# bar, yellow/cyan timers), with no per-column style plumbing needed.
THEME = Theme({
    "bar.back": "#888888",
    "bar.complete": "#ffeab0",
    "bar.finished": "#b0ffb3",
    "bar.pulse": "#ffeab0",
    "progress.percentage": "#ffeab0",
    "progress.elapsed": "#888888",
    "progress.remaining": "#888888",
})

console = Console(highlight=False, theme=THEME)


def info(message: str) -> None:
    console.print(message, style="#ffeab0", soft_wrap=True)


def success(message: str) -> None:
    console.print(message, style="bold #b0ffb3", soft_wrap=True)


def warning(message: str) -> None:
    console.print(message, style="bold #d6b0ff", soft_wrap=True)


def error(message: str) -> None:
    console.print(message, style="bold #ffb0b0", soft_wrap=True)


def is_tty(stream=None) -> bool:
    """
    Checked fresh on every call (never cached) so pytest's capsys - which
    swaps sys.stdout for a non-tty capture object only for a test's
    duration - is detected correctly regardless of when this module was
    imported.
    """
    stream = stream if stream is not None else sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


PROMPT_STYLE = questionary.Style([
    ("qmark", "fg:#ffeab0 bold"),
    ("question", "bold"),
    ("answer", "fg:#ffeab0 bold"),
    ("pointer", "fg:#ffeab0 bold"),
    ("highlighted", "fg:#ffeab0 bold"),
    ("selected", "fg:#b0ffb3"),
    ("instruction", "fg:#888888"),
])
