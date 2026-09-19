import sys

import questionary
from rich.console import Console

# Shared styling for the CLI. Lives at the top level (not under cli/) because
# nn/training/trainer.py needs the same console/tty-detection logic, and nn/
# must never depend on cli/ (cli/ depends on core/nn/storage, never the
# reverse).

console = Console(highlight=False)


def info(message: str) -> None:
    console.print(message, style="cyan", soft_wrap=True)


def success(message: str) -> None:
    console.print(message, style="bold green", soft_wrap=True)


def warning(message: str) -> None:
    console.print(message, style="bold yellow", soft_wrap=True)


def error(message: str) -> None:
    console.print(message, style="bold red", soft_wrap=True)


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
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:#ffeab0 bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:green"),
    ("instruction", "fg:#888888"),
])
