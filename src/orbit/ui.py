import shutil
import sys

import questionary
from rich.console import Console
from rich.theme import Theme

# Shared styling for the CLI. Lives at the top level (not under cli/) because
# nn/training/trainer.py needs the same console/tty-detection logic, and nn/
# must never depend on cli/ (cli/ depends on core/nn/storage, never the
# reverse).

# When stdout/stderr aren't attached to a real Windows console - piped
# through Git Bash/mintty, redirected to a file, wrapped by another process -
# Windows' PEP 528 UTF-8 console handling doesn't kick in, and Python falls
# back to the system locale's codepage (commonly cp1252 on a US/UK Windows
# install). That codepage can't encode this module's own output (the REPL
# banner's box-drawing characters, em dashes in prompts/messages, etc.), so
# a plain console.print() crashes the whole CLI before anything is shown.
# Forcing UTF-8 here removes the dependency on a real console being
# attached; wrapped in try/except since not every stdout stand-in supports
# .reconfigure() (e.g. pytest's capsys substitute).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

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

# legacy_windows=False forces rich to emit plain ANSI/VT escape sequences
# instead of shelling out to the legacy Win32 console API. That API path
# encodes to the console's active codepage (often cp1252) with strict error
# handling, so any non-cp1252 character - the box-drawing banner in
# cli/repl.py, its em dashes, etc. - raises a bare UnicodeEncodeError and
# crashes the whole REPL before a single prompt is shown. Forcing the ANSI
# path instead relies on Python's own stdout text handling, which (PEP 528)
# talks to a real Windows console in UTF-8/UTF-16 regardless of codepage,
# and works unmodified under non-native terminals (Windows Terminal, Git
# Bash/mintty, WSL) that already speak ANSI natively.
# shutil.get_terminal_size() reports the real terminal's width when one is
# attached (so a genuinely narrow terminal is still respected), and falls
# back to `fallback=` only when it isn't - e.g. output piped/redirected, or
# pytest's capsys. rich.Console's own built-in fallback for that same case
# is a conservative 80 columns, which orbit list's growing set of metric
# columns (loss, epochs, duration, gradient norm, accuracy, ...) can now
# exceed, forcing rich to truncate the Name column down to a few characters.
# Passing width= explicitly here uses this wider fallback instead, at the
# cost of no longer re-detecting a live terminal resize mid-session (an
# acceptable trade for a CLI tool, and consistent with most CLIs' behavior).
_terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
console = Console(highlight=False, theme=THEME, legacy_windows=False, width=_terminal_width)


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
