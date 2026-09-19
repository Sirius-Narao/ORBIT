import shlex

import questionary

from orbit.cli.parser import build_parser, _dispatch
from orbit.ui import console, error, success, PROMPT_STYLE

BANNER = r"""
  ██████╗ ██████╗ ██████╗ ██╗████████╗
  ██╔═══██╗██╔══██╗██╔══██╗██║╚══██╔══╝
  ██║   ██║██████╔╝██████╔╝██║   ██║
  ██║   ██║██╔══██╗██╔══██╗██║   ██║
  ╚██████╔╝██║  ██║██████╔╝██║   ██║
   ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝   ╚═╝  v0.1
"""

# Meta-commands the REPL handles itself - not real argparse subcommands, so
# they need to be documented separately from parser.format_help().
_META_COMMANDS = "  help, ?          Show this help\n  exit, quit        Leave the ORBIT REPL"


def _print_banner() -> None:
    console.print(BANNER, style="bold #ffeab0", markup=False)
    console.print("  ═════════════════════════════════════════════════", style="#ffeab0")
    console.print("  Open Research & Benchmarking Intelligence Toolkit", style="bold")
    console.print("  ═════════════════════════════════════════════════", style="#ffeab0")
    console.print("  Type [bold #ffeab0]help[/] for commands · [bold #ffeab0]exit[/] to quit\n", style="#888888")


def _print_help(parser) -> None:
    console.print(parser.format_help(), style="#888888", markup=False)
    console.print(_META_COMMANDS, style="#888888")


def _read_line() -> str:
    """
    Uses .unsafe_ask() (unlike the .ask() used by every other prompt in the
    codebase) so Ctrl-C and Ctrl-D surface as distinct exceptions instead of
    both collapsing into a None return - the REPL loop needs to tell "clear
    the line" apart from "exit the session".
    """
    return questionary.text("orbit>", style=PROMPT_STYLE).unsafe_ask()


def repl() -> None:
    parser = build_parser()
    _print_banner()

    while True:
        try:
            line = _read_line()
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            break

        line = line.strip()
        if not line:
            continue
        if line in ("exit", "quit"):
            break
        if line in ("help", "?"):
            _print_help(parser)
            continue

        try:
            tokens = shlex.split(line)
        except ValueError as e:
            error(f"Could not parse input: {e}")
            continue

        try:
            args = parser.parse_args(tokens)
        except SystemExit:
            # argparse already printed its own usage/error/help - don't let
            # one bad line kill the whole session.
            continue

        try:
            _dispatch(args)
        except Exception as e:
            error(str(e))

    console.print()
    console.print("\"What's happened's happened. Which is an expression of faith in the "
    "mechanics of the world. It's not an excuse to do nothing.\"", style="#ffeab0 italic")
    console.print("— Tenet, dir. Christopher Nolan (2020)", style="#ffeab0 italic")
    console.print()

