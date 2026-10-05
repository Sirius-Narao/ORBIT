import shlex

import questionary
from rich.text import Text

from orbit.cli.parser import build_parser, _dispatch
from orbit.ui import console, error, success, PROMPT_STYLE

BANNER = r"""
                                                           ······
                                            ················    ····••
                                   ··········                      •••
                   ██████╗ ██████╗ ██████╗ ██╗████████╗         ••••
                  ██╔═══██╗██╔══██╗██╔══██╗██║╚══██╔══╝      ••••
               ···██║   ██║██████╔╝██████╔╝██║   ██║    ••••••
          ······  ██║   ██║██╔══██╗██╔══██╗██║   █•••••••
       ····       ╚██████╔╝██║  ██║██████╔╝█•••••••║
    ····           ╚═════╝ ╚═╝  ╚═╝╚•••••••••╝   ╚═╝  v0.2
  ···                      ••••••••••
  ······    ···········●••••
       ······
"""

# Meta-commands the REPL handles itself - not real argparse subcommands, so
# they need to be documented separately from parser.format_help().
_META_COMMANDS = "  help, ?          Show this help\n  exit, quit        Leave the ORBIT REPL"


# The subtitle is letter-spaced in capitals to read larger than plain text,
# split over two lines to stay within the banner's width.
_SUBTITLE_LINES = (("Open", "Research", "&", "Benchmarking"), ("Intelligence", "Toolkit"))
_BANNER_WIDTH = max(len(line) for line in BANNER.splitlines())


def _spaced(word: str) -> str:
    return " ".join(word.upper())


def _subtitle_line(words) -> Text:
    """One letter-spaced subtitle line, the O-R-B-I-T initials highlighted."""
    line = Text()
    for i, word in enumerate(words):
        if i:
            line.append("   ")
        if word[0].isalpha():
            line.append(word[0].upper(), style="bold #ffeab0")
            line.append(_spaced(word)[1:], style="bold #f5f5f5")
        else:
            line.append(word, style="#888888")
    return line


def _centered(line: Text) -> Text:
    """Centers under the banner rather than across the whole terminal."""
    return Text(" " * max(0, (_BANNER_WIDTH - line.cell_len) // 2)) + line


def _print_banner() -> None:
    console.print(BANNER, style="bold #ffeab0", markup=False)
    rule = Text("✦ " + "─" * 62 + " ✦", style="#ffeab0")
    for line in (rule, *map(_subtitle_line, _SUBTITLE_LINES), rule):
        console.print(_centered(line))
    hint = Text.from_markup("Type [bold #ffeab0]help[/] for commands · [bold #ffeab0]exit[/] to quit", style="#888888")
    console.print()
    console.print(_centered(hint))
    console.print()


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
            _dispatch(args, in_repl=True)
        except Exception as e:
            error(str(e))

    console.print()
    console.print("\"What's happened's happened. Which is an expression of faith in the "
    "mechanics of the world. It's not an excuse to do nothing.\"", style="#ffeab0 italic")
    console.print("— Tenet, dir. Christopher Nolan (2020)", style="#ffeab0 italic")
    console.print()

