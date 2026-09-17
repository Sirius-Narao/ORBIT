import argparse

from orbit.cli.commands.new import create_experiment
from orbit.cli.commands.run import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("new", help="Create a new experiment interactively")

    run_parser = subparsers.add_parser("run", help="Run a saved experiment")
    run_parser.add_argument("name", help="Name of the experiment to run")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "new":
        create_experiment()
    elif args.command == "run":
        results = run_experiment(args.name)
        print(f"Final loss: {results.final_loss}")
