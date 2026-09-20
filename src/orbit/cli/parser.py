import argparse
import sys

from orbit.cli.commands.new import create_experiment
from orbit.cli.commands.run import run_experiment
from orbit.cli.commands.list import list_experiments
from orbit.cli.commands.delete import delete_experiments
from orbit.cli.commands.inspect import inspect_experiment
from orbit.cli.commands.import_dataset import import_dataset
from orbit.cli.commands.compare import compare_experiments
from orbit.cli.commands.reproduce import reproduce_experiment
from orbit.cli.commands.copy import copy_experiment
from orbit.cli.commands.rename import rename_experiment
from orbit.cli.commands.plot import plot_experiment
from orbit.ui import console, success, warning

# Below this relative improvement between the first and last epoch's loss,
# a run is flagged as stalled rather than just slow - it's a generic signal
# (works for any dataset/model), not a XOR-specific heuristic.
STALLED_RUN_IMPROVEMENT_THRESHOLD = 0.05


def _warn_if_stalled(results) -> None:
    """
    Print a warning if training barely moved the loss. A run that "completes"
    without crashing can still have learned nothing - e.g. a model with too
    little capacity for the dataset, a learning rate that's too small, or
    too few epochs. This can't tell you *which* cause it is, just that the
    result is suspicious and worth a second look.
    """
    first_loss = results.loss_history[0]
    if len(results.loss_history) < 2 or first_loss == 0:
        return

    improvement = (first_loss - results.final_loss) / first_loss
    if improvement < STALLED_RUN_IMPROVEMENT_THRESHOLD:
        warning(
            f"Warning: loss barely improved ({first_loss:.4f} -> "
            f"{results.final_loss:.4f}). The model may be too small for "
            "this dataset, the learning rate may be too low, or it may "
            "need more epochs."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # New command
    new_parser = subparsers.add_parser("new", help="Create a new experiment interactively")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments and their results")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a saved experiment")
    run_parser.add_argument("name", help="Name of the experiment to run")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Show an experiment's config and results")
    inspect_parser.add_argument("name", help="Name of the experiment to inspect")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a saved experiment")
    delete_parser.add_argument("name", nargs="?", help="Name of the experiment to delete (omit with --all)")
    delete_parser.add_argument("--all", action="store_true", help="Delete every experiment")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments by config and results")
    compare_parser.add_argument("names", nargs="*", help="Names of experiments to compare (omit with --all)")
    compare_parser.add_argument("--all", action="store_true", help="Compare every experiment")

    # Reproduce command:
    reproduce_parser = subparsers.add_parser("reproduce", help="Re-run a saved experiment and check the result matches")
    reproduce_parser.add_argument("name", help="Name of the experiment to reproduce")

    # Copy command
    copy_parser = subparsers.add_parser("copy", help="Create a new experiment from an existing one's config")
    copy_parser.add_argument("source", help="Name of the experiment to copy from")

    # Rename command
    rename_parser = subparsers.add_parser("rename", help="Rename a saved experiment")
    rename_parser.add_argument("old_name", help="Current name of the experiment")
    rename_parser.add_argument("new_name", help="New name for the experiment")

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot an experiment's recorded loss curve")
    plot_parser.add_argument("name", help="Name of the experiment to plot")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import a CSV file as a named dataset")
    import_parser.add_argument("csv_path", help="Path to the CSV file to import")
    import_parser.add_argument("--name", help="Name to register the dataset under (default: the file's name)")
    import_parser.add_argument(
        "--target", nargs="+", help="Output/target column(s) (default: prompted interactively)"
    )

    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "new":
        create_experiment()
    elif args.command == "list":
        list_experiments()
    elif args.command == "delete":
        delete_experiments(args.name, is_all=args.all)
    elif args.command == "run":
        results = run_experiment(args.name)
        success(f"Final loss: {results.final_loss}")
        _warn_if_stalled(results)
        console.print()
    elif args.command == "inspect":
        inspect_experiment(args.name)
    elif args.command == "compare":
        compare_experiments(args.names, is_all=args.all)
    elif args.command == "reproduce":
        reproduce_experiment(args.name)
    elif args.command == "copy":
        copy_experiment(args.source)
    elif args.command == "rename":
        rename_experiment(args.old_name, args.new_name)
    elif args.command == "plot":
        plot_experiment(args.name)
    elif args.command == "import":
        import_dataset(args.csv_path, name=args.name, target_columns=args.target)


def main() -> None:
    if len(sys.argv) <= 1:
        from orbit.cli.repl import repl

        repl()
        return

    parser = build_parser()
    args = parser.parse_args()
    _dispatch(args)
