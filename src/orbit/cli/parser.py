import argparse
import sys

from orbit.cli.commands.init import init_project
from orbit.cli.commands.new import create_experiment
from orbit.cli.commands.run import run_experiment
from orbit.cli.commands.train import train_experiment
from orbit.cli.commands.test import test_experiment
from orbit.cli.commands.list import list_experiments
from orbit.cli.commands.delete import delete_experiments
from orbit.cli.commands.inspect import inspect_experiment
from orbit.cli.commands.import_dataset import import_dataset
from orbit.cli.commands.compare import compare_experiments
from orbit.cli.commands.reproduce import reproduce_experiment
from orbit.cli.commands.copy import copy_experiment
from orbit.cli.commands.rename import rename_experiment
from orbit.cli.commands.plotloss import plot_experiment
from orbit.cli.commands.plot import plot_experiments
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

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new ORBIT project (.orbits/ directory structure)")

    # New command
    new_parser = subparsers.add_parser("new", help="Create a new experiment interactively")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments and their results")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a saved experiment, or configure a new one and run it")
    run_parser.add_argument(
        "name", nargs="?", default=None,
        help="Name of the experiment to run (omit to configure a new one interactively first)",
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a saved experiment without evaluating its test split")
    train_parser.add_argument("name", help="Name of the experiment to train")

    # Test command
    test_parser = subparsers.add_parser(
        "test", help="Evaluate an already-trained experiment against its held-out test split"
    )
    test_parser.add_argument("name", help="Name of the experiment to test")

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
    compare_parser.add_argument("--plotloss", action="store_true", help="Also save a comparison plot of loss curves")
    compare_parser.add_argument(
        "--logscale", action="store_true",
        help="Use a log-scale y-axis for --plotloss (helps see small changes late in training)",
    )

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

    # Plotloss command
    plotloss_parser = subparsers.add_parser("plotloss", help="Plot an experiment's recorded loss curve")
    plotloss_parser.add_argument("name", help="Name of the experiment to plot")
    plotloss_parser.add_argument(
        "--logscale", action="store_true",
        help="Use a log-scale y-axis (helps see small changes late in training)",
    )

    # Plot command (interactive multi-metric)
    plot_parser = subparsers.add_parser("plot", help="Interactively plot one or more experiments' metrics")
    plot_parser.add_argument("names", nargs="*", help="Names of experiments to plot (omit with --all)")
    plot_parser.add_argument("--all", action="store_true", help="Plot every experiment")
    plot_parser.add_argument(
        "--logscale", action="store_true",
        help="Use a log-scale y-axis where applicable (helps see small changes late in training)",
    )
    plot_parser.add_argument(
        "--metrics", nargs="+",
        choices=["loss", "accuracy", "gradient_norm", "test_loss", "test_accuracy"],
        help="Metric(s) to plot (default: prompted interactively)",
    )

    # Import command
    import_parser = subparsers.add_parser("import", help="Import a CSV file as a named dataset")
    import_parser.add_argument("csv_path", help="Path to the CSV file to import")
    import_parser.add_argument("--name", help="Name to register the dataset under (default: the file's name)")
    import_parser.add_argument(
        "--target", nargs="+", help="Output/target column(s) (default: prompted interactively)"
    )

    return parser


def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "init":
        init_project()
    elif args.command == "new":
        create_experiment()
    elif args.command == "list":
        list_experiments()
    elif args.command == "delete":
        delete_experiments(args.name, is_all=args.all)
    elif args.command == "run":
        name = args.name
        if name is None:
            config_path = create_experiment()
            name = config_path.parent.name
        results = run_experiment(name)
        success(f"Final loss: {results.final_loss}")
        if results.test_loss is not None:
            success(f"Test loss: {results.test_loss}")
            if results.test_accuracy is not None:
                success(f"Test accuracy: {results.test_accuracy:.2%}")
        _warn_if_stalled(results)
        console.print()
    elif args.command == "train":
        results = train_experiment(args.name)
        success(f"Final loss: {results.final_loss}")
        _warn_if_stalled(results)
        console.print()
    elif args.command == "test":
        results = test_experiment(args.name)
        if results is not None:
            if results.test_loss is not None:
                success(f"Test loss: {results.test_loss}")
            if results.test_accuracy is not None:
                success(f"Test accuracy: {results.test_accuracy:.2%}")
            console.print()
    elif args.command == "inspect":
        inspect_experiment(args.name)
    elif args.command == "compare":
        compare_experiments(args.names, is_all=args.all, plot_loss=args.plotloss, log_scale=args.logscale)
    elif args.command == "reproduce":
        reproduce_experiment(args.name)
    elif args.command == "copy":
        copy_experiment(args.source)
    elif args.command == "rename":
        rename_experiment(args.old_name, args.new_name)
    elif args.command == "plotloss":
        plot_experiment(args.name, log_scale=args.logscale)
    elif args.command == "plot":
        plot_experiments(args.names, is_all=args.all, log_scale=args.logscale, metrics=args.metrics)
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
