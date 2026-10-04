import argparse
import sys

from orbit.cli.commands.init import init_project
from orbit.cli.commands.new import create_experiment
from orbit.cli.commands.run import DIVERGENCE_HINT, run_experiment
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
from orbit.cli.commands.network import network_experiment
from orbit.cli.commands.animate import animate_experiment
from orbit.cli.commands.boundary import boundary_experiment
from orbit.cli.commands.health import health_experiment
from orbit.cli.commands.sweep import (
    create_sweep,
    start_sweep,
    sweep_status,
    compare_sweep,
    export_sweep,
    plot_sweep,
    delete_sweep,
)
from orbit.core.ranking import RANK_METRICS
from orbit.core.metrics import format_metric, metric_label
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
    if len(results.loss_history) < 2 or results.loss_history[0] == 0:
        return
    first_loss = results.loss_history[0]

    improvement = (first_loss - results.final_loss) / first_loss
    if improvement < STALLED_RUN_IMPROVEMENT_THRESHOLD:
        warning(
            f"Warning: loss barely improved ({first_loss:.4f} -> "
            f"{results.final_loss:.4f}). The model may be too small for "
            "this dataset, the learning rate may be too low, or it may "
            "need more epochs."
        )


def _report_training(results, show_test_metrics: bool = True) -> None:
    """
    The one-line outcome of orbit run/train: the final loss, or - if the
    loss blew up to inf/NaN and training stopped early - what happened and
    what to try instead (the stalled-run check is meaningless then).
    """
    diverged_at = getattr(results, "diverged_at_epoch", None)
    if diverged_at is not None:
        warning(
            f"Training diverged at epoch {diverged_at} - the loss became inf/NaN, "
            f"so training stopped early. {DIVERGENCE_HINT}"
        )
        return
    success(f"Final loss: {results.final_loss}")
    if show_test_metrics:
        _report_test_metrics(results)
    _warn_if_stalled(results)


def _report_test_metrics(results) -> None:
    if results.test_loss is not None:
        success(f"Test loss: {results.test_loss}")
    if results.test_accuracy is not None:
        task = results.hyperparams.get("task")
        label = metric_label(task)
        name = "accuracy" if label == "Accuracy" else label
        success(f"Test {name}: {format_metric(results.test_accuracy, task)}")


_BY_HELP = (
    "Rank by one or more metrics - " + ", ".join(RANK_METRICS) + "; several are combined by "
    "average rank. A bare --by asks which ones interactively"
)


_SHOW_HELP = "Also open the figure in a window (not available inside the orbit REPL)"


_WATCH_HELP = "Watch the network learn live in the terminal (each neuron as a colored cell) instead of a progress bar"


_TEST_HELP = "Rank by the test metrics only (test_loss, test_accuracy - whichever were recorded)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="orbit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new ORBIT project (.orbits/ directory structure)")

    # New command
    new_parser = subparsers.add_parser("new", help="Create a new experiment interactively")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all experiments and their results")
    list_parser.add_argument(
        "--runs", action="store_true", help="Also list sweep runs individually (hidden by default)"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a saved experiment, or configure a new one and run it")
    run_parser.add_argument(
        "name", nargs="?", default=None,
        help="Name of the experiment to run (omit to configure a new one interactively first)",
    )
    run_parser.add_argument("--watch", action="store_true", help=_WATCH_HELP)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a saved experiment without evaluating its test split")
    train_parser.add_argument("name", help="Name of the experiment to train")
    train_parser.add_argument("--watch", action="store_true", help=_WATCH_HELP)

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
    compare_metrics = compare_parser.add_mutually_exclusive_group()
    compare_metrics.add_argument(
        "--by", nargs="*", choices=list(RANK_METRICS), metavar="METRIC",
        help=_BY_HELP + " (default: no ranking)",
    )
    compare_metrics.add_argument("--test", action="store_true", help=_TEST_HELP)

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
        choices=["loss", "accuracy", "gradient_norm", "layer_gradient_norm", "test_loss", "test_accuracy"],
        help="Metric(s) to plot (default: prompted interactively)",
    )

    # Network command (diagram of the model itself)
    network_parser = subparsers.add_parser(
        "network", help="Draw an experiment's network: nodes, weights and activations"
    )
    network_parser.add_argument("name", help="Name of the experiment to draw")
    network_parser.add_argument(
        "--sample", type=int,
        help="Color nodes by this training sample's activations (default: mean over the training set)",
    )
    network_parser.add_argument("--output", help="PNG path (default: .orbits/experiments/<name>/results/)")
    network_parser.add_argument("--show", action="store_true", help=_SHOW_HELP)

    # Animate command (training animation from recorded weight snapshots)
    animate_parser = subparsers.add_parser(
        "animate", help="Animate an experiment's training: the network and loss, epoch by epoch"
    )
    animate_parser.add_argument("name", help="Name of the experiment to animate")
    animate_parser.add_argument(
        "--sample", type=int,
        help="Color nodes by this training sample's activations (default: mean over the training set)",
    )
    animate_parser.add_argument(
        "--mp4", action="store_true", help="Save an MP4 instead of a GIF (needs ffmpeg)"
    )
    animate_parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    animate_parser.add_argument("--output", help="Output path (default: .orbits/experiments/<name>/results/)")
    animate_parser.add_argument("--logscale", action="store_true", help="Use a log-scale y-axis for the loss")
    animate_parser.add_argument("--show", action="store_true", help=_SHOW_HELP)

    # Boundary command (decision boundary of a 2-input model)
    boundary_parser = subparsers.add_parser(
        "boundary", help="Plot a 2-input model's decision boundary over its data"
    )
    boundary_parser.add_argument("name", help="Name of the experiment to plot")
    boundary_parser.add_argument(
        "--animate", action="store_true", help="Animate the boundary forming over training (GIF by default)"
    )
    boundary_parser.add_argument("--mp4", action="store_true", help="With --animate: save an MP4 (needs ffmpeg)")
    boundary_parser.add_argument("--fps", type=int, default=10, help="With --animate: frames per second (default: 10)")
    boundary_parser.add_argument("--output", help="Output path (default: .orbits/experiments/<name>/results/)")
    boundary_parser.add_argument("--show", action="store_true", help=_SHOW_HELP)

    # Health command (dead/saturated units per hidden layer)
    health_parser = subparsers.add_parser(
        "health", help="Check every hidden layer for dead or saturated units"
    )
    health_parser.add_argument("name", help="Name of the experiment to check")
    health_parser.add_argument("--output", help="PNG path (default: .orbits/experiments/<name>/results/health.png)")

    # Sweep command (nested subcommands)
    sweep_parser = subparsers.add_parser("sweep", help="Hyperparameter sweeps over a base experiment")
    sweep_subparsers = sweep_parser.add_subparsers(dest="sweep_command", required=True)

    sweep_create_parser = sweep_subparsers.add_parser(
        "create", help="Interactively define a grid over a base experiment's hyperparameters"
    )
    sweep_create_parser.add_argument("name", help="Name of the new sweep")
    sweep_create_parser.add_argument("--base", required=True, help="Experiment to use as the base config")

    sweep_start_parser = sweep_subparsers.add_parser(
        "start", help="Train every run not done yet (re-run to resume an interrupted sweep)"
    )
    sweep_start_parser.add_argument("name", help="Name of the sweep")

    sweep_status_parser = sweep_subparsers.add_parser("status", help="Show each run's status")
    sweep_status_parser.add_argument("name", help="Name of the sweep")

    sweep_compare_parser = sweep_subparsers.add_parser("compare", help="Rank the sweep's runs and show the best")
    sweep_compare_parser.add_argument("name", help="Name of the sweep")
    sweep_compare_metrics = sweep_compare_parser.add_mutually_exclusive_group()
    sweep_compare_metrics.add_argument(
        "--by", nargs="*", choices=list(RANK_METRICS), metavar="METRIC",
        help=_BY_HELP + " (default: test_loss if the sweep has a test split, else final_loss)",
    )
    sweep_compare_metrics.add_argument(
        "--all", action="store_true", dest="all_metrics",
        help="Rank by every quality metric the sweep's runs recorded (all but duration_seconds)",
    )
    sweep_compare_metrics.add_argument("--test", action="store_true", help=_TEST_HELP)

    sweep_export_parser = sweep_subparsers.add_parser("export", help="Export every run's params and metrics to CSV")
    sweep_export_parser.add_argument("name", help="Name of the sweep")
    sweep_export_parser.add_argument("--output", help="CSV path (default: .orbits/sweeps/<name>/results.csv)")

    sweep_plot_parser = sweep_subparsers.add_parser("plot", help="Plot the sweep's finished runs")
    sweep_plot_parser.add_argument("name", help="Name of the sweep")
    sweep_plot_parser.add_argument(
        "--logscale", action="store_true",
        help="Use a log-scale y-axis where applicable (helps see small changes late in training)",
    )
    sweep_plot_parser.add_argument(
        "--metrics", nargs="+",
        choices=["loss", "accuracy", "gradient_norm", "layer_gradient_norm", "test_loss", "test_accuracy"],
        help="Metric(s) to plot (default: prompted interactively)",
    )

    sweep_delete_parser = sweep_subparsers.add_parser(
        "delete", help="Delete a sweep and every run in it (asks first)"
    )
    sweep_delete_parser.add_argument("name", help="Name of the sweep")
    sweep_delete_parser.add_argument("--yes", action="store_true", help="Delete without asking")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import a CSV file as a named dataset")
    import_parser.add_argument("csv_path", help="Path to the CSV file to import")
    import_parser.add_argument("--name", help="Name to register the dataset under (default: the file's name)")
    import_parser.add_argument(
        "--target", nargs="+", help="Output/target column(s) (default: prompted interactively)"
    )

    return parser


def _resolve_show(args: argparse.Namespace, in_repl: bool) -> bool:
    """
    --show opens a matplotlib window, which needs an interactive backend.
    Switching backends inside the REPL's long-lived process is fragile, so
    there the file is saved and the window skipped.
    """
    if not getattr(args, "show", False):
        return False
    if in_repl:
        warning("--show is not available in the REPL - open the saved file instead.")
        return False
    return True


def _dispatch(args: argparse.Namespace, in_repl: bool = False) -> None:
    if args.command == "init":
        init_project()
    elif args.command == "new":
        create_experiment()
    elif args.command == "list":
        list_experiments(show_runs=args.runs)
    elif args.command == "delete":
        delete_experiments(args.name, is_all=args.all)
    elif args.command == "run":
        name = args.name
        if name is None:
            config_path = create_experiment()
            name = config_path.parent.name
        results = run_experiment(name, watch=args.watch)
        _report_training(results)
        console.print()
    elif args.command == "train":
        results = train_experiment(args.name, watch=args.watch)
        _report_training(results, show_test_metrics=False)
        console.print()
    elif args.command == "test":
        results = test_experiment(args.name)
        if results is not None:
            _report_test_metrics(results)
            console.print()
    elif args.command == "inspect":
        inspect_experiment(args.name)
    elif args.command == "compare":
        compare_experiments(
            args.names, is_all=args.all, plot_loss=args.plotloss, log_scale=args.logscale, by=args.by,
            test_only=args.test,
        )
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
    elif args.command == "network":
        network_experiment(
            args.name, sample=args.sample, output=args.output, show=_resolve_show(args, in_repl)
        )
    elif args.command == "animate":
        animate_experiment(
            args.name, sample=args.sample, video_format="mp4" if args.mp4 else "gif", fps=args.fps,
            output=args.output, log_scale=args.logscale, show=_resolve_show(args, in_repl),
        )
    elif args.command == "boundary":
        boundary_experiment(
            args.name, animate=args.animate, video_format="mp4" if args.mp4 else "gif", fps=args.fps,
            output=args.output, show=_resolve_show(args, in_repl),
        )
    elif args.command == "health":
        health_experiment(args.name, output=args.output)
    elif args.command == "sweep":
        _dispatch_sweep(args)
    elif args.command == "import":
        import_dataset(args.csv_path, name=args.name, target_columns=args.target)


def _dispatch_sweep(args: argparse.Namespace) -> None:
    if args.sweep_command == "create":
        create_sweep(args.name, args.base)
    elif args.sweep_command == "start":
        start_sweep(args.name)
    elif args.sweep_command == "status":
        sweep_status(args.name)
    elif args.sweep_command == "compare":
        compare_sweep(args.name, by=args.by, all_metrics=args.all_metrics, test_only=args.test)
    elif args.sweep_command == "export":
        export_sweep(args.name, output=args.output)
    elif args.sweep_command == "plot":
        plot_sweep(args.name, log_scale=args.logscale, metrics=args.metrics)
    elif args.sweep_command == "delete":
        delete_sweep(args.name, yes=args.yes)


def main() -> None:
    if len(sys.argv) <= 1:
        from orbit.cli.repl import repl

        repl()
        return

    parser = build_parser()
    args = parser.parse_args()
    _dispatch(args)
