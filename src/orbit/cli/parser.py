import argparse

from orbit.cli.commands.new import create_experiment
from orbit.cli.commands.run import run_experiment
from orbit.cli.commands.import_dataset import import_dataset
from orbit.ui import success, warning

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

    subparsers.add_parser("new", help="Create a new experiment interactively")

    run_parser = subparsers.add_parser("run", help="Run a saved experiment")
    run_parser.add_argument("name", help="Name of the experiment to run")

    import_parser = subparsers.add_parser("import", help="Import a CSV file as a named dataset")
    import_parser.add_argument("csv_path", help="Path to the CSV file to import")
    import_parser.add_argument("--name", help="Name to register the dataset under (default: the file's name)")
    import_parser.add_argument(
        "--target", nargs="+", help="Output/target column(s) (default: prompted interactively)"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "new":
        create_experiment()
    elif args.command == "run":
        results = run_experiment(args.name)
        success(f"Final loss: {results.final_loss}")
        _warn_if_stalled(results)
    elif args.command == "import":
        import_dataset(args.csv_path, name=args.name, target_columns=args.target)
