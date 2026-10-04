from orbit.cli.commands.network import training_arrays
from orbit.core import Results
from orbit.ui import is_tty, warning


def run_with_optional_watch(experiment, name: str, watch: bool, skip_test: bool = False) -> Results:
    """
    experiment.run(skip_test=...), optionally under the live terminal view
    (--watch). That view replaces the usual progress bar - two rich Live
    displays can't be shown at once - and needs a real terminal; without
    one it falls back to the normal output with a warning.
    """
    if not watch:
        return experiment.run(skip_test=skip_test)

    if not is_tty():
        warning("--watch needs an interactive terminal - showing the normal output instead.")
        return experiment.run(skip_test=skip_test)

    from orbit.visualization.terminal import WatchView

    X, _ = training_arrays(experiment)
    experiment.verbose = False
    with WatchView(experiment.model, X, experiment.epochs, name) as view:
        return experiment.run(skip_test=skip_test, on_epoch_end=view)
