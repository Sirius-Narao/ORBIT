import pathlib
from typing import Optional

from rich.table import Table

from orbit.cli.commands.network import load_trained_experiment, training_arrays
from orbit.nn.introspection import forward_trace, network_structure
from orbit.storage import EXPERIMENTS_ROOT, experiment_dir
from orbit.ui import console, success, warning
from orbit.visualization.health import hidden_layer_health, plot_activation_health


def _percent(value) -> str:
    return "-" if value is None else f"{value:.0%}"


def health_experiment(
    name: str, output: Optional[str] = None, root: pathlib.Path = EXPERIMENTS_ROOT
) -> Optional[pathlib.Path]:
    """
    Check how every hidden layer's units behave over the training set: a
    table of mean/std plus the dead (ReLU) or saturated (Tanh/Sigmoid)
    fraction, a warning per unhealthy layer, and a PNG of per-layer
    activation histograms.
    """
    loaded = load_trained_experiment(name, root)
    if loaded is None:
        return None
    _, experiment, trained = loaded

    structure = network_structure(experiment.model)
    if len(structure) < 2:
        warning(f"{name} has no hidden layers - nothing to check.")
        return None

    X, _ = training_arrays(experiment)
    trace = forward_trace(experiment.model, X)
    health = hidden_layer_health(structure, trace)

    table = Table(title=f"{name} — activation health over {len(X)} training samples", title_style="bold #ffeab0")
    for column in ("Layer", "Units", "Mean", "Std", "Dead (ReLU)", "Saturated"):
        table.add_column(column, justify="left" if column == "Layer" else "right")
    for layer in health:
        style = "#ffb0b0" if layer.problems else None
        table.add_row(
            layer.label, str(layer.units), f"{layer.mean:.3g}", f"{layer.std:.3g}",
            _percent(layer.dead_fraction), _percent(layer.saturated_fraction), style=style,
        )
    console.print()
    console.print(table)

    if not trained:
        warning(f"{name} has not been run yet - these are its initial (untrained) activations.")
    for layer in health:
        for problem in layer.problems:
            warning(f"{layer.label}: {problem}.")
    if any(layer.problems for layer in health):
        warning(
            "Dead or saturated units barely pass gradient. Try a lower learning rate, "
            "input normalization (\"normalize\"), or a different activation."
        )

    title = f"{name} — Activation health" + ("" if trained else " (untrained)")
    output_path = pathlib.Path(output) if output else experiment_dir(name, root=root) / "results" / "health.png"
    plot_activation_health(health, trace, title, output_path)
    success(f"Saved activation histograms to {output_path}")
    console.print()
    return output_path
