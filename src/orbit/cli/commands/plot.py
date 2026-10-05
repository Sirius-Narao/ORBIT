import pathlib
from typing import List, Optional

import questionary

from orbit.storage import experiment_dir, load_results
from orbit.ui import console, success, warning, PROMPT_STYLE
from orbit.visualization import (
    plot_loss,
    plot_loss_comparison,
    plot_accuracy,
    plot_accuracy_comparison,
    plot_gradient_norm,
    plot_gradient_norm_comparison,
    plot_layer_gradient_norms,
    plot_test_loss,
    plot_test_accuracy,
)
from orbit.cli.commands.compare import _comparison_filename
from orbit.storage import workspace
from orbit.storage.workspace import experiments_root

# Ordered so questionary.checkbox presents per-epoch histories first, then
# the two scalar test metrics.
_METRIC_CHOICES = [
    ("Training loss", "loss"),
    ("Accuracy / R²", "accuracy"),
    ("Gradient norm", "gradient_norm"),
    ("Gradient norm per layer", "layer_gradient_norm"),
    ("Test loss", "test_loss"),
    ("Test accuracy / R²", "test_accuracy"),
]
_LABEL_TO_KEY = dict(_METRIC_CHOICES)
_VALID_KEYS = set(_LABEL_TO_KEY.values())

# Each line-metric entry: (history attribute, single-plot fn, comparison-plot fn, filename slug)
_LINE_METRICS = {
    "loss": ("loss_history", plot_loss, plot_loss_comparison, "loss"),
    "accuracy": ("accuracy_history", plot_accuracy, plot_accuracy_comparison, "accuracy"),
    "gradient_norm": (
        "gradient_norm_history", plot_gradient_norm, plot_gradient_norm_comparison, "gradient_norm",
    ),
}
# Per-experiment metrics: several lines per experiment already (one per
# layer), so experiments are never overlaid - each gets its own file in its
# own results/ dir, even when several are plotted at once.
# Entry: (history attribute, plot fn, filename slug)
_PER_EXPERIMENT_METRICS = {
    "layer_gradient_norm": ("layer_gradient_norm_history", plot_layer_gradient_norms, "layer_gradient_norm"),
}
# Each bar-metric entry: (scalar attribute, bar-plot fn, filename slug)
_BAR_METRICS = {
    "test_loss": ("test_loss", plot_test_loss, "test_loss"),
    "test_accuracy": ("test_accuracy", plot_test_accuracy, "test_accuracy"),
}


def plot_experiments(
    names: Optional[list] = None,
    is_all: bool = False,
    log_scale: bool = False,
    metrics: Optional[list] = None,
    root: Optional[pathlib.Path] = None,
    comparisons_root: Optional[pathlib.Path] = None,
) -> List[pathlib.Path]:
    root = experiments_root(root)
    comparisons_root = workspace.comparisons_root(comparisons_root)
    if is_all:
        if not root.exists():
            console.print()
            warning("No experiments found.")
            console.print()
            return []
        names = sorted(p.name for p in root.iterdir() if p.is_dir())

    if not names:
        console.print()
        warning("No experiments found.")
        console.print()
        return []

    results_list = []
    for name in names:
        exp_dir = experiment_dir(name, root=root)
        config_path = exp_dir / "experiment.json"
        if not config_path.exists():
            warning(f"{name} was not found, skipping.")
            continue

        results_path = exp_dir / "results" / "results.json"
        if not results_path.exists():
            warning(f"{name} has not been run yet, skipping.")
            continue

        results_list.append((name, exp_dir, load_results(results_path)))

    if not results_list:
        console.print()
        warning("No experiments with results to plot.")
        console.print()
        return []

    if metrics is None:
        chosen_labels = questionary.checkbox(
            "Which metric(s) would you like to plot?",
            choices=[label for label, _ in _METRIC_CHOICES],
            style=PROMPT_STYLE,
        ).ask()
        if not chosen_labels:
            console.print()
            warning("No metrics selected.")
            console.print()
            return []
        metric_keys = [_LABEL_TO_KEY[label] for label in chosen_labels]
    else:
        unknown = [m for m in metrics if m not in _VALID_KEYS]
        if unknown:
            raise ValueError(f"Unknown metric(s): {unknown!r} (valid: {sorted(_VALID_KEYS)})")
        metric_keys = metrics

    is_multi = len(results_list) > 1
    extension = "_log_scale" if log_scale else ""
    output_paths = []

    console.print()
    for key in metric_keys:
        if key in _PER_EXPERIMENT_METRICS:
            attr, plot_fn, slug = _PER_EXPERIMENT_METRICS[key]
            for n, exp_dir, r in results_list:
                if not getattr(r, attr):
                    warning(f"{n} has no recorded {key.replace('_', ' ')} (re-run it to record one), skipping.")
                    continue
                output_path = plot_fn(r, exp_dir / "results" / f"{slug}{extension}.png", log_scale=log_scale)
                success(f"Saved {key.replace('_', ' ')} plot to {output_path}")
                output_paths.append(output_path)
            continue

        is_bar = key in _BAR_METRICS
        if is_bar:
            attr, bar_fn, slug = _BAR_METRICS[key]
            eligible = [(n, d, r) for n, d, r in results_list if getattr(r, attr) is not None]
        else:
            attr, single_fn, multi_fn, slug = _LINE_METRICS[key]
            eligible = [(n, d, r) for n, d, r in results_list if getattr(r, attr)]

        missing = [n for n, _, _ in results_list if n not in {e[0] for e in eligible}]
        for n in missing:
            warning(f"{n} has no recorded {key.replace('_', ' ')}, skipping for this metric.")

        if not eligible:
            warning(f"No experiments have {key.replace('_', ' ')}, skipping this metric entirely.")
            continue

        if is_multi:
            output_path = comparisons_root / _comparison_filename(
                [r.name for _, _, r in eligible], log_scale=log_scale, metric=slug
            )
            eligible_results = [r for _, _, r in eligible]
            plot_fn = bar_fn if is_bar else multi_fn
            output_path = plot_fn(eligible_results, output_path, log_scale=log_scale)
        else:
            _, exp_dir, r = eligible[0]
            output_path = exp_dir / "results" / f"{slug}{extension}.png"
            output_path = bar_fn([r], output_path, log_scale=log_scale) if is_bar \
                else single_fn(r, output_path, log_scale=log_scale)

        success(f"Saved {key.replace('_', ' ')} plot to {output_path}")
        output_paths.append(output_path)

    if not output_paths:
        warning("Nothing was plotted.")

    console.print()
    return output_paths
