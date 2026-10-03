import pathlib
from typing import List

from orbit.core import Results
from orbit.visualization.accuracy import accuracy_axis_label
from orbit.visualization.comparison_style import bar_figure_height, use_horizontal_bars


def _bar_chart(
    results_list: List[Results],
    *,
    value_fn,
    ylabel: str,
    title: str,
    output_path: pathlib.Path,
    log_scale: bool,
) -> pathlib.Path:
    """
    Shared bar-chart renderer for scalar per-experiment metrics
    (test_loss/test_accuracy). Unlike loss/accuracy/gradient norm, these
    aren't per-epoch histories, so there's no line to plot - one bar per
    experiment generalizes a single experiment (1 bar) and a comparison
    (N bars) with no code-shape difference, unlike the line-plot metrics'
    separate single/comparison functions.

    A handful of bars render vertically, as they always have. Past that
    (e.g. a sweep's runs), rotated names under vertical bars overlap and get
    cropped, so the chart flips to horizontal bars - names down the y-axis,
    first experiment at the top, each bar labeled with its value since
    reading 20+ bars off an axis is impractical - in a figure that grows
    taller with the bar count. See comparison_style.py.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [r.name for r in results_list]
    values = [value_fn(r) for r in results_list]

    if use_horizontal_bars(len(names)):
        fig, ax = plt.subplots(figsize=(6.4, bar_figure_height(len(names))))
        bars = ax.barh(names, values, color="#b0ffb3")
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.4g", padding=3, fontsize="small")
        ax.set_ylabel("Experiment")
        ax.set_xlabel(ylabel)
        if log_scale:
            ax.set_xscale("log")
        # headroom so the longest bar's value label stays inside the axes
        ax.margins(x=0.15, y=0.01)
    else:
        fig, ax = plt.subplots()
        ax.bar(names, values, color="#b0ffb3")
        ax.set_xlabel("Experiment")
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        fig.autofmt_xdate(rotation=30)

    if log_scale:
        title += " (log scale)"
    ax.set_title(title)

    # bbox_inches="tight" keeps long experiment names from being cropped.
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_test_loss(
    results_list: List[Results], output_path: pathlib.Path, log_scale: bool = False
) -> pathlib.Path:
    if not results_list:
        raise ValueError("No results to plot.")
    return _bar_chart(
        results_list, value_fn=lambda r: r.test_loss, ylabel="Test Loss",
        title="Test Loss", output_path=output_path, log_scale=log_scale,
    )


def plot_test_accuracy(
    results_list: List[Results], output_path: pathlib.Path, log_scale: bool = False
) -> pathlib.Path:
    if not results_list:
        raise ValueError("No results to plot.")
    label = f"Test {accuracy_axis_label(results_list)}"
    return _bar_chart(
        results_list, value_fn=lambda r: r.test_accuracy, ylabel=label,
        title=label, output_path=output_path, log_scale=log_scale,
    )
