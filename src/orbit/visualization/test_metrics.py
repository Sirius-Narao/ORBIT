import pathlib
from typing import List

from orbit.core import Results
from orbit.visualization.accuracy import accuracy_axis_label


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
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = [r.name for r in results_list]
    values = [value_fn(r) for r in results_list]

    fig, ax = plt.subplots()
    ax.bar(names, values, color="#b0ffb3")
    ax.set_xlabel("Experiment")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)
    fig.autofmt_xdate(rotation=30)

    fig.savefig(output_path)
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
