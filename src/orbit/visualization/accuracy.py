import pathlib
from typing import List

from orbit.core import Results


def plot_accuracy(results: Results, output_path: pathlib.Path, log_scale: bool = False) -> pathlib.Path:
    """
    Render results.accuracy_history as a line plot and save it to output_path.
    See visualization/loss.py's plot_loss for why matplotlib is imported
    lazily here and what log_scale does. Values are plotted as the raw 0-1
    fraction accuracy_history stores, not multiplied by 100.
    """
    if not results.accuracy_history:
        raise ValueError(f"{results.name} has no recorded accuracy history to plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    epochs = range(1, len(results.accuracy_history) + 1)
    ax.plot(epochs, results.accuracy_history, color="#b0ffb3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    title = f"{results.name} — Accuracy"
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)

    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_accuracy_comparison(
    results_list: List[Results], output_path: pathlib.Path, log_scale: bool = False
) -> pathlib.Path:
    """
    Overlay each Results' accuracy_history on one line plot and save it to
    output_path. See plot_loss_comparison's docstring for the rationale
    behind the multi-color legend instead of a single fixed accent color.
    """
    if not results_list:
        raise ValueError("No results to compare.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for results in results_list:
        epochs = range(1, len(results.accuracy_history) + 1)
        ax.plot(epochs, results.accuracy_history, label=results.name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    title = "Accuracy Comparison"
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)

    return output_path
