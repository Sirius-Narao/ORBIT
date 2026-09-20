import pathlib
from typing import List

from orbit.core import Results


def plot_loss(results: Results, output_path: pathlib.Path, log_scale: bool = False) -> pathlib.Path:
    """
    Render results.loss_history as a line plot and save it to output_path.

    matplotlib is imported lazily here (not at module scope) because
    cli/parser.py imports every cli/commands/*.py module unconditionally at
    startup - a module-level matplotlib import would make every orbit
    invocation, even orbit list, pay matplotlib's import cost. The Agg
    backend is forced since this only ever saves to disk - there's no
    --show/interactive-window option in v1.

    log_scale renders the y-axis on a log scale instead of linear - loss
    typically drops fast early on then changes much more slowly later, which
    on a linear axis squashes those later, often more interesting
    differences down near zero.
    """
    if not results.loss_history:
        raise ValueError(f"{results.name} has no recorded loss history to plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    epochs = range(1, len(results.loss_history) + 1)
    ax.plot(epochs, results.loss_history, color="#b0ffb3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title = f"{results.name} — Training Loss"
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)

    fig.savefig(output_path)
    plt.close(fig)

    return output_path


def plot_loss_comparison(
    results_list: List[Results], output_path: pathlib.Path, log_scale: bool = False
) -> pathlib.Path:
    """
    Overlay each Results' loss_history on one line plot and save it to
    output_path. See plot_loss's docstring for why matplotlib is imported
    lazily here, and what log_scale does. Unlike plot_loss's single fixed
    accent color, multiple lines need matplotlib's default color cycle to
    stay distinguishable.
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
        epochs = range(1, len(results.loss_history) + 1)
        ax.plot(epochs, results.loss_history, label=results.name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    title = "Training Loss Comparison"
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)
    ax.legend()

    fig.savefig(output_path)
    plt.close(fig)

    return output_path
