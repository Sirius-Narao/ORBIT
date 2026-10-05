import pathlib
from typing import List

from orbit.core import Results
from orbit.visualization.comparison_style import add_legend, line_colors
from orbit.visualization.theme import theme_color, themed


@themed
def plot_gradient_norm(results: Results, output_path: pathlib.Path, log_scale: bool = False) -> pathlib.Path:
    """
    Render results.gradient_norm_history as a line plot and save it to
    output_path. See visualization/loss.py's plot_loss for why matplotlib
    is imported lazily here and what log_scale does.
    """
    if not results.gradient_norm_history:
        raise ValueError(f"{results.name} has no recorded gradient norm history to plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    epochs = range(1, len(results.gradient_norm_history) + 1)
    ax.plot(epochs, results.gradient_norm_history, color=theme_color("accent"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    title = f"{results.name} — Gradient Norm"
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)

    fig.savefig(output_path)
    plt.close(fig)

    return output_path


@themed
def plot_gradient_norm_comparison(
    results_list: List[Results], output_path: pathlib.Path, log_scale: bool = False
) -> pathlib.Path:
    """
    Overlay each Results' gradient_norm_history on one line plot and save
    it to output_path. See plot_loss_comparison's docstring for the
    rationale behind the multi-color legend instead of a single fixed
    accent color.
    """
    if not results_list:
        raise ValueError("No results to compare.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    for results, color in zip(results_list, line_colors(len(results_list))):
        epochs = range(1, len(results.gradient_norm_history) + 1)
        ax.plot(epochs, results.gradient_norm_history, label=results.name, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    title = "Gradient Norm Comparison"
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)
    add_legend(ax, len(results_list))

    # bbox_inches="tight" keeps an outside legend (many lines) from being cropped.
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path


@themed
def plot_layer_gradient_norms(results: Results, output_path: pathlib.Path, log_scale: bool = False) -> pathlib.Path:
    """
    One line per weight matrix's gradient norm (results.layer_gradient_norm_history),
    first layer first - where the global gradient norm can hide that the
    early layers barely learn (vanishing gradients) or blow up. Log scale
    is usually the readable choice here, since layers can differ by orders
    of magnitude.
    """
    if not results.layer_gradient_norm_history:
        raise ValueError(f"{results.name} has no recorded per-layer gradient norms to plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    layers = list(results.layer_gradient_norm_history.items())
    fig, ax = plt.subplots()
    for index, ((name, norms), color) in enumerate(zip(layers, line_colors(len(layers)))):
        epochs = range(1, len(norms) + 1)
        ax.plot(epochs, norms, label=f"Linear {index + 1} ({name})", color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    title = f"{results.name} — Gradient Norm per Layer"
    if log_scale:
        ax.set_yscale("log")
        title += " (log scale)"
    ax.set_title(title)
    add_legend(ax, len(layers))

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path
