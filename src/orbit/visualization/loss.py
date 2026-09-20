import pathlib

from orbit.core import Results


def plot_loss(results: Results, output_path: pathlib.Path) -> pathlib.Path:
    """
    Render results.loss_history as a line plot and save it to output_path.

    matplotlib is imported lazily here (not at module scope) because
    cli/parser.py imports every cli/commands/*.py module unconditionally at
    startup - a module-level matplotlib import would make every orbit
    invocation, even orbit list, pay matplotlib's import cost. The Agg
    backend is forced since this only ever saves to disk - there's no
    --show/interactive-window option in v1.
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
    ax.set_title(f"{results.name} — Training Loss")

    fig.savefig(output_path)
    plt.close(fig)

    return output_path
