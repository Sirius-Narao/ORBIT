"""
Decision boundaries for models with exactly two inputs: the input plane
colored by what the model predicts at every point, with the data on top -
what the model actually learned, not just how low its loss got.

- One output, classification (binary task, or 0/1 targets with no task):
  predicted probability on a diverging map, with the 0.5 boundary drawn.
- One output, regression: the predicted value on a sequential map.
- Several outputs: the argmax class's region.
"""
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from orbit.visualization.backend import pyplot

GRID_RESOLUTION = 200
GRID_MARGIN = 0.15
CLASS_CMAP = "tab10"


@dataclass
class BoundaryData:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_test: Optional[np.ndarray] = None
    Y_test: Optional[np.ndarray] = None
    task: Optional[str] = None
    axis_labels: Tuple[str, str] = ("Input 1", "Input 2")


def boundary_grid(X: np.ndarray, resolution: int = GRID_RESOLUTION, margin: float = GRID_MARGIN):
    """
    (xx, yy, points): a resolution x resolution mesh over X's two features,
    padded by `margin` of each feature's range, and the same mesh flattened
    to (resolution^2, 2) model inputs.
    """
    if X.shape[1] != 2:
        raise ValueError(f"A decision boundary needs exactly 2 input features, got {X.shape[1]}.")
    low, high = X.min(axis=0), X.max(axis=0)
    pad = np.maximum(high - low, 1e-6) * margin
    xs = np.linspace(low[0] - pad[0], high[0] + pad[0], resolution)
    ys = np.linspace(low[1] - pad[1], high[1] + pad[1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def class_labels(Y: np.ndarray, n_outputs: int) -> np.ndarray:
    """Integer class per sample, from one-hot rows or class-index targets."""
    Y = np.asarray(Y)
    if Y.ndim == 2 and Y.shape[1] == n_outputs and n_outputs > 1:
        return np.argmax(Y, axis=1)
    return Y.reshape(len(Y)).astype(int)


def is_classification(data: BoundaryData, n_outputs: int) -> bool:
    if n_outputs > 1 or data.task in ("binary_classification", "multiclass_classification"):
        return True
    if data.task is not None:
        return False
    return bool(np.isin(np.unique(data.Y_train), [0, 1]).all())


def draw_boundary(ax, xx, yy, predictions: np.ndarray, data: BoundaryData,
                  value_range: Optional[Tuple[float, float]] = None):
    """
    Draw one decision-boundary panel onto ax. predictions: the model's
    output on boundary_grid's points, shape (resolution^2, n_outputs).
    value_range fixes the regression color scale (e.g. across animation
    frames). Returns the filled-contour artist (for a colorbar).
    """
    import matplotlib
    from matplotlib.colors import BoundaryNorm, Normalize

    n_outputs = predictions.shape[1]
    shape = xx.shape

    if n_outputs > 1:
        cmap = matplotlib.colormaps[CLASS_CMAP]
        norm = BoundaryNorm(np.arange(-0.5, n_outputs + 0.5), cmap.N)
        regions = np.argmax(predictions, axis=1).reshape(shape)
        filled = ax.contourf(xx, yy, regions, levels=np.arange(-0.5, n_outputs + 0.5), cmap=cmap, norm=norm, alpha=0.35)
        train_colors = class_labels(data.Y_train, n_outputs)
        test_colors = class_labels(data.Y_test, n_outputs) if data.X_test is not None else None
    elif is_classification(data, n_outputs):
        cmap = matplotlib.colormaps["RdBu_r"]
        norm = Normalize(0.0, 1.0)
        surface = predictions[:, 0].reshape(shape)
        filled = ax.contourf(xx, yy, surface, levels=np.linspace(0, 1, 21), cmap=cmap, norm=norm, alpha=0.75, extend="both")
        if surface.min() < 0.5 < surface.max():
            ax.contour(xx, yy, surface, levels=[0.5], colors="black", linewidths=1.5)
        train_colors = data.Y_train[:, 0]
        test_colors = data.Y_test[:, 0] if data.X_test is not None else None
    else:
        cmap = matplotlib.colormaps["viridis"]
        low, high = value_range if value_range is not None else (data.Y_train.min(), data.Y_train.max())
        norm = Normalize(low, high)
        surface = predictions[:, 0].reshape(shape)
        filled = ax.contourf(xx, yy, surface, levels=np.linspace(low, high, 21), cmap=cmap, norm=norm, alpha=0.75, extend="both")
        train_colors = data.Y_train[:, 0]
        test_colors = data.Y_test[:, 0] if data.X_test is not None else None

    ax.scatter(data.X_train[:, 0], data.X_train[:, 1], c=train_colors, cmap=cmap, norm=norm,
               edgecolors="black", linewidths=1.0, s=70, zorder=3, label="train")
    if data.X_test is not None:
        ax.scatter(data.X_test[:, 0], data.X_test[:, 1], facecolors="none",
                   edgecolors=cmap(norm(test_colors)), linewidths=2.0, s=80, zorder=3, label="test")
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(data.axis_labels[0])
    ax.set_ylabel(data.axis_labels[1])
    return filled


def _colorbar(fig, ax, filled, data: BoundaryData, n_outputs: int):
    if n_outputs > 1:
        bar = fig.colorbar(filled, ax=ax, ticks=range(n_outputs))
        bar.set_label("Predicted class")
    elif is_classification(data, n_outputs):
        bar = fig.colorbar(filled, ax=ax, ticks=[0, 0.5, 1])
        bar.set_label("Predicted output")
    else:
        bar = fig.colorbar(filled, ax=ax)
        bar.set_label("Predicted value")


def plot_boundary(xx, yy, predictions: np.ndarray, data: BoundaryData, title: str,
                  output_path: pathlib.Path, show: bool = False) -> pathlib.Path:
    from orbit.visualization.backend import can_show

    plt = pyplot(show)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    filled = draw_boundary(ax, xx, yy, predictions, data)
    _colorbar(fig, ax, filled, data, predictions.shape[1])
    ax.set_title(title)

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show and can_show(plt):
        plt.show()
    plt.close(fig)
    return output_path


def animate_boundary(xx, yy, frames: List[Tuple[int, np.ndarray]], data: BoundaryData, title: str,
                     output_path: pathlib.Path, value_range: Optional[Tuple[float, float]] = None,
                     video_format: str = "gif", fps: int = 10, show: bool = False) -> pathlib.Path:
    """
    frames: [(epoch, predictions on the grid), ...], one per weight
    snapshot. value_range should span every frame (regression) so the
    colors mean the same thing throughout.
    """
    from orbit.visualization.animation import save_animation

    if not frames:
        raise ValueError("No frames to animate.")

    plt = pyplot(show)
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_axes([0.1, 0.1, 0.68, 0.78])
    cax = fig.add_axes([0.82, 0.1, 0.03, 0.78])
    n_outputs = frames[0][1].shape[1]

    def update(index):
        epoch, predictions = frames[index]
        ax.clear()
        cax.clear()
        filled = draw_boundary(ax, xx, yy, predictions, data, value_range=value_range)
        fig.colorbar(filled, cax=cax)
        ax.set_title(f"{title} — epoch {epoch}", fontsize=11)

    return save_animation(fig, update, len(frames), output_path, video_format, fps, show, plt)
