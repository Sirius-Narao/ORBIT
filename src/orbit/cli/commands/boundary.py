import pathlib
from typing import Optional

import numpy as np
from rich.markup import escape

from orbit.cli.commands.animate import snapshot_weights
from orbit.cli.commands.network import load_trained_experiment, training_arrays
from orbit.nn.introspection import forward_trace, network_structure, set_weights
from orbit.storage import dataset_exists, experiment_dir, load_dataset_manifest
from orbit.ui import info, success, warning
from orbit.visualization.animation import DEFAULT_FPS, VideoWriterError
from orbit.visualization.boundary import BoundaryData, animate_boundary, boundary_grid, plot_boundary
from orbit.storage.workspace import experiments_root


def _stack(dataset):
    samples = [dataset[i] for i in range(len(dataset))]
    return (
        np.stack([np.asarray(x, dtype=float) for x, _ in samples]),
        np.stack([np.asarray(y, dtype=float).reshape(-1) for _, y in samples]),
    )


def _axis_labels(config: dict):
    """The two input column names for an imported dataset, else generic ones."""
    names = ["Input 1", "Input 2"]
    if dataset_exists(config["dataset"]):
        names = list(load_dataset_manifest(config["dataset"])["input_columns"])[:2]
    if config.get("normalize", "none") != "none":
        names = [f"{label} (normalized)" for label in names]
    return tuple(names)


def boundary_experiment(
    name: str,
    animate: bool = False,
    video_format: str = "gif",
    fps: int = DEFAULT_FPS,
    output: Optional[str] = None,
    show: bool = False,
    root: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    """
    Plot what a 2-input model predicts over the whole input plane, with the
    training (filled) and test (hollow) points on top. animate=True renders
    it at every recorded weight snapshot instead, so you watch the boundary
    form.
    """
    root = experiments_root(root)
    loaded = load_trained_experiment(name, root)
    if loaded is None:
        return None
    config, experiment, trained = loaded

    X_train, Y_train = training_arrays(experiment)
    Y_train = Y_train.reshape(len(Y_train), -1)
    if X_train.shape[1] != 2:
        warning(
            f"A decision boundary needs exactly 2 input features - {name} has {X_train.shape[1]}. "
            f"Try 'orbit network {name}' instead."
        )
        return None

    X_test = Y_test = None
    if experiment.test_dataloader is not None:
        X_test, Y_test = _stack(experiment.test_dataloader.dataset)

    data = BoundaryData(X_train, Y_train, X_test, Y_test, task=config.get("task"), axis_labels=_axis_labels(config))
    all_X = X_train if X_test is None else np.vstack([X_train, X_test])
    xx, yy, grid = boundary_grid(all_X)
    model = experiment.model
    exp_dir = experiment_dir(name, root=root)

    if not animate:
        predictions = forward_trace(model, grid)[-1]
        title = f"{name} — Decision boundary"
        if not trained:
            title += " (untrained)"
            warning(f"{name} has not been run yet - drew its initial (untrained) predictions.")
        output_path = pathlib.Path(output) if output else exp_dir / "results" / "boundary.png"
        plot_boundary(xx, yy, predictions, data, title, output_path, show=show)
        success(f"Saved decision boundary to {output_path}")
        return output_path

    snapshots = snapshot_weights(name, network_structure(model), root)
    if snapshots is None:
        return None
    frames = []
    for epoch, weights in snapshots:
        set_weights(model, weights)
        frames.append((epoch, forward_trace(model, grid)[-1]))

    # For regression, one color scale across every frame (and the targets).
    values = np.concatenate([Y_train.ravel()] + [predictions[:, 0] for _, predictions in frames])
    value_range = (float(values.min()), float(values.max()))

    output_path = pathlib.Path(output) if output else exp_dir / "results" / f"boundary.{video_format}"
    info(f"Rendering {len(frames)} frames...")
    try:
        animate_boundary(
            xx, yy, frames, data, f"{name} — Decision boundary", output_path,
            value_range=value_range, video_format=video_format, fps=fps, show=show,
        )
    except VideoWriterError as e:
        warning(escape(str(e)))
        return None
    success(f"Saved decision boundary animation to {output_path}")
    return output_path
