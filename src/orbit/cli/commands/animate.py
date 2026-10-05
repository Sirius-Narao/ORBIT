import pathlib
from typing import List, Optional, Tuple

import numpy as np
from rich.markup import escape

from orbit.cli.commands.network import load_trained_experiment, training_arrays
from orbit.nn.introspection import forward_trace, network_structure, set_weights, weights_from_arrays
from orbit.storage import experiment_dir, load_results, load_snapshots, snapshots_exist
from orbit.ui import info, success, warning
from orbit.visualization.animation import DEFAULT_FPS, Frame, VideoWriterError, animate_training
from orbit.visualization.network import rank_nodes
from orbit.storage.workspace import experiments_root


def snapshot_weights(name: str, structure, root: pathlib.Path) -> Optional[List[Tuple[int, list]]]:
    """
    [(epoch, [(W, b), ...]), ...] for every recorded snapshot, or None (with
    a warning) when there's no history, or one that no longer fits the
    experiment's architecture (experiment.json edited after the run).
    """
    if not snapshots_exist(name, root=root):
        warning(
            f"{name} has no recorded training history - run it with 'orbit run {name}' "
            f"(or 'orbit train {name}') to record one."
        )
        return None

    history = load_snapshots(name, root=root)
    try:
        return [
            (int(epoch), weights_from_arrays(structure, {key: values[i] for key, values in history["params"].items()}))
            for i, epoch in enumerate(history["epochs"])
        ]
    except KeyError:
        warning(f"{name}'s training history doesn't match its current model - re-run it to record a new one.")
        return None


def full_loss_history(name: str, root: pathlib.Path, fallback: np.ndarray) -> List[float]:
    """The per-epoch loss from results.json; the snapshots' own (sparser) losses if it's missing."""
    results_path = experiment_dir(name, root=root) / "results" / "results.json"
    if results_path.exists():
        return list(load_results(results_path).loss_history)
    return [float(loss) for loss in fallback if not np.isnan(loss)]


def animate_experiment(
    name: str,
    sample: Optional[int] = None,
    video_format: str = "gif",
    fps: int = DEFAULT_FPS,
    output: Optional[str] = None,
    log_scale: bool = False,
    show: bool = False,
    root: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    """
    Animate an experiment's training from its recorded weight snapshots.
    Nodes show the mean activation over the training set (or training
    sample N's), recomputed at every snapshot.
    """
    root = experiments_root(root)
    loaded = load_trained_experiment(name, root)
    if loaded is None:
        return None
    _, experiment, _ = loaded

    model = experiment.model
    structure = network_structure(model)
    snapshots = snapshot_weights(name, structure, root)
    if snapshots is None:
        return None

    X, _ = training_arrays(experiment)
    if sample is not None and not 0 <= sample < len(X):
        warning(f"Sample {sample} is out of range - the training set has {len(X)} samples (0-{len(X) - 1}).")
        return None

    frames, traces = [], []
    for epoch, weights in snapshots:
        set_weights(model, weights)
        trace = forward_trace(model, X)
        traces.append(trace)
        activations = [column.mean(axis=0) if sample is None else column[sample] for column in trace]
        frames.append(Frame(epoch, weights, activations))

    # Fixed across every frame, so a color change means the model changed.
    columns = range(len(traces[0]))
    value_ranges = [
        (min(float(t[c].min()) for t in traces), max(float(t[c].max()) for t in traces)) for c in columns
    ]
    weight_scales = [
        max(float(np.max(np.abs(weights[layer][0]))) for _, weights in snapshots)
        for layer in range(len(structure))
    ]

    history = load_snapshots(name, root=root)
    loss_history = full_loss_history(name, root, history["loss"])

    title = f"{name} — Training"
    if sample is not None:
        title += f" (sample {sample})"
    stem = "training" if sample is None else f"training_sample{sample}"
    output_path = (
        pathlib.Path(output) if output
        else experiment_dir(name, root=root) / "results" / f"{stem}.{video_format}"
    )

    info(f"Rendering {len(frames)} frames...")
    try:
        animate_training(
            structure, frames, loss_history, title, output_path,
            value_ranges=value_ranges, weight_scales=weight_scales,
            ranked_nodes=rank_nodes(traces[-1]),
            video_format=video_format, fps=fps, log_scale=log_scale, show=show,
        )
    except VideoWriterError as e:
        warning(escape(str(e)))  # escaped: the hint's ".[video]" isn't rich markup
        return None

    success(f"Saved training animation to {output_path}")
    return output_path
