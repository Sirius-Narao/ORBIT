import json
import pathlib
from typing import Optional

import numpy as np

from orbit.core.config import load_experiment
from orbit.nn.introspection import forward_trace, network_structure, weights_of
from orbit.storage import EXPERIMENTS_ROOT, checkpoint_exists, experiment_dir, load_checkpoint
from orbit.ui import success, warning
from orbit.visualization.network import plot_network, rank_nodes


def _format_vector(values: np.ndarray, limit: int = 6) -> str:
    values = np.asarray(values, dtype=float).ravel()
    shown = ", ".join(f"{v:.3g}" for v in values[:limit])
    if len(values) > limit:
        shown += f", … (+{len(values) - limit})"
    return f"[{shown}]"


def training_arrays(experiment):
    """Every training sample's (X, Y), stacked - after the split and normalization load_experiment applied."""
    dataset = experiment.dataloader.dataset
    samples = [dataset[i] for i in range(len(dataset))]
    X = np.stack([np.asarray(x, dtype=float) for x, _ in samples])
    Y = np.stack([np.asarray(y, dtype=float) for _, y in samples])
    return X, Y


def load_trained_experiment(name: str, root: pathlib.Path):
    """
    (config, experiment, trained) for a saved experiment, with its
    checkpoint loaded if one exists; None if the experiment doesn't exist.
    Without a checkpoint the model keeps the initial weights load_experiment
    built (the seeded ones, if the config has a seed), so an experiment can
    be looked at before it's ever been run.
    """
    config_path = experiment_dir(name, root=root) / "experiment.json"
    if not config_path.exists():
        warning(f"{name} was not found.")
        return None

    with open(config_path) as f:
        config = json.load(f)

    experiment = load_experiment(config)
    trained = checkpoint_exists(name, root=root)
    if trained:
        load_checkpoint(name, experiment.model, root=root)
    return config, experiment, trained


def network_experiment(
    name: str,
    sample: Optional[int] = None,
    output: Optional[str] = None,
    show: bool = False,
    root: pathlib.Path = EXPERIMENTS_ROOT,
) -> Optional[pathlib.Path]:
    """
    Draw an experiment's network as a PNG. Node colors are the mean
    activation over the training set, or one training sample's activations
    with sample=N - either way normalized against each layer's range over
    the whole training set, so a "bright" node really is firing strongly
    for this model.
    """
    loaded = load_trained_experiment(name, root)
    if loaded is None:
        return None
    config, experiment, trained = loaded

    X, Y = training_arrays(experiment)
    if sample is not None and not 0 <= sample < len(X):
        warning(f"Sample {sample} is out of range - the training set has {len(X)} samples (0-{len(X) - 1}).")
        return None

    model = experiment.model
    structure = network_structure(model)
    trace = forward_trace(model, X)
    value_ranges = [(float(column.min()), float(column.max())) for column in trace]

    if sample is None:
        activations = [column.mean(axis=0) for column in trace]
        subtitle = f"Mean activation over {len(X)} training samples"
    else:
        activations = [column[sample] for column in trace]
        normalized = " (normalized)" if config.get("normalize", "none") != "none" else ""
        subtitle = (
            f"Sample {sample}: input{normalized} {_format_vector(X[sample])} → "
            f"prediction {_format_vector(trace[-1][sample])}, target {_format_vector(Y[sample])}"
        )

    title = f"{name} — Network"
    if not trained:
        title += " (untrained — initial weights)"

    exp_dir = experiment_dir(name, root=root)
    filename = "network.png" if sample is None else f"network_sample{sample}.png"
    output_path = pathlib.Path(output) if output else exp_dir / "results" / filename

    plot_network(
        structure, weights_of(model), activations, title, output_path,
        value_ranges=value_ranges, subtitle=subtitle, ranked_nodes=rank_nodes(trace), show=show,
    )

    if not trained:
        warning(f"{name} has not been run yet - drew its initial (untrained) weights.")
    success(f"Saved network diagram to {output_path}")
    return output_path
