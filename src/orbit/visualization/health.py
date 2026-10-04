"""
Activation health: how each hidden layer's units actually behave over a
dataset - the failure modes that let a run "finish" while learning little.

- Dead ReLU units: output 0 for every sample, so they pass no gradient and
  can never recover.
- Saturated Tanh/Sigmoid units: stuck on a flat part of the curve, where
  the gradient is ~0 (what unscaled weight init did to wide Tanh layers
  before Linear switched to Xavier init).
"""
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from orbit.nn.introspection import LayerInfo
from orbit.visualization.network import column_label

TANH_SATURATION = 0.97       # |tanh| above this: gradient 1 - a^2 < ~0.06
SIGMOID_SATURATION = 0.03    # sigmoid below this or above 1 - this: gradient a(1 - a) < ~0.03
# A layer is flagged when more than this fraction of it is dead/saturated.
UNHEALTHY_FRACTION = 0.5


@dataclass
class LayerHealth:
    label: str
    activation: Optional[str]
    units: int
    mean: float
    std: float
    dead_fraction: Optional[float]       # ReLU only: units that are 0 for every sample
    saturated_fraction: Optional[float]  # Tanh/Sigmoid only: activation values in the flat zone

    @property
    def problems(self) -> List[str]:
        found = []
        if self.dead_fraction is not None and self.dead_fraction > UNHEALTHY_FRACTION:
            found.append(f"{self.dead_fraction:.0%} of its ReLU units are dead")
        if self.saturated_fraction is not None and self.saturated_fraction > UNHEALTHY_FRACTION:
            found.append(f"{self.saturated_fraction:.0%} of its activations are saturated")
        return found


def saturated_mask(values: np.ndarray, activation: Optional[str]) -> Optional[np.ndarray]:
    if activation == "Tanh":
        return np.abs(values) > TANH_SATURATION
    if activation == "Sigmoid":
        return (values < SIGMOID_SATURATION) | (values > 1 - SIGMOID_SATURATION)
    return None


def layer_health(label: str, activation: Optional[str], values: np.ndarray) -> LayerHealth:
    """Stats for one column's activations, shape (n_samples, units)."""
    dead = float(np.mean(np.all(values <= 0, axis=0))) if activation == "ReLU" else None
    mask = saturated_mask(values, activation)
    saturated = float(np.mean(mask)) if mask is not None else None
    return LayerHealth(label, activation, values.shape[1], float(values.mean()), float(values.std()), dead, saturated)


def hidden_layer_health(structure: List[LayerInfo], trace: Sequence[np.ndarray]) -> List[LayerHealth]:
    """
    One LayerHealth per hidden column. The input isn't a layer, and the
    output layer's saturation is usually the point (a confident sigmoid
    output), so both are left out.
    """
    return [
        layer_health(column_label(structure, column), structure[column - 1].activation, trace[column])
        for column in range(1, len(structure))
    ]


def plot_activation_health(health: List[LayerHealth], trace: Sequence[np.ndarray], title: str,
                           output_path: pathlib.Path) -> pathlib.Path:
    """One histogram per hidden layer, with saturated zones shaded and the stats in each subplot's title."""
    if not health:
        raise ValueError("No hidden layers to plot.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = len(health)
    columns = min(count, 3)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.4 * rows), squeeze=False)

    for index, layer in enumerate(health):
        ax = axes[index // columns][index % columns]
        values = trace[index + 1].ravel()
        unhealthy = bool(layer.problems)
        ax.hist(values, bins=40, color="#ffb0b0" if unhealthy else "#b0ffb3", edgecolor="#555555", linewidth=0.4)

        if layer.activation == "Tanh":
            ax.axvspan(-1, -TANH_SATURATION, color="#d6b0ff", alpha=0.4)
            ax.axvspan(TANH_SATURATION, 1, color="#d6b0ff", alpha=0.4)
        elif layer.activation == "Sigmoid":
            ax.axvspan(0, SIGMOID_SATURATION, color="#d6b0ff", alpha=0.4)
            ax.axvspan(1 - SIGMOID_SATURATION, 1, color="#d6b0ff", alpha=0.4)

        details = [f"mean {layer.mean:.3g}, std {layer.std:.3g}"]
        if layer.dead_fraction is not None:
            details.append(f"dead units {layer.dead_fraction:.0%}")
        if layer.saturated_fraction is not None:
            details.append(f"saturated {layer.saturated_fraction:.0%}")
        ax.set_title(f"{layer.label}\n" + " · ".join(details), fontsize=9)
        ax.set_xlabel("Activation")
        ax.set_ylabel("Count")

    for index in range(count, rows * columns):
        axes[index // columns][index % columns].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return output_path
