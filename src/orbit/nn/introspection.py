"""
Read-only views into a Sequential model's structure, weights and per-layer
activations - what the visualization code (network diagrams, animations,
activation health, the live terminal view) draws from.

A model is seen as columns of nodes: the input features, then one column per
Linear layer (its output units). Any activations directly after a Linear are
applied to that Linear's column, so "Linear(2, 8), Tanh" is one column of 8
Tanh units, not two columns.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from orbit.core import Tensor
from orbit.nn import Sequential
from orbit.nn.layers.linear import Linear


@dataclass
class LayerInfo:
    key: str                  # the Linear's module key in the Sequential, e.g. "0"
    in_features: int
    out_features: int
    activation: Optional[str]  # e.g. "Tanh"; None when the Linear has none after it


def network_structure(model: Sequential) -> List[LayerInfo]:
    """
    One LayerInfo per Linear, in order, each with the activation(s) that
    follow it before the next Linear (joined with "+" in the rare case of
    several, e.g. "ReLU+Sigmoid").
    """
    structure: List[LayerInfo] = []
    pending_activations: List[str] = []

    def close_layer():
        if structure:
            structure[-1].activation = "+".join(pending_activations) or None
        pending_activations.clear()

    for key, layer in model.named_children():
        if isinstance(layer, Linear):
            close_layer()
            in_features, out_features = layer.weight.data.shape
            structure.append(LayerInfo(key, in_features, out_features, None))
        elif structure:
            pending_activations.append(type(layer).__name__)
    close_layer()

    if not structure:
        raise ValueError("Model has no Linear layer to visualize.")
    return structure


def column_sizes(structure: List[LayerInfo]) -> List[int]:
    """Node count of every column: the input, then each Linear's output."""
    return [structure[0].in_features] + [layer.out_features for layer in structure]


def forward_trace(model: Sequential, X: np.ndarray) -> List[np.ndarray]:
    """
    Forward a batch X (n_samples, n_features) through the model one module
    at a time, recording each column's post-activation values: element 0 is
    X itself, element k is the output of the k-th Linear after its
    activation(s), each of shape (n_samples, column_width). Forward only -
    nothing is backpropagated, so the model's grads are left untouched.
    """
    X = np.asarray(X, dtype=float)
    columns = [X]
    layers = list(model.children())
    x = Tensor(X)
    seen_linear = False

    for i, layer in enumerate(layers):
        x = layer(x)
        seen_linear = seen_linear or isinstance(layer, Linear)
        next_is_linear = i + 1 < len(layers) and isinstance(layers[i + 1], Linear)
        if seen_linear and (next_is_linear or i + 1 == len(layers)):
            columns.append(np.array(x.data, dtype=float))

    return columns


def weights_of(model: Sequential) -> List[Tuple[np.ndarray, np.ndarray]]:
    """A copy of every Linear's (weight, bias), in order."""
    return [
        (layer.weight.data.copy(), layer.bias.data.copy())
        for layer in model.children()
        if isinstance(layer, Linear)
    ]


def weights_from_arrays(
    structure: List[LayerInfo], arrays: Dict[str, np.ndarray]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    (weight, bias) per Linear from a named_parameters()-keyed mapping, e.g. a
    loaded checkpoint ("0.weight", "0.bias", ...).
    """
    return [(arrays[f"{layer.key}.weight"], arrays[f"{layer.key}.bias"]) for layer in structure]


def set_weights(model: Sequential, weights: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Overwrite every Linear's (weight, bias) in place, e.g. with a snapshot's."""
    linears = [layer for layer in model.children() if isinstance(layer, Linear)]
    if len(linears) != len(weights):
        raise ValueError(f"Model has {len(linears)} Linear layers but {len(weights)} weight pairs were given.")
    for layer, (W, b) in zip(linears, weights):
        layer.weight.data = np.array(W, dtype=float)
        layer.bias.data = np.array(b, dtype=float)
