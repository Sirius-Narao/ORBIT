"""
Network diagrams: the model drawn as columns of nodes joined by its actual
weights.

- Edges: color = weight sign (RdBu_r - red positive, blue negative),
  thickness/opacity = |w| relative to the largest |w| in that layer.
- Nodes: fill = activation value (magma), normalized per column, since a
  ReLU column's values and a Sigmoid column's live on different scales.
  Outline = the unit's bias, on the same diverging scale as the edges.

draw_network() only draws onto an Axes, so animations can redraw it frame
by frame; plot_network() is the save-a-PNG wrapper.
"""
import pathlib
from typing import List, Optional, Sequence, Tuple

import numpy as np

from orbit.nn.introspection import LayerInfo, column_sizes
from orbit.visualization.backend import can_show, pyplot

# A column wider than this shows its first and last few nodes plus a
# "⋮ +k" marker - otherwise e.g. a 784-pixel input column would be an
# unreadable smear, and drawing every one of its edges would be slow.
MAX_NODES_PER_COLUMN = 16
HEAD_NODES = 8
TAIL_NODES = 7
# Print each node's value inside it only when the column is short enough
# for the text to fit.
MAX_LABELED_NODES = 12

COLUMN_SPACING = 3.0
NODE_RADIUS = 0.32
WEIGHT_CMAP = "RdBu_r"
ACTIVATION_CMAP = "magma"


def visible_nodes(size: int, ranked: Optional[Sequence[int]] = None) -> Tuple[List[int], int]:
    """
    (indices of the nodes drawn, in index order; number hidden).

    ranked: the column's node indices, most important first (see
    rank_nodes). When given, a wide column shows its top
    MAX_NODES_PER_COLUMN - 1 nodes; without it, its first HEAD_NODES and
    last TAIL_NODES. Either way one slot is left for the "⋮ +k" marker.
    """
    if size <= MAX_NODES_PER_COLUMN:
        return list(range(size)), 0
    if ranked is not None:
        shown = sorted(int(i) for i in list(ranked)[: MAX_NODES_PER_COLUMN - 1])
    else:
        shown = list(range(HEAD_NODES)) + list(range(size - TAIL_NODES, size))
    return shown, size - len(shown)


def rank_nodes(trace: Sequence[np.ndarray]) -> List[List[int]]:
    """
    Per column, node indices ordered by mean |activation| over the samples
    in a forward_trace(), most active first - what a wide column shows
    instead of an arbitrary first/last few, since in a 128-unit ReLU layer
    most units are often silent for any given input.
    """
    return [list(np.argsort(-np.abs(column).mean(axis=0), kind="stable")) for column in trace]


def _node_positions(
    size: int, column: int, ranked: Optional[Sequence[int]] = None
) -> Tuple[dict, Optional[Tuple[float, float]]]:
    """
    {node index: (x, y)} for one column's visible nodes, centered on y = 0
    and listed top-down, plus where the "⋮" marker goes (None if nothing is
    hidden): between head and tail for a first/last column, at the bottom
    for a ranked one, whose hidden nodes aren't contiguous.
    """
    shown, hidden = visible_nodes(size, ranked)
    slots = len(shown) + (1 if hidden else 0)
    x = column * COLUMN_SPACING
    top = (slots - 1) / 2
    marker_slot = None
    if hidden:
        marker_slot = len(shown) if ranked is not None else HEAD_NODES

    positions, marker = {}, None
    slot = 0
    for i, node in enumerate(shown):
        if slot == marker_slot:
            slot += 1
        positions[node] = (x, top - slot)
        slot += 1
    if marker_slot is not None:
        marker = (x, top - marker_slot)
    return positions, marker


def column_label(structure: List[LayerInfo], column: int) -> str:
    if column == 0:
        return f"Input ({structure[0].in_features})"
    layer = structure[column - 1]
    name = "Output" if column == len(structure) else "Linear"
    label = f"{name} {layer.out_features}"
    if layer.activation:
        label += f" · {layer.activation}"
    return label


def _normalize(values: np.ndarray, value_range: Optional[Tuple[float, float]]) -> np.ndarray:
    low, high = value_range if value_range is not None else (float(np.min(values)), float(np.max(values)))
    if high - low < 1e-12:
        return np.full(values.shape, 0.5)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def draw_network(
    ax,
    structure: List[LayerInfo],
    weights: Sequence[Tuple[np.ndarray, np.ndarray]],
    activations: Sequence[np.ndarray],
    value_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    weight_scales: Optional[Sequence[float]] = None,
    ranked_nodes: Optional[Sequence[Sequence[int]]] = None,
) -> None:
    """
    Draw the network onto ax.

    activations: one 1D array per column (column 0 = the input), e.g. one
    sample's forward_trace() row or the mean over a dataset.
    value_ranges: optional (low, high) per column to normalize node colors
    against - pass the range over the whole dataset (or, in an animation,
    over every frame) so colors stay comparable; default is each column's
    own min/max.
    weight_scales: optional max |w| per layer for edge/bias colors - same
    idea, fixed across animation frames; default is each layer's own max.
    ranked_nodes: optional per-column importance order (rank_nodes) that
    picks which nodes a wide column shows; default is its first/last few.
    """
    import matplotlib
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    weight_cmap = matplotlib.colormaps[WEIGHT_CMAP]
    activation_cmap = matplotlib.colormaps[ACTIVATION_CMAP]

    sizes = column_sizes(structure)
    layout = [
        _node_positions(size, column, ranked_nodes[column] if ranked_nodes is not None else None)
        for column, size in enumerate(sizes)
    ]

    # --- edges ---------------------------------------------------------
    for layer_index, (W, _) in enumerate(weights):
        scale = weight_scales[layer_index] if weight_scales is not None else float(np.max(np.abs(W)))
        scale = max(scale, 1e-12)
        sources, _ = layout[layer_index]
        targets, _ = layout[layer_index + 1]

        segments, colors, widths = [], [], []
        for i, (x0, y0) in sources.items():
            for j, (x1, y1) in targets.items():
                w = float(W[i, j])
                strength = min(abs(w) / scale, 1.0)
                color = list(weight_cmap(0.5 + 0.5 * np.clip(w / scale, -1.0, 1.0)))
                color[3] = 0.15 + 0.85 * strength
                segments.append([(x0 + NODE_RADIUS, y0), (x1 - NODE_RADIUS, y1)])
                colors.append(color)
                widths.append(0.3 + 2.7 * strength)
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, zorder=1))

    # --- nodes ---------------------------------------------------------
    for column, (positions, marker) in enumerate(layout):
        values = np.asarray(activations[column], dtype=float)
        value_range = value_ranges[column] if value_ranges is not None else None
        normalized = _normalize(values, value_range)
        label_values = len(positions) <= MAX_LABELED_NODES

        bias, bias_scale = None, 1.0
        if column > 0:
            W, bias = weights[column - 1]
            bias_scale = weight_scales[column - 1] if weight_scales is not None else float(np.max(np.abs(W)))
            bias_scale = max(bias_scale, 1e-12)

        for node, (x, y) in positions.items():
            fill = activation_cmap(normalized[node])
            if bias is None:
                outline = "#888888"
            else:
                outline = weight_cmap(0.5 + 0.5 * np.clip(bias[node] / bias_scale, -1.0, 1.0))
            ax.add_patch(Circle((x, y), NODE_RADIUS, facecolor=fill, edgecolor=outline, linewidth=2.0, zorder=2))
            if label_values:
                text_color = "white" if normalized[node] < 0.6 else "black"
                ax.text(x, y, f"{values[node]:.2f}", ha="center", va="center", fontsize=7, color=text_color, zorder=3)

        if marker is not None:
            hidden = sizes[column] - len(positions)
            ax.text(marker[0], marker[1], f"⋮\n+{hidden}", ha="center", va="center", fontsize=9, color="#555555")

    # --- framing -------------------------------------------------------
    tallest = max(len(p) + (1 if m else 0) for p, m in layout)
    bottom = -(tallest - 1) / 2 - 1.0
    for column in range(len(sizes)):
        ax.text(column * COLUMN_SPACING, bottom, column_label(structure, column), ha="center", va="top", fontsize=9)

    ax.set_xlim(-1.2, (len(sizes) - 1) * COLUMN_SPACING + 1.2)
    ax.set_ylim(bottom - 0.8, (tallest - 1) / 2 + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def network_figure_size(structure: List[LayerInfo]) -> Tuple[float, float]:
    sizes = column_sizes(structure)
    tallest = max(min(size, MAX_NODES_PER_COLUMN) for size in sizes)
    return max(7.0, 2.4 * len(sizes) + 2.0), max(4.5, 0.6 * tallest + 2.0)


def add_network_colorbars(fig, ax, caxes=None) -> None:
    """
    Legends for the two color scales: activation (nodes) and weight
    sign/size (edges, outlines). By default two short horizontal bars side
    by side under the diagram, placed in ax's coordinates so they follow
    the (aspect-locked) network wherever it ends up in the figure. An
    animation, which clears ax every frame, passes its own figure-level
    caxes=(activation_ax, weight_ax) instead.
    """
    import matplotlib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if caxes is None:
        caxes = (ax.inset_axes([0.08, -0.06, 0.36, 0.025]), ax.inset_axes([0.56, -0.06, 0.36, 0.025]))
    activation_cax, weight_cax = caxes

    activation_bar = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=matplotlib.colormaps[ACTIVATION_CMAP]),
        cax=activation_cax, orientation="horizontal", ticks=[0, 1],
    )
    activation_bar.ax.set_xticklabels(["low", "high"], fontsize=8)
    activation_bar.ax.set_title("Node: activation (per layer)", fontsize=8)

    weight_bar = fig.colorbar(
        ScalarMappable(norm=Normalize(-1, 1), cmap=matplotlib.colormaps[WEIGHT_CMAP]),
        cax=weight_cax, orientation="horizontal", ticks=[-1, 0, 1],
    )
    weight_bar.ax.set_xticklabels(["−", "0", "+"], fontsize=8)
    weight_bar.ax.set_title("Link: weight · outline: bias", fontsize=8)


def plot_network(
    structure: List[LayerInfo],
    weights: Sequence[Tuple[np.ndarray, np.ndarray]],
    activations: Sequence[np.ndarray],
    title: str,
    output_path: pathlib.Path,
    value_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    subtitle: Optional[str] = None,
    ranked_nodes: Optional[Sequence[Sequence[int]]] = None,
    show: bool = False,
) -> pathlib.Path:
    """
    Save a network diagram as a PNG at output_path (and, with show=True,
    also open it in a window - see backend.pyplot).
    """
    plt = pyplot(show)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=network_figure_size(structure))
    draw_network(ax, structure, weights, activations, value_ranges=value_ranges, ranked_nodes=ranked_nodes)
    add_network_colorbars(fig, ax)
    ax.set_title(title if subtitle is None else f"{title}\n{subtitle}", fontsize=11)

    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    if show and can_show(plt):
        plt.show()
    plt.close(fig)

    return output_path
