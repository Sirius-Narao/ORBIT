import numpy as np

from orbit.nn import Sequential
from orbit.nn.layers.linear import Linear
from orbit.nn.activations import ReLU, Tanh
from orbit.nn.introspection import forward_trace, network_structure, weights_of
from orbit.visualization.network import (
    MAX_NODES_PER_COLUMN,
    column_label,
    plot_network,
    rank_nodes,
    visible_nodes,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_visible_nodes_shows_every_node_of_a_narrow_column():
    assert visible_nodes(5) == ([0, 1, 2, 3, 4], 0)


def test_visible_nodes_shows_first_and_last_nodes_of_a_wide_column():
    shown, hidden = visible_nodes(100)

    assert shown == list(range(8)) + list(range(93, 100))
    assert hidden == 85


def test_visible_nodes_shows_the_top_ranked_nodes_in_index_order():
    ranked = list(range(99, -1, -1))  # highest index = most important

    shown, hidden = visible_nodes(100, ranked)

    assert shown == list(range(100 - (MAX_NODES_PER_COLUMN - 1), 100))
    assert hidden == 100 - (MAX_NODES_PER_COLUMN - 1)


def test_rank_nodes_orders_by_mean_absolute_activation():
    trace = [np.array([[0.1, -3.0, 1.0], [0.1, -1.0, 1.0]])]

    # Mean |a|: [0.1, 2.0, 1.0].
    assert rank_nodes(trace) == [[1, 2, 0]]


def test_column_label_names_input_hidden_and_output_columns():
    structure = network_structure(Sequential(Linear(2, 8), Tanh(), Linear(8, 1)))

    assert column_label(structure, 0) == "Input (2)"
    assert column_label(structure, 1) == "Linear 8 · Tanh"
    assert column_label(structure, 2) == "Output 1"


def _render(model, X, tmp_path, **kwargs):
    structure = network_structure(model)
    trace = forward_trace(model, X)
    output_path = tmp_path / "network.png"
    returned = plot_network(
        structure, weights_of(model), [column.mean(axis=0) for column in trace], "test", output_path,
        value_ranges=[(column.min(), column.max()) for column in trace], **kwargs,
    )
    return returned, output_path


def test_plot_network_saves_a_png(tmp_path):
    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1))

    returned, output_path = _render(model, np.random.randn(4, 2), tmp_path, subtitle="sample 0")

    assert returned == output_path
    assert output_path.read_bytes()[:8] == PNG_MAGIC


def test_plot_network_handles_wide_columns(tmp_path):
    model = Sequential(Linear(30, 64), ReLU(), Linear(64, 2))
    X = np.random.randn(10, 30)

    _, output_path = _render(model, X, tmp_path, ranked_nodes=rank_nodes(forward_trace(model, X)))

    assert output_path.read_bytes()[:8] == PNG_MAGIC


def test_plot_network_handles_a_constant_column(tmp_path):
    # All-zero weights: every hidden unit is identical, so the column's
    # range is a single value - must not divide by zero.
    model = Sequential(Linear(2, 3), ReLU(), Linear(3, 1))
    for param in model.parameters():
        param.data = np.zeros_like(param.data)

    _, output_path = _render(model, np.ones((2, 2)), tmp_path)

    assert output_path.read_bytes()[:8] == PNG_MAGIC
