import numpy as np
import pytest

from orbit.nn import Sequential
from orbit.nn.layers.linear import Linear
from orbit.nn.activations import ReLU, Sigmoid, Tanh
from orbit.nn.introspection import forward_trace, network_structure
from orbit.visualization.health import hidden_layer_health, layer_health, plot_activation_health


def test_layer_health_counts_relu_units_dead_for_every_sample():
    # Unit 0 fires once, unit 1 never, unit 2 always: 1 of 3 dead.
    values = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 3.0]])

    health = layer_health("h", "ReLU", values)

    assert np.isclose(health.dead_fraction, 1 / 3)
    assert health.saturated_fraction is None
    assert health.problems == []


def test_layer_health_counts_saturated_tanh_values():
    values = np.array([[0.99, -0.995], [0.1, 0.98]])  # 3 of 4 beyond |0.97|

    health = layer_health("h", "Tanh", values)

    assert np.isclose(health.saturated_fraction, 0.75)
    assert health.dead_fraction is None
    assert health.problems == ["75% of its activations are saturated"]


def test_layer_health_counts_saturated_sigmoid_values_at_both_ends():
    values = np.array([[0.01, 0.5], [0.99, 0.5]])

    assert np.isclose(layer_health("h", "Sigmoid", values).saturated_fraction, 0.5)


def test_layer_health_flags_a_mostly_dead_relu_layer():
    values = np.array([[0.0, 0.0, 0.0, 1.0]])

    assert layer_health("h", "ReLU", values).problems == ["75% of its ReLU units are dead"]


def test_hidden_layer_health_skips_input_and_output_columns():
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 3), Tanh(), Linear(3, 1), Sigmoid())
    structure = network_structure(model)

    health = hidden_layer_health(structure, forward_trace(model, np.random.randn(6, 2)))

    assert [(h.label, h.activation, h.units) for h in health] == [
        ("Linear 4 · ReLU", "ReLU", 4),
        ("Linear 3 · Tanh", "Tanh", 3),
    ]


def test_plot_activation_health_saves_a_png(tmp_path):
    model = Sequential(Linear(2, 4), ReLU(), Linear(4, 3), Tanh(), Linear(3, 1))
    trace = forward_trace(model, np.random.randn(6, 2))
    health = hidden_layer_health(network_structure(model), trace)

    output = plot_activation_health(health, trace, "t", tmp_path / "health.png")

    assert output.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_activation_health_raises_without_hidden_layers(tmp_path):
    with pytest.raises(ValueError):
        plot_activation_health([], [], "t", tmp_path / "health.png")
