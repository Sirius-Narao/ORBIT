import numpy as np
import pytest

from orbit.nn import Sequential
from orbit.nn.layers.linear import Linear
from orbit.nn.activations import ReLU, Sigmoid, Tanh
from orbit.nn.introspection import (
    column_sizes,
    forward_trace,
    network_structure,
    set_weights,
    weights_from_arrays,
    weights_of,
)


def fixed_model():
    """2 -> 2 (ReLU) -> 1 (Sigmoid) with hand-picked weights."""
    model = Sequential(Linear(2, 2), ReLU(), Linear(2, 1), Sigmoid())
    set_weights(model, [
        (np.array([[1.0, -1.0], [2.0, 1.0]]), np.array([0.0, 0.5])),
        (np.array([[1.0], [-2.0]]), np.array([0.25])),
    ])
    return model


def test_network_structure_groups_each_linear_with_its_activation():
    model = Sequential(Linear(2, 8), Tanh(), Linear(8, 3), Linear(3, 1), Sigmoid())

    structure = network_structure(model)

    assert [(l.key, l.in_features, l.out_features, l.activation) for l in structure] == [
        ("0", 2, 8, "Tanh"),
        ("2", 8, 3, None),
        ("3", 3, 1, "Sigmoid"),
    ]
    assert column_sizes(structure) == [2, 8, 3, 1]


def test_network_structure_raises_without_a_linear_layer():
    with pytest.raises(ValueError):
        network_structure(Sequential(ReLU()))


def test_forward_trace_records_post_activation_values_per_column():
    model = fixed_model()
    X = np.array([[1.0, 1.0]])

    trace = forward_trace(model, X)

    # Hidden pre-activation: [1*1 + 1*2 + 0, 1*-1 + 1*1 + 0.5] = [3, 0.5];
    # ReLU leaves both. Output pre-activation: 3*1 + 0.5*-2 + 0.25 = 2.25,
    # then Sigmoid.
    assert len(trace) == 3
    assert np.allclose(trace[0], [[1.0, 1.0]])
    assert np.allclose(trace[1], [[3.0, 0.5]])
    assert np.allclose(trace[2], [[1 / (1 + np.exp(-2.25))]])


def test_forward_trace_applies_relu_to_the_hidden_column():
    model = fixed_model()

    trace = forward_trace(model, np.array([[0.0, -1.0]]))

    # Pre-activation [-2, -0.5] -> ReLU -> [0, 0].
    assert np.allclose(trace[1], [[0.0, 0.0]])


def test_forward_trace_matches_the_model_output_for_a_batch():
    model = Sequential(Linear(3, 4), Tanh(), Linear(4, 2))
    X = np.random.randn(5, 3)

    trace = forward_trace(model, X)

    from orbit.core import Tensor
    assert np.allclose(trace[-1], model(Tensor(X)).data)
    assert [column.shape for column in trace] == [(5, 3), (5, 4), (5, 2)]


def test_weights_round_trip_through_named_parameter_arrays():
    model = fixed_model()
    structure = network_structure(model)
    arrays = {name: param.data for name, param in model.named_parameters()}

    from_arrays = weights_from_arrays(structure, arrays)

    for (W1, b1), (W2, b2) in zip(weights_of(model), from_arrays):
        assert np.array_equal(W1, W2)
        assert np.array_equal(b1, b2)


def test_weights_of_returns_copies():
    model = fixed_model()

    weights = weights_of(model)
    weights[0][0][0, 0] = 99.0

    assert model._modules["0"].weight.data[0, 0] == 1.0


def test_set_weights_rejects_a_mismatched_layer_count():
    with pytest.raises(ValueError):
        set_weights(fixed_model(), [(np.zeros((2, 2)), np.zeros(2))])
