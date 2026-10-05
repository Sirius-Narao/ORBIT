import copy

from orbit.sweeps.generator import expand_grid

BASE = {
    "name": "base_exp",
    "dataset": "xor",
    "model": [{"type": "Linear", "in_features": 2, "neurons": 1}, {"type": "Sigmoid"}],
    "loss": "MSE",
    "optimizer": "SGD",
    "learning_rate": 1.0,
    "epochs": 5,
    "seed": 1,
}


def test_expand_grid_is_the_full_product_in_key_order():
    runs = expand_grid("s", BASE, {"learning_rate": [0.1, 0.5], "seed": [1, 2, 3]})

    assert len(runs) == 6
    assert [params for _, params, _ in runs] == [
        {"learning_rate": 0.1, "seed": 1},
        {"learning_rate": 0.1, "seed": 2},
        {"learning_rate": 0.1, "seed": 3},
        {"learning_rate": 0.5, "seed": 1},
        {"learning_rate": 0.5, "seed": 2},
        {"learning_rate": 0.5, "seed": 3},
    ]


def test_expand_grid_names_runs_with_zero_padded_index():
    runs = expand_grid("lr", BASE, {"learning_rate": [0.1, 0.2]})

    assert [name for name, _, _ in runs] == ["lr_001", "lr_002"]
    assert [config["name"] for _, _, config in runs] == ["lr_001", "lr_002"]


def test_expand_grid_pads_wider_past_999_runs():
    runs = expand_grid("big", BASE, {"seed": list(range(1000))})

    assert runs[0][0] == "big_0001"
    assert runs[-1][0] == "big_1000"


def test_expand_grid_applies_params_over_a_copy_of_the_base():
    base = copy.deepcopy(BASE)
    runs = expand_grid("s", base, {"learning_rate": [0.3]})
    _, _, config = runs[0]

    assert config["learning_rate"] == 0.3
    assert config["dataset"] == "xor"
    assert config["model"] == BASE["model"]
    # the base itself is untouched, including nested objects
    assert base == BASE
    config["model"].append({"type": "Tanh"})
    assert base["model"] == BASE["model"]


def test_expand_grid_drops_momentum_for_adam_and_dedupes():
    """
    SGD/Adam x momentum [0, 0.9] is 4 raw combinations, but momentum has no
    meaning for Adam: both Adam runs collapse to the same config, giving 3.
    """
    runs = expand_grid("s", BASE, {"optimizer": ["SGD", "Adam"], "momentum": [0, 0.9]})

    assert [params for _, params, _ in runs] == [
        {"optimizer": "SGD", "momentum": 0},
        {"optimizer": "SGD", "momentum": 0.9},
        {"optimizer": "Adam"},
    ]
    adam_config = runs[2][2]
    assert "momentum" not in adam_config


def test_expand_grid_drops_base_momentum_when_switching_to_adam():
    base = dict(BASE, momentum=0.9)
    runs = expand_grid("s", base, {"optimizer": ["SGD", "Adam"]})

    assert runs[0][2]["momentum"] == 0.9
    assert "momentum" not in runs[1][2]


def test_expand_grid_normalize_none_removes_the_key():
    base = dict(BASE, normalize="standard")
    runs = expand_grid("s", base, {"normalize": ["none", "minmax"]})

    assert "normalize" not in runs[0][2]
    assert runs[0][1] == {}
    assert runs[1][2]["normalize"] == "minmax"


# --- architecture fields -------------------------------------------------------------

import numpy as np
import pytest

from orbit.core.config import build_model
from orbit.core.dataset import TensorDataset
from orbit.sweeps.generator import apply_architecture, describe_architecture

XOR_MODEL = [
    {"type": "Linear", "in_features": 2, "neurons": 8},
    {"type": "Tanh"},
    {"type": "Linear", "neurons": 1},
    {"type": "Sigmoid"},
]
# Same shape as the real WineS base: 11 -> 128 -> ReLU -> 128 -> ReLU -> 1
WINE_MODEL = [
    {"type": "Linear", "in_features": 11, "neurons": 128},
    {"type": "ReLU"},
    {"type": "Linear", "neurons": 128},
    {"type": "ReLU"},
    {"type": "Linear", "neurons": 1},
]
TAPERED_MODEL = [
    {"type": "Linear", "in_features": 11, "neurons": 128},
    {"type": "ReLU"},
    {"type": "Linear", "neurons": 64},
    {"type": "ReLU"},
    {"type": "Linear", "neurons": 1},
]
LINEAR_MODEL = [{"type": "Linear", "in_features": 11, "neurons": 1}]


def _dataset(n_in, n_out):
    return TensorDataset(np.zeros((4, n_in)), np.zeros((4, n_out)))


def test_describe_architecture():
    assert describe_architecture(XOR_MODEL) == {"hidden_width": 8, "hidden_depth": 1, "activation": "Tanh"}
    assert describe_architecture(WINE_MODEL) == {"hidden_width": 128, "hidden_depth": 2, "activation": "ReLU"}
    assert describe_architecture(TAPERED_MODEL)["hidden_width"] == 128  # first hidden width
    assert describe_architecture(LINEAR_MODEL) == {"hidden_width": None, "hidden_depth": 0, "activation": None}


def test_apply_architecture_keeps_the_output_head():
    """XOR, width 16 x depth 2: the Linear(->1) + Sigmoid head is untouched."""
    model = apply_architecture(XOR_MODEL, hidden_width=16, hidden_depth=2)

    assert model == [
        {"type": "Linear", "in_features": 2, "neurons": 16},
        {"type": "Tanh"},
        {"type": "Linear", "neurons": 16},
        {"type": "Tanh"},
        {"type": "Linear", "neurons": 1},
        {"type": "Sigmoid"},
    ]
    build_model(model, _dataset(2, 1))


def test_apply_architecture_activation_only_keeps_widths_and_depth():
    model = apply_architecture(WINE_MODEL, activation="Tanh")

    assert model == [
        {"type": "Linear", "in_features": 11, "neurons": 128},
        {"type": "Tanh"},
        {"type": "Linear", "neurons": 128},
        {"type": "Tanh"},
        {"type": "Linear", "neurons": 1},
    ]


def test_apply_architecture_depth_only_reuses_the_base_width():
    model = apply_architecture(WINE_MODEL, hidden_depth=1)

    assert model == [
        {"type": "Linear", "in_features": 11, "neurons": 128},
        {"type": "ReLU"},
        {"type": "Linear", "neurons": 1},
    ]
    build_model(model, _dataset(11, 1))


def test_apply_architecture_depth_zero_is_a_linear_model():
    """No hidden layers: in_features moves onto the head's Linear."""
    model = apply_architecture(WINE_MODEL, hidden_depth=0)

    assert model == [{"type": "Linear", "in_features": 11, "neurons": 1}]
    build_model(model, _dataset(11, 1))


def test_apply_architecture_tapered_base_uses_its_first_width():
    model = apply_architecture(TAPERED_MODEL, hidden_depth=3)

    assert [layer["neurons"] for layer in model if layer["type"] == "Linear"] == [128, 128, 128, 1]


def test_apply_architecture_does_not_mutate_the_base():
    before = [dict(layer) for layer in WINE_MODEL]

    apply_architecture(WINE_MODEL, hidden_width=4, hidden_depth=0)

    assert WINE_MODEL == before


def test_apply_architecture_needs_a_width_when_the_base_has_no_hidden_layer():
    with pytest.raises(ValueError, match="hidden_width"):
        apply_architecture(LINEAR_MODEL, hidden_depth=2)

    model = apply_architecture(LINEAR_MODEL, hidden_width=4, hidden_depth=1, activation="ReLU")
    assert model == [
        {"type": "Linear", "in_features": 11, "neurons": 4},
        {"type": "ReLU"},
        {"type": "Linear", "neurons": 1},
    ]


def test_expand_grid_architecture_rewrites_the_model_not_the_config_keys():
    base = dict(BASE, model=XOR_MODEL)

    runs = expand_grid("a", base, {"hidden_width": [4], "hidden_depth": [2], "activation": ["ReLU"]})
    _, params, config = runs[0]

    assert params == {"hidden_width": 4, "hidden_depth": 2, "activation": "ReLU"}
    for key in ("hidden_width", "hidden_depth", "activation"):
        assert key not in config
    assert config["model"] == apply_architecture(XOR_MODEL, 4, 2, "ReLU")


def test_expand_grid_depth_zero_collapses_width_and_activation():
    """depth [0, 1] x width [8, 16] x activation [Tanh, ReLU]: all four depth-0 combos are one model."""
    base = dict(BASE, model=XOR_MODEL)

    runs = expand_grid(
        "a", base, {"hidden_depth": [0, 1], "hidden_width": [8, 16], "activation": ["Tanh", "ReLU"]}
    )

    assert [params for _, params, _ in runs] == [
        {"hidden_depth": 0},
        {"hidden_depth": 1, "hidden_width": 8, "activation": "Tanh"},
        {"hidden_depth": 1, "hidden_width": 8, "activation": "ReLU"},
        {"hidden_depth": 1, "hidden_width": 16, "activation": "Tanh"},
        {"hidden_depth": 1, "hidden_width": 16, "activation": "ReLU"},
    ]
    for _, _, config in runs:
        build_model(config["model"], _dataset(2, 1))


def test_expand_grid_mixes_architecture_and_hyperparameters():
    base = dict(BASE, model=WINE_MODEL)

    runs = expand_grid("a", base, {"learning_rate": [0.01, 0.1], "hidden_width": [32, 64]})

    assert len(runs) == 4
    _, params, config = runs[3]
    assert params == {"learning_rate": 0.1, "hidden_width": 64}
    assert config["learning_rate"] == 0.1
    assert [layer["neurons"] for layer in config["model"] if layer["type"] == "Linear"] == [64, 64, 1]



def test_expand_grid_grad_clip_zero_means_off_and_removes_the_key():
    runs = expand_grid("s", BASE, {"grad_clip": [0, 1.0]})

    assert "grad_clip" not in runs[0][2]
    assert runs[1][2]["grad_clip"] == 1.0


def test_expand_grid_grad_clip_zero_dedupes_with_a_base_without_clipping():
    runs = expand_grid("s", BASE, {"grad_clip": [0, 0.0], "learning_rate": [0.1]})

    assert len(runs) == 1


def test_patience_is_dropped_where_there_is_no_validation_set():
    runs = expand_grid("s", BASE, {"validation_split": [0, 0.2], "patience": [10]})

    assert len(runs) == 2
    assert "validation_split" not in runs[0][2] and "patience" not in runs[0][2]
    assert runs[1][2]["validation_split"] == 0.2 and runs[1][2]["patience"] == 10
