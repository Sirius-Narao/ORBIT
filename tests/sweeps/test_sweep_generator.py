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
