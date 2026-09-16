import numpy as np
import pytest

from orbit.core.config import build_dataset, build_model, build_loss, build_optimizer, load_experiment
from orbit.core import Results
from orbit.core.experiment import Experiment
from orbit.nn.layers import Linear
from orbit.nn.activations import Tanh, Sigmoid
from orbit.nn.losses import MSE, CrossEntropy
from orbit.nn.optimizers import SGD


def test_build_dataset_xor_has_four_rows_of_two_features():
    dataset = build_dataset("xor")

    assert len(dataset) == 4
    x, y = dataset[0]
    assert x.shape == (2,)
    assert y.shape == (1,)


def test_build_dataset_unknown_name_raises():
    with pytest.raises(ValueError):
        build_dataset("mnist")


def test_build_model_infers_in_features_from_previous_layer():
    model = build_model([
        {"type": "Linear", "in_features": 2, "neurons": 8},
        {"type": "Tanh"},
        {"type": "Linear", "neurons": 1},
        {"type": "Sigmoid"},
    ])

    layers = list(model._modules.values())
    assert isinstance(layers[0], Linear)
    assert layers[0].weight.shape == (2, 8)
    assert isinstance(layers[1], Tanh)
    assert isinstance(layers[2], Linear)
    # second Linear's in_features (8) must come from the first Linear's
    # neurons, not be re-specified in the config
    assert layers[2].weight.shape == (8, 1)
    assert isinstance(layers[3], Sigmoid)


def test_build_model_first_layer_missing_in_features_raises():
    with pytest.raises(ValueError):
        build_model([{"type": "Linear", "neurons": 8}])


def test_build_model_unknown_type_raises():
    with pytest.raises(ValueError):
        build_model([{"type": "Conv2d", "neurons": 8}])


def test_build_loss_resolves_mse_and_cross_entropy():
    assert isinstance(build_loss("MSE"), MSE)
    assert isinstance(build_loss("CrossEntropy"), CrossEntropy)


def test_build_loss_unknown_name_raises():
    with pytest.raises(ValueError):
        build_loss("Huber")


def test_build_optimizer_resolves_sgd_with_given_lr():
    model = build_model([{"type": "Linear", "in_features": 2, "neurons": 1}])
    optimizer = build_optimizer("SGD", model.parameters(), lr=0.05)

    assert isinstance(optimizer, SGD)
    assert optimizer.lr == 0.05


def test_build_optimizer_unknown_name_raises():
    with pytest.raises(ValueError):
        build_optimizer("Adam", [], lr=0.01)


def test_load_experiment_returns_unrun_experiment():
    config = {
        "name": "xor_mlp_01",
        "dataset": "xor",
        "model": [
            {"type": "Linear", "in_features": 2, "neurons": 8},
            {"type": "Tanh"},
            {"type": "Linear", "neurons": 1},
            {"type": "Sigmoid"},
        ],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": 2.0,
        "batch_size": 4,
        "epochs": 5,
    }

    experiment = load_experiment(config)

    assert isinstance(experiment, Experiment)
    assert experiment.name == "xor_mlp_01"
    assert experiment.dataloader.batch_size == 4
    assert experiment.optimizer.lr == 2.0
    assert experiment.epochs == 5


def test_load_experiment_batch_size_defaults_to_dataloader_default():
    config = {
        "dataset": "xor",
        "model": [{"type": "Linear", "in_features": 2, "neurons": 1}],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": 0.1,
        "epochs": 1,
    }

    experiment = load_experiment(config)

    assert experiment.dataloader.batch_size == 32


def test_load_experiment_trains_and_produces_results():
    np.random.seed(0)
    config = {
        "name": "xor_mlp_01",
        "dataset": "xor",
        "model": [
            {"type": "Linear", "in_features": 2, "neurons": 8},
            {"type": "Tanh"},
            {"type": "Linear", "neurons": 1},
            {"type": "Sigmoid"},
        ],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": 2.0,
        "batch_size": 4,
        "epochs": 1500,
    }

    results = load_experiment(config).run()

    assert isinstance(results, Results)
    assert results.final_loss < results.loss_history[0]
