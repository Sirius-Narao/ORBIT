import numpy as np
import pytest

import orbit.core.config as config_module
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


def _fake_imported_dataset(monkeypatch, tmp_path, manifest, csv_text, name="housing"):
    """
    Stands in for what orbit import would have written to disk: a
    dataset.json manifest plus a data.csv, without touching the real
    .orbits/datasets/ directory. Patches the storage helpers build_dataset
    actually calls, the same way tests/cli/test_new.py patches
    experiment_dir rather than the real filesystem location.
    """
    dataset_path = tmp_path / name
    dataset_path.mkdir()
    (dataset_path / "data.csv").write_text(csv_text)

    monkeypatch.setattr(config_module, "dataset_exists", lambda n: n == name)
    monkeypatch.setattr(config_module, "dataset_dir", lambda n: dataset_path)
    monkeypatch.setattr(config_module, "load_dataset_manifest", lambda n: manifest)


def test_build_dataset_resolves_an_imported_csv_dataset(monkeypatch, tmp_path):
    _fake_imported_dataset(
        monkeypatch,
        tmp_path,
        manifest={"input_columns": ["a", "b"], "output_columns": ["c"]},
        csv_text="a,b,c\n1,2,3\n4,5,6\n",
    )

    dataset = build_dataset("housing")

    assert len(dataset) == 2
    x, y = dataset[0]
    assert list(x) == [1.0, 2.0]
    assert list(y) == [3.0]


def test_build_dataset_imported_csv_non_numeric_cell_raises(monkeypatch, tmp_path):
    _fake_imported_dataset(
        monkeypatch,
        tmp_path,
        manifest={"input_columns": ["a", "b"], "output_columns": ["c"]},
        csv_text="a,b,c\n1,not_a_number,3\n",
    )

    with pytest.raises(ValueError):
        build_dataset("housing")


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


def test_build_model_without_dataset_skips_shape_validation():
    # build_model must keep working standalone, with no dataset argument,
    # for callers (like the tests above) that only care about layer wiring.
    model = build_model([{"type": "Linear", "in_features": 2, "neurons": 1}])

    layers = list(model._modules.values())
    assert layers[0].weight.shape == (2, 1)


def test_build_model_matching_dataset_input_and_output_shape_succeeds():
    dataset = build_dataset("xor")  # 2 input features, 1 output feature

    model = build_model(
        [
            {"type": "Linear", "in_features": 2, "neurons": 8},
            {"type": "Linear", "neurons": 1},
        ],
        dataset,
    )

    layers = list(model._modules.values())
    assert layers[0].weight.shape == (2, 8)
    assert layers[1].weight.shape == (8, 1)


def test_build_model_mismatched_dataset_input_shape_raises():
    dataset = build_dataset("xor")  # 2 input features

    with pytest.raises(ValueError):
        build_model(
            [{"type": "Linear", "in_features": 5, "neurons": 1}],
            dataset,
        )


def test_build_model_mismatched_dataset_output_shape_raises():
    dataset = build_dataset("xor")  # 1 output feature

    with pytest.raises(ValueError):
        build_model(
            [{"type": "Linear", "in_features": 2, "neurons": 5}],
            dataset,
        )


def test_build_model_output_shape_checked_past_activation_layers():
    dataset = build_dataset("xor")  # 1 output feature

    with pytest.raises(ValueError):
        build_model(
            [
                {"type": "Linear", "in_features": 2, "neurons": 8},
                {"type": "Tanh"},
                {"type": "Linear", "neurons": 5},
                {"type": "Sigmoid"},
            ],
            dataset,
        )


def test_build_model_missing_in_features_raises_even_with_dataset():
    dataset = build_dataset("xor")

    # A dataset being present should not silently fill in a missing
    # in_features - the config still has to state it explicitly, per the
    # documented experiment.json schema. The dataset is only there to catch
    # a *wrong* value, not to supply a missing one.
    with pytest.raises(ValueError):
        build_model([{"type": "Linear", "neurons": 8}], dataset)


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


def _seeded_config(seed=None):
    config = {
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
        "epochs": 50,
    }
    if seed is not None:
        config["seed"] = seed
    return config


def test_load_experiment_with_seed_produces_identical_runs():
    results_a = load_experiment(_seeded_config(seed=7)).run()
    results_b = load_experiment(_seeded_config(seed=7)).run()

    assert results_a.final_loss == results_b.final_loss
    assert results_a.loss_history == results_b.loss_history


def test_load_experiment_without_seed_does_not_reset_rng():
    """
    A config with no "seed" key must not call np.random.seed() at all -
    otherwise every unseeded config would accidentally become reproducible
    by resetting to some fixed default, which is not what "no seed" means.
    Verified by seeding the global RNG to two different states beforehand
    and checking an unseeded load_experiment() run picks up each state
    (i.e. actually draws from wherever the RNG already was).
    """
    np.random.seed(1)
    results_a = load_experiment(_seeded_config()).run()

    np.random.seed(2)
    results_b = load_experiment(_seeded_config()).run()

    assert results_a.loss_history != results_b.loss_history
