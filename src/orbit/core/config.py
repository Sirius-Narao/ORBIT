import numpy as np
from typing import Optional

from orbit.core import Dataset, TensorDataset, DataLoader
from orbit.core.experiment import Experiment
from orbit.nn import Sequential
from orbit.nn.layers import Linear
from orbit.nn.activations import ReLU, Tanh, Sigmoid, Softmax
from orbit.nn.losses import MSE, CrossEntropy
from orbit.nn.optimizers import SGD

# --- dataset registry -------------------------------------------------------
# Each entry is a zero-arg factory (not a built Dataset) so every call to
# build_dataset() returns a fresh object instead of a shared/reused one.

def _xor_dataset() -> Dataset:
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]
    return TensorDataset(np.array(X), np.array(Y))

DATASET_REGISTRY = {
    "xor": _xor_dataset,
}

def build_dataset(name: str) -> Dataset:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name!r}")
    return DATASET_REGISTRY[name]()

# --- model registry ----------------------------------------------------------
# Types split into "shape-changing" (Linear, needs in/out features) and
# "shape-preserving" (activations, take no constructor args).

LAYER_REGISTRY = {
    "Linear": Linear,
    "ReLU": ReLU,
    "Tanh": Tanh,
    "Sigmoid": Sigmoid,
    "Softmax": Softmax,
}

def build_model(layer_configs: list) -> Sequential:
    layers = []
    in_features = None

    for i, layer_config in enumerate(layer_configs):
        layer_type = layer_config.get("type")
        if layer_type not in LAYER_REGISTRY:
            raise ValueError(f"Unknown layer type: {layer_type!r}")

        if layer_type == "Linear":
            if i == 0:
                in_features = layer_config.get("in_features")
                if in_features is None:
                    raise ValueError("The first layer must specify 'in_features'")
            out_features = layer_config["neurons"]
            layers.append(Linear(in_features, out_features))
            in_features = out_features
        else:
            layers.append(LAYER_REGISTRY[layer_type]())

    return Sequential(*layers)

# --- loss registry -------------------------------------------------------

LOSS_REGISTRY = {
    "MSE": MSE,
    "CrossEntropy": CrossEntropy,
}

def build_loss(name: str):
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name!r}")
    return LOSS_REGISTRY[name]()

# --- optimizer registry -------------------------------------------------------
# Only SGD exists as an Optimizer subclass right now (v1 scope) - kept as a
# registry rather than an if-check so adding Adam later is a one-line change.

OPTIMIZER_REGISTRY = {
    "SGD": SGD,
}

def build_optimizer(name: str, parameters, lr: float):
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name!r}")
    return OPTIMIZER_REGISTRY[name](parameters, lr=lr)

# --- top-level loader -------------------------------------------------------

def load_experiment(config: dict) -> Experiment:
    """
    Build an Experiment from a parsed experiment.json-shaped dict. Does not
    run it - the caller decides when to call .run().
    """
    dataset = build_dataset(config["dataset"])
    model = build_model(config["model"])
    loss_fn = build_loss(config["loss"])
    optimizer = build_optimizer(config["optimizer"], model.parameters(), config["learning_rate"])
    dataloader = DataLoader(dataset, batch_size=config.get("batch_size", 32))

    return Experiment(
        model,
        loss_fn,
        optimizer,
        dataloader,
        config["epochs"],
        name=config.get("name"),
        verbose=True,
        log_every=100
    )

if __name__ == "__main__":
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
        "epochs": 3000,
    }
    experiment = load_experiment(config)
    results = experiment.run()
    print(results)
