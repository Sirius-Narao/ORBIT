import csv
import numpy as np
from typing import Optional

from orbit.core import (
    Dataset,
    TensorDataset,
    DataLoader,
    train_test_split,
    NormalizedDataset,
    fit_normalizer,
)
from orbit.core.experiment import Experiment
from orbit.core.metrics import accuracy, accuracy_multiclass
from orbit.nn import Sequential
from orbit.nn.layers import Linear
from orbit.nn.activations import ReLU, Tanh, Sigmoid, Softmax
from orbit.nn.losses import MSE, CrossEntropy
from orbit.nn.optimizers import SGD
from orbit.storage import dataset_exists, dataset_dir, load_dataset_manifest, list_imported_dataset_names

# --- dataset registry -------------------------------------------------------
# Each entry is a zero-arg factory (not a built Dataset) so every call to
# build_dataset() returns a fresh object instead of a shared/reused one.
# Imported (CSV) datasets are a separate lookup, not merged into this dict -
# see _load_csv_dataset() and build_dataset()'s fallback below.

def _xor_dataset() -> Dataset:
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]
    return TensorDataset(np.array(X), np.array(Y))

DATASET_REGISTRY = {
    "xor": _xor_dataset,
}

def _load_csv_dataset(name: str) -> Dataset:
    manifest = load_dataset_manifest(name)
    input_columns = manifest["input_columns"]
    output_columns = manifest["output_columns"]

    csv_path = dataset_dir(name) / "data.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    def _column_values(row, columns):
        values = []
        for column in columns:
            try:
                values.append(float(row[column]))
            except ValueError:
                raise ValueError(
                    f"Dataset {name!r}: column {column!r} has a non-numeric "
                    f"value ({row[column]!r}) - only numeric CSV columns are "
                    "supported"
                )
        return values

    X = [_column_values(row, input_columns) for row in rows]
    Y = [_column_values(row, output_columns) for row in rows]

    return TensorDataset(np.array(X), np.array(Y))

def list_dataset_names() -> list:
    return list(DATASET_REGISTRY) + list_imported_dataset_names()

def build_dataset(name: str) -> Dataset:
    if name in DATASET_REGISTRY:
        return DATASET_REGISTRY[name]()
    if dataset_exists(name):
        return _load_csv_dataset(name)
    raise ValueError(f"Unknown dataset: {name!r}")

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

def build_model(layer_configs: list, dataset: Optional[Dataset] = None) -> Sequential:
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
                if dataset is not None and in_features != dataset.input_shape:
                    raise ValueError(
                        f"Model's first layer expects in_features={in_features}, "
                        f"but dataset {dataset!r} provides {dataset.input_shape} "
                        "input feature(s)"
                    )
            out_features = layer_config["neurons"]
            layers.append(Linear(in_features, out_features))
            in_features = out_features
        else:
            layers.append(LAYER_REGISTRY[layer_type]())

    # After the loop, in_features holds the last Linear layer's out_features
    # (activations don't change it) - i.e. the model's actual output width.
    if dataset is not None and in_features != dataset.output_shape:
        raise ValueError(
            f"Model's last layer produces {in_features} output feature(s), "
            f"but dataset {dataset!r} expects {dataset.output_shape} "
            "output feature(s)"
        )

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

# --- task registry -------------------------------------------------------
# Maps a config's optional "task" field to the accuracy function Trainer
# should apply each epoch. Dict-based (not if/elif) so a future task type -
# e.g. a language-modeling task, once ORBIT has the building blocks for one -
# is a one-line addition here rather than a redesign of Trainer/Experiment.

TASK_REGISTRY = {
    "binary_classification": accuracy,
    "multiclass_classification": accuracy_multiclass,
}

def build_accuracy_fn(task: Optional[str]):
    if task is None:
        return None
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task!r}")
    return TASK_REGISTRY[task]

# --- top-level loader -------------------------------------------------------

def load_experiment(config: dict) -> Experiment:
    """
    Build an Experiment from a parsed experiment.json-shaped dict. Does not
    run it - the caller decides when to call .run().

    An optional "seed" is applied via np.random.seed() before anything else
    is built, since Linear's weight init and DataLoader's shuffle both draw
    from the global numpy RNG - seeding first fixes both for the whole run,
    same as tests/integration/test_reproducibility.py's build_and_run(). No
    seed means no call at all, so an unseeded config's randomness is left
    untouched (not reset to some fixed default).

    An optional "test_split" holds out a fraction of the dataset for
    evaluation after training: key absent means no split at all (today's
    behavior, unchanged); present with null/no value defaults to 0.2;
    present with a float uses that ratio. The split is drawn from the same
    seeded RNG stream, right after the dataset is built and before model
    init consumes it, so a seeded config's split is reproducible too.

    An optional "normalize" ("standard" or "minmax"; absent/"none" = off)
    rescales the inputs. Stats are fit on the train split only and reused
    for the test split, so no test-set information leaks into training.
    Nothing is persisted: rebuilding from the same seeded config (e.g.
    orbit test) re-derives the identical split and therefore identical stats.
    """
    seed = config.get("seed")
    if seed is not None:
        np.random.seed(seed)

    dataset = build_dataset(config["dataset"])

    if "test_split" in config:
        test_split = config["test_split"] if config["test_split"] is not None else 0.2
        train_dataset, test_dataset = train_test_split(dataset, test_split)
    else:
        train_dataset, test_dataset = dataset, None

    method = config.get("normalize", "none")
    if method != "none":
        X_train = np.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        stats = fit_normalizer(X_train, method)
        train_dataset = NormalizedDataset(train_dataset, stats)
        if test_dataset is not None:
            test_dataset = NormalizedDataset(test_dataset, stats)

    model = build_model(config["model"], train_dataset)
    loss_fn = build_loss(config["loss"])
    optimizer = build_optimizer(config["optimizer"], model.parameters(), config["learning_rate"])
    dataloader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32))
    test_dataloader = (
        DataLoader(test_dataset, batch_size=config.get("batch_size", 32), shuffle=False)
        if test_dataset is not None
        else None
    )
    accuracy_fn = build_accuracy_fn(config.get("task"))

    return Experiment(
        model,
        loss_fn,
        optimizer,
        dataloader,
        config["epochs"],
        name=config.get("name"),
        verbose=True,
        log_every=100,
        accuracy_fn=accuracy_fn,
        test_dataloader=test_dataloader,
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
