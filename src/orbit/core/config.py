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
from orbit.core.toy_datasets import TOY_DATASETS
from orbit.core.columns import class_indices, encode_column
import functools

from orbit.core.metrics import accuracy, accuracy_multiclass, regression_tolerance, r2_score
from orbit.nn import Sequential
from orbit.nn.layers import Linear
from orbit.nn.activations import ReLU, Tanh, Sigmoid, Softmax
from orbit.nn.losses import MSE, CrossEntropy
from orbit.nn.optimizers import SGD, Adam
from orbit.storage import (
    FEATURES_FILENAME,
    dataset_dir,
    dataset_exists,
    list_imported_dataset_names,
    load_dataset_manifest,
)

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
    **TOY_DATASETS,  # moons, circles, spirals, blobs - see core/toy_datasets.py
}

def _load_csv_dataset(name: str) -> Dataset:
    """
    Encode an imported CSV with its manifest: each input column becomes 1+
    numbers per its type (see core/columns.py), concatenated in manifest
    order. A single categorical target becomes class indices (num_classes).
    """
    manifest = load_dataset_manifest(name)
    input_columns = manifest["input_columns"]
    output_columns = manifest["output_columns"]
    column_specs = manifest.get("columns", {})

    directory = dataset_dir(name)
    with open(directory / "data.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    images = {}
    if any(column_specs.get(c, {}).get("type") == "image" for c in input_columns):
        with np.load(directory / FEATURES_FILENAME) as features:
            images = {key: features[key] for key in features.files}

    def _encode(column):
        values = [row[column] for row in rows]
        try:
            return encode_column(column, column_specs.get(column), values, images.get(column))
        except ValueError as e:
            raise ValueError(f"Dataset {name!r}: {e}")

    X = np.concatenate([_encode(c) for c in input_columns], axis=1)

    target_specs = [column_specs.get(c) for c in output_columns]
    if len(output_columns) == 1 and target_specs[0] is not None and target_specs[0]["type"] == "categorical":
        column, spec = output_columns[0], target_specs[0]
        Y = class_indices(column, spec, [row[column] for row in rows])
        return TensorDataset(X, Y, num_classes=len(spec["categories"]))

    Y = np.concatenate([_encode(c) for c in output_columns], axis=1)
    return TensorDataset(X, Y)

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
# OPTIMIZER_OPTIONS lists the optional config fields each optimizer accepts
# (flat keys in experiment.json, all with standard defaults when absent):
# "momentum" for SGD (default 0 = plain SGD), "betas"/"eps" for Adam
# (default (0.9, 0.999) / 1e-8).

OPTIMIZER_REGISTRY = {
    "SGD": SGD,
    "Adam": Adam,
}

OPTIMIZER_OPTIONS = {
    "SGD": ("momentum",),
    "Adam": ("betas", "eps"),
}

ALL_OPTIMIZER_OPTIONS = tuple(
    option for options in OPTIMIZER_OPTIONS.values() for option in options
)

def build_optimizer(name: str, parameters, lr: float, **options):
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name!r}")
    # An option meant for a different optimizer (e.g. "momentum" on Adam)
    # would otherwise be silently ignored - fail loudly instead.
    for option in options:
        if option not in OPTIMIZER_OPTIONS[name]:
            raise ValueError(f"Optimizer {name!r} does not accept option {option!r}")
    if "betas" in options:
        options["betas"] = tuple(options["betas"])  # JSON stores it as a list
    return OPTIMIZER_REGISTRY[name](parameters, lr=lr, **options)

# --- task registry -------------------------------------------------------
# Maps a config's optional "task" field to the accuracy function Trainer
# should apply each epoch. Dict-based (not if/elif) so a future task type -
# e.g. a language-modeling task, once ORBIT has the building blocks for one -
# is a one-line addition here rather than a redesign of Trainer/Experiment.

TASK_REGISTRY = {
    "binary_classification": accuracy,
    "multiclass_classification": accuracy_multiclass,
    "regression_tolerance": regression_tolerance,
    "regression_r2": r2_score,
}

DEFAULT_ACCURACY_TOLERANCE = 0.5

def build_accuracy_fn(task: Optional[str], tolerance: Optional[float] = None):
    if task is None:
        return None
    if task not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task!r}")
    if task == "regression_tolerance":
        return functools.partial(
            regression_tolerance,
            tolerance=tolerance if tolerance is not None else DEFAULT_ACCURACY_TOLERANCE,
        )
    return TASK_REGISTRY[task]

# --- top-level loader -------------------------------------------------------

def parse_grad_clip(config: dict) -> Optional[float]:
    """
    The config's optional "grad_clip" (max global gradient norm per batch,
    see Trainer's _clip_gradients). Absent, null or 0 means no clipping -
    0 is accepted so a sweep can include an "off" value in its grid.
    """
    value = config.get("grad_clip")
    if value is None or value == 0:
        return None
    if value < 0:
        raise ValueError(f'"grad_clip" must be positive (or 0 for no clipping), got {value!r}')
    return float(value)

def parse_early_stopping(config: dict) -> tuple:
    """
    (validation_split, patience) from a config, both None when absent.
    validation_split must be in (0, 1); patience a whole number >= 1 that
    needs a validation_split (early stopping watches the validation loss).
    """
    validation_split = config.get("validation_split")
    patience = config.get("patience")
    if validation_split is not None:
        if isinstance(validation_split, bool) or not 0 < float(validation_split) < 1:
            raise ValueError(f"validation_split must be between 0 and 1, got {validation_split!r}")
        validation_split = float(validation_split)
    if patience is not None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError(f"patience must be a whole number >= 1, got {patience!r}")
        if validation_split is None:
            raise ValueError("patience (early stopping) needs a validation_split to watch")
    return validation_split, patience


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

    An optional "grad_clip" caps every batch's global gradient norm (see
    parse_grad_clip); absent means training exactly as before it existed.

    An optional "validation_split" holds out a further fraction of the
    dataset that is measured after every epoch, and "patience" turns that
    into early stopping (see parse_early_stopping and Trainer._validate).
    """
    grad_clip = parse_grad_clip(config)
    seed = config.get("seed")
    if seed is not None:
        np.random.seed(seed)

    dataset = build_dataset(config["dataset"])

    if "test_split" in config:
        test_split = config["test_split"] if config["test_split"] is not None else 0.2
        train_dataset, test_dataset = train_test_split(dataset, test_split)
    else:
        test_split = 0.0
        train_dataset, test_dataset = dataset, None

    # The validation set is drawn from what's left after the test split, so
    # a config without "validation_split" draws nothing extra from the RNG
    # and its seeded run is unchanged. validation_split is a fraction of the
    # whole dataset, hence the rescaling: 0.2 of all rows is
    # 0.2 / (1 - test_split) of the remaining ones.
    validation_split, patience = parse_early_stopping(config)
    val_dataset = None
    if validation_split:
        if validation_split + test_split >= 1:
            raise ValueError("validation_split + test_split must leave some rows for training")
        train_dataset, val_dataset = train_test_split(train_dataset, validation_split / (1 - test_split))

    method = config.get("normalize", "none")
    if method != "none":
        X_train = np.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        stats = fit_normalizer(X_train, method)
        train_dataset = NormalizedDataset(train_dataset, stats)
        if test_dataset is not None:
            test_dataset = NormalizedDataset(test_dataset, stats)
        if val_dataset is not None:
            val_dataset = NormalizedDataset(val_dataset, stats)

    model = build_model(config["model"], train_dataset)
    loss_fn = build_loss(config["loss"])
    optimizer_options = {k: config[k] for k in ALL_OPTIMIZER_OPTIONS if k in config}
    optimizer = build_optimizer(
        config["optimizer"], model.parameters(), config["learning_rate"], **optimizer_options
    )
    dataloader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32))
    test_dataloader = (
        DataLoader(test_dataset, batch_size=config.get("batch_size", 32), shuffle=False)
        if test_dataset is not None
        else None
    )
    val_dataloader = (
        DataLoader(val_dataset, batch_size=config.get("batch_size", 32), shuffle=False)
        if val_dataset is not None
        else None
    )
    task = config.get("task")
    accuracy_fn = build_accuracy_fn(task, config.get("accuracy_tolerance"))
    accuracy_tolerance = (
        config.get("accuracy_tolerance", DEFAULT_ACCURACY_TOLERANCE)
        if task == "regression_tolerance"
        else None
    )

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
        task=task,
        accuracy_tolerance=accuracy_tolerance,
        grad_clip=grad_clip,
        val_dataloader=val_dataloader,
        patience=patience,
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
