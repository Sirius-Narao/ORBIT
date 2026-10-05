import json
import pathlib

import numpy as np
import questionary
from rich.table import Table

from orbit.core.config import (
    DEFAULT_ACCURACY_TOLERANCE,
    LAYER_REGISTRY,
    LOSS_REGISTRY,
    OPTIMIZER_REGISTRY,
    TASK_REGISTRY,
    build_dataset,
    build_model,
    list_dataset_names,
)
from orbit.core.dataset import NORMALIZE_METHODS
from orbit.settings import load_settings
from orbit.storage import experiment_dir
from orbit.ui import PROMPT_STYLE, console, info, success, warning

NORMALIZE_CHOICES = ["none"] + list(NORMALIZE_METHODS)

# --- small input helpers -----------------------------------------------------
# questionary.select is arrow-keys + enter by nature. For free-form numbers
# (in_features, neurons, learning_rate, ...) there's no fixed set of choices
# to select from, so those still use questionary.text - but validated so a
# bad value re-prompts instead of producing a broken experiment.json.

def _ask_int(message, default=""):
    def validate(text):
        return text.strip().isdigit() or "Please enter a whole number"

    answer = questionary.text(message, default=default, validate=validate, style=PROMPT_STYLE).ask()
    return int(answer.strip())


def _ask_optional_int(message, default=""):
    def validate(text):
        text = text.strip()
        return text == "" or text.isdigit() or "Please enter a whole number, or leave blank"

    answer = questionary.text(message, default=default, validate=validate, style=PROMPT_STYLE).ask()
    answer = answer.strip()
    return int(answer) if answer else None


def _ask_float(message, default=""):
    def validate(text):
        try:
            float(text)
            return True
        except ValueError:
            return "Please enter a number"

    answer = questionary.text(message, default=default, validate=validate, style=PROMPT_STYLE).ask()
    return float(answer)


def _ask_grad_clip(default=""):
    """
    Optional max gradient norm ("grad_clip"); blank or 0 means no clipping,
    returned as None so the key is left out of experiment.json.
    """
    def validate(text):
        text = text.strip()
        if text == "":
            return True
        try:
            return True if float(text) >= 0 else "Please enter a positive number, 0, or leave blank"
        except ValueError:
            return "Please enter a number, or leave blank"

    answer = questionary.text(
        "Gradient clipping (max norm, blank = off):", default=default, validate=validate, style=PROMPT_STYLE
    ).ask()
    answer = answer.strip()
    return float(answer) if answer and float(answer) > 0 else None


VALIDATION_SPLIT_PROMPT = "Validation split fraction (0-1, blank = none):"
PATIENCE_PROMPT = "Early stopping patience (epochs without improvement, blank = off):"


def _ask_early_stopping(default_split="", default_patience=""):
    """
    Optional validation split, then - only when one was given - the early
    stopping patience. Returns (validation_split, patience), None for blank.
    """
    def validate_split(text):
        text = text.strip()
        if text == "":
            return True
        try:
            return True if 0 < float(text) < 1 else "Please enter a fraction between 0 and 1, or leave blank"
        except ValueError:
            return "Please enter a number, or leave blank"

    answer = questionary.text(
        VALIDATION_SPLIT_PROMPT, default=default_split, validate=validate_split, style=PROMPT_STYLE
    ).ask().strip()
    if not answer:
        return None, None
    validation_split = float(answer)

    def validate_patience(text):
        text = text.strip()
        return text == "" or (text.isdigit() and int(text) >= 1) or "Please enter a whole number >= 1, or leave blank"

    answer = questionary.text(
        PATIENCE_PROMPT, default=default_patience, validate=validate_patience, style=PROMPT_STYLE
    ).ask().strip()
    return validation_split, int(answer) if answer else None


def _ask_optional_float(message, default=""):
    def validate(text):
        text = text.strip()
        if text == "":
            return True
        try:
            float(text)
            return True
        except ValueError:
            return "Please enter a number, or leave blank"

    answer = questionary.text(message, default=default, validate=validate, style=PROMPT_STYLE).ask()
    answer = answer.strip()
    return float(answer) if answer else None


def _ask_model_layers(dataset) -> list:
    """
    dataset is the Dataset the user already picked - its input_shape fills
    in the first Linear layer's in_features instead of asking (so it can't
    drift out of sync with the dataset), and once the user says "Done",
    build_model(layers, dataset) is used to check the model's final output
    width against dataset.output_shape too. Reusing build_model here, rather
    than re-deriving the output width by hand, keeps this check identical to
    the one orbit run applies when loading a saved (or hand-edited) config.
    """
    layers = []
    layer_choices = list(LAYER_REGISTRY.keys()) + ["Done"]

    while True:
        choice = questionary.select("Add a layer:", choices=layer_choices, style=PROMPT_STYLE).ask()

        if choice == "Done":
            if not layers:
                warning("A model needs at least one layer.")
                continue
            try:
                build_model(layers, dataset)
            except ValueError as e:
                warning(f"Invalid model: {e}")
                continue
            return layers

        layer = {"type": choice}
        if choice == "Linear":
            if not layers:
                layer["in_features"] = dataset.input_shape
            layer["neurons"] = _ask_int("  neurons (size of this layer's output):")

        layers.append(layer)


# --- top-level command --------------------------------------------------------

def create_experiment() -> pathlib.Path:
    """
    Interactively build an experiment.json-shaped config and save it to
    <workspace>/experiments/<name>/experiment.json. Does not run it - that's
    run_experiment()'s job.

    The hyperparameter prompts are pre-filled from the settings file's
    [defaults] (see orbit/settings.py). Whatever the user accepts is written
    into experiment.json explicitly, so the saved experiment never depends
    on the settings file afterwards.
    """
    defaults = load_settings()["defaults"]
    name = questionary.text("Experiment name:", style=PROMPT_STYLE).ask()
    dataset_name = questionary.select("Dataset:", choices=list_dataset_names(), style=PROMPT_STYLE).ask()
    dataset = build_dataset(dataset_name)
    if dataset.num_classes is not None:
        info(
            f"Dataset {dataset_name!r}: {dataset.input_shape} input feature(s), "
            f"{dataset.num_classes} classes - end the model with Linear({dataset.num_classes}) "
            "(no Softmax: CrossEntropy applies it), use CrossEntropy and multiclass_classification"
        )
    else:
        info(
            f"Dataset {dataset_name!r}: {dataset.input_shape} input feature(s), "
            f"{dataset.output_shape} output feature(s)"
        )
    model = _ask_model_layers(dataset)
    loss = questionary.select("Loss:", choices=list(LOSS_REGISTRY.keys()), style=PROMPT_STYLE).ask()
    task_choice = questionary.select(
        "Track accuracy?", choices=["No (not tracked)"] + list(TASK_REGISTRY.keys()), style=PROMPT_STYLE
    ).ask()
    accuracy_tolerance = None
    if task_choice == "regression_tolerance":
        accuracy_tolerance = _ask_float(
            "Accuracy tolerance (|prediction - target| counted as correct):",
            default=str(DEFAULT_ACCURACY_TOLERANCE),
        )
    optimizer = questionary.select(
        "Optimizer:", choices=list(OPTIMIZER_REGISTRY.keys()), default=defaults["optimizer"], style=PROMPT_STYLE
    ).ask()
    momentum = _ask_momentum(optimizer, default=_format_default(defaults["momentum"]))
    normalize = questionary.select(
        "Normalize inputs?", choices=NORMALIZE_CHOICES, default=defaults["normalize"], style=PROMPT_STYLE
    ).ask()
    learning_rate = _ask_float("Learning rate:", default=_format_default(default_learning_rate(optimizer, defaults)))
    grad_clip = _ask_grad_clip(default=_format_default(defaults["grad_clip"], zero_as_blank=True))
    batch_size = _ask_optional_int("Batch size (blank = 32):", default=_format_default(defaults["batch_size"]))
    epochs = _ask_int("Epochs:", default=_format_default(defaults["epochs"]))
    test_split = _ask_optional_float(
        "Test split fraction (0-1, blank = no split):",
        default=_format_default(defaults["test_split"], zero_as_blank=True),
    )
    if not test_split:
        test_split = None
    validation_split, patience = _ask_early_stopping()
    seed = _ask_optional_int("Seed (blank = random):")
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))

    config = {
        "name": name,
        "dataset": dataset_name,
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
    }
    if batch_size is not None:
        config["batch_size"] = batch_size
    if test_split is not None:
        config["test_split"] = test_split
    if task_choice != "No (not tracked)":
        config["task"] = task_choice
    if accuracy_tolerance is not None:
        config["accuracy_tolerance"] = accuracy_tolerance
    if normalize != "none":
        config["normalize"] = normalize
    if momentum:
        config["momentum"] = momentum
    if grad_clip is not None:
        config["grad_clip"] = grad_clip
    if validation_split is not None:
        config["validation_split"] = validation_split
    if patience is not None:
        config["patience"] = patience

    _print_config_summary(config)

    exp_dir = experiment_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "experiment.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    success(f"Saved experiment config to {config_path}")
    console.print()
    return config_path


def default_learning_rate(optimizer: str, defaults: dict) -> float:
    """
    The settings' pre-filled learning rate for an optimizer. Adam gets its
    own (adam_learning_rate, conventionally 1e-3): its step is roughly lr per
    weight regardless of gradient scale, so SGD's default would be far too big.
    """
    return defaults["adam_learning_rate"] if optimizer == "Adam" else defaults["learning_rate"]


def _format_default(value, zero_as_blank: bool = False) -> str:
    """A settings value as a prompt's pre-filled text: 0.1 -> "0.1", 32 -> "32"."""
    if zero_as_blank and not value:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)


def _ask_momentum(optimizer, default="0"):
    """Momentum prompt, SGD only - returns None for every other optimizer."""
    if optimizer != "SGD":
        return None
    return _ask_float("Momentum (0 = plain SGD):", default=default)


def _format_optimizer(config: dict) -> str:
    """e.g. "SGD", "SGD (momentum 0.9)", "Adam", "Adam (betas 0.9, 0.99, eps 1e-07)"."""
    details = []
    if config.get("momentum"):
        details.append(f"momentum {config['momentum']}")
    if "betas" in config:
        details.append("betas " + ", ".join(str(b) for b in config["betas"]))
    if "eps" in config:
        details.append(f"eps {config['eps']}")
    name = config["optimizer"]
    return f"{name} ({', '.join(details)})" if details else name


def _format_model_summary(layers: list) -> str:
    parts = []
    for layer in layers:
        if layer["type"] == "Linear":
            in_features = layer.get("in_features", "?")
            parts.append(f"Linear({in_features}->{layer['neurons']})")
        else:
            parts.append(layer["type"])
    return " -> ".join(parts)


def _print_config_summary(config: dict) -> None:
    table = Table(title="Experiment Summary", show_header=False)
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("Name", config["name"])
    table.add_row("Dataset", config["dataset"])
    table.add_row("Model", _format_model_summary(config["model"]))
    table.add_row("Loss", config["loss"])
    task = config.get("task", "none (not tracked)")
    if task == "regression_tolerance":
        task += f" (±{config.get('accuracy_tolerance', DEFAULT_ACCURACY_TOLERANCE)})"
    table.add_row("Task", task)
    table.add_row("Optimizer", _format_optimizer(config))
    table.add_row("Learning rate", str(config["learning_rate"]))
    table.add_row("Grad clip", str(config["grad_clip"]) if config.get("grad_clip") else "off")
    table.add_row("Batch size", str(config.get("batch_size", "32 (default)")))
    table.add_row("Epochs", str(config["epochs"]))
    table.add_row("Test split", str(config.get("test_split", "none")))
    table.add_row("Validation split", str(config.get("validation_split", "none")))
    table.add_row(
        "Early stopping",
        f"patience {config['patience']}" if config.get("patience") else "off",
    )
    table.add_row("Normalize", config.get("normalize", "none"))
    table.add_row("Seed", str(config.get("seed", "none (not reproducible)")))
    console.print()
    console.print(table)
    console.print()
