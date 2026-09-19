import json
import pathlib

import questionary
from rich.table import Table

from orbit.core.config import (
    LAYER_REGISTRY,
    LOSS_REGISTRY,
    OPTIMIZER_REGISTRY,
    build_dataset,
    build_model,
    list_dataset_names,
)
from orbit.storage import experiment_dir
from orbit.ui import PROMPT_STYLE, console, info, success, warning

# --- small input helpers -----------------------------------------------------
# questionary.select is arrow-keys + enter by nature. For free-form numbers
# (in_features, neurons, learning_rate, ...) there's no fixed set of choices
# to select from, so those still use questionary.text - but validated so a
# bad value re-prompts instead of producing a broken experiment.json.

def _ask_int(message):
    def validate(text):
        return text.strip().isdigit() or "Please enter a whole number"

    answer = questionary.text(message, validate=validate, style=PROMPT_STYLE).ask()
    return int(answer.strip())


def _ask_optional_int(message):
    def validate(text):
        text = text.strip()
        return text == "" or text.isdigit() or "Please enter a whole number, or leave blank"

    answer = questionary.text(message, validate=validate, style=PROMPT_STYLE).ask()
    answer = answer.strip()
    return int(answer) if answer else None


def _ask_float(message):
    def validate(text):
        try:
            float(text)
            return True
        except ValueError:
            return "Please enter a number"

    answer = questionary.text(message, validate=validate, style=PROMPT_STYLE).ask()
    return float(answer)


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
    .orbits/experiments/<name>/experiment.json. Does not run it - that's
    run_experiment()'s job.
    """
    name = questionary.text("Experiment name:", style=PROMPT_STYLE).ask()
    dataset_name = questionary.select("Dataset:", choices=list_dataset_names(), style=PROMPT_STYLE).ask()
    dataset = build_dataset(dataset_name)
    info(
        f"Dataset {dataset_name!r}: {dataset.input_shape} input feature(s), "
        f"{dataset.output_shape} output feature(s)"
    )
    model = _ask_model_layers(dataset)
    loss = questionary.select("Loss:", choices=list(LOSS_REGISTRY.keys()), style=PROMPT_STYLE).ask()
    optimizer = questionary.select("Optimizer:", choices=list(OPTIMIZER_REGISTRY.keys()), style=PROMPT_STYLE).ask()
    learning_rate = _ask_float("Learning rate:")
    batch_size = _ask_optional_int("Batch size (blank = default 32):")
    epochs = _ask_int("Epochs:")

    config = {
        "name": name,
        "dataset": dataset_name,
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
    if batch_size is not None:
        config["batch_size"] = batch_size

    _print_config_summary(config)

    exp_dir = experiment_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "experiment.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    success(f"Saved experiment config to {config_path}")
    return config_path


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
    table.add_row("Optimizer", config["optimizer"])
    table.add_row("Learning rate", str(config["learning_rate"]))
    table.add_row("Batch size", str(config.get("batch_size", "32 (default)")))
    table.add_row("Epochs", str(config["epochs"]))
    console.print(table)
