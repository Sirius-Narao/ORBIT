import json
import pathlib

import questionary

from orbit.core.config import (
    DATASET_REGISTRY,
    LAYER_REGISTRY,
    LOSS_REGISTRY,
    OPTIMIZER_REGISTRY,
)
from orbit.storage import experiment_dir

# --- small input helpers -----------------------------------------------------
# questionary.select is arrow-keys + enter by nature. For free-form numbers
# (in_features, neurons, learning_rate, ...) there's no fixed set of choices
# to select from, so those still use questionary.text - but validated so a
# bad value re-prompts instead of producing a broken experiment.json.

def _ask_int(message):
    def validate(text):
        return text.strip().isdigit() or "Please enter a whole number"

    answer = questionary.text(message, validate=validate).ask()
    return int(answer.strip())


def _ask_optional_int(message):
    def validate(text):
        text = text.strip()
        return text == "" or text.isdigit() or "Please enter a whole number, or leave blank"

    answer = questionary.text(message, validate=validate).ask()
    answer = answer.strip()
    return int(answer) if answer else None


def _ask_float(message):
    def validate(text):
        try:
            float(text)
            return True
        except ValueError:
            return "Please enter a number"

    answer = questionary.text(message, validate=validate).ask()
    return float(answer)


def _ask_model_layers() -> list:
    layers = []
    layer_choices = list(LAYER_REGISTRY.keys()) + ["Done"]

    while True:
        choice = questionary.select("Add a layer:", choices=layer_choices).ask()

        if choice == "Done":
            if not layers:
                print("A model needs at least one layer.")
                continue
            return layers

        layer = {"type": choice}
        if choice == "Linear":
            if not layers:
                layer["in_features"] = _ask_int("  in_features (size of the input):")
            layer["neurons"] = _ask_int("  neurons (size of this layer's output):")

        layers.append(layer)


# --- top-level command --------------------------------------------------------

def create_experiment() -> pathlib.Path:
    """
    Interactively build an experiment.json-shaped config and save it to
    .orbits/experiments/<name>/experiment.json. Does not run it - that's
    run_experiment()'s job.
    """
    name = questionary.text("Experiment name:").ask()
    dataset = questionary.select("Dataset:", choices=list(DATASET_REGISTRY.keys())).ask()
    model = _ask_model_layers()
    loss = questionary.select("Loss:", choices=list(LOSS_REGISTRY.keys())).ask()
    optimizer = questionary.select("Optimizer:", choices=list(OPTIMIZER_REGISTRY.keys())).ask()
    learning_rate = _ask_float("Learning rate:")
    batch_size = _ask_optional_int("Batch size (blank = default 32):")
    epochs = _ask_int("Epochs:")

    config = {
        "name": name,
        "dataset": dataset,
        "model": model,
        "loss": loss,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
    if batch_size is not None:
        config["batch_size"] = batch_size

    exp_dir = experiment_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "experiment.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved experiment config to {config_path}")
    return config_path
