import json
import pathlib

import numpy as np
import questionary

from orbit.cli.commands.new import (
    NORMALIZE_CHOICES,
    _ask_float,
    _ask_int,
    _ask_optional_float,
    _ask_optional_int,
    _print_config_summary,
)
from orbit.storage import EXPERIMENTS_ROOT, experiment_dir
from orbit.ui import PROMPT_STYLE, console, success, warning


def copy_experiment(source_name: str, root: pathlib.Path = EXPERIMENTS_ROOT):
    """
    Build a new experiment.json from an existing one's config: dataset,
    model, loss, optimizer, and task (if set) are copied unchanged (editing
    those means running orbit new from scratch); normalize/learning_rate/
    batch_size/epochs/test_split/seed are all re-prompted with the source's
    values pre-filled as defaults.
    Keeping the source's seed (just hit enter) is deliberately the default
    - it's what lets a copy isolate the effect of a hyperparameter change
    from random-init/shuffle noise, the same way an ablation study holds
    everything but one variable fixed. Clearing the field picks a fresh
    random seed instead, for a copy that's meant to be a genuinely
    independent run.
    """
    source_config_path = experiment_dir(source_name, root=root) / "experiment.json"

    if not source_config_path.exists():
        warning(f"{source_name} was not found.")
        return None

    with open(source_config_path) as f:
        source = json.load(f)

    new_name = questionary.text("New experiment name:", style=PROMPT_STYLE).ask()
    normalize = questionary.select(
        "Normalize inputs?",
        choices=NORMALIZE_CHOICES,
        default=source.get("normalize", "none"),
        style=PROMPT_STYLE,
    ).ask()
    learning_rate = _ask_float("Learning rate:", default=str(source["learning_rate"]))
    batch_size = _ask_optional_int(
        "Batch size (blank = default 32):", default=str(source.get("batch_size", ""))
    )
    epochs = _ask_int("Epochs:", default=str(source["epochs"]))
    test_split = _ask_optional_float(
        "Test split fraction (0-1, blank = no split):", default=str(source.get("test_split", ""))
    )
    seed = _ask_optional_int(
        "Seed (blank = random):", default=str(source.get("seed", ""))
    )
    if seed is None:
        seed = int(np.random.randint(0, 2**31 - 1))

    config = {
        "name": new_name,
        "dataset": source["dataset"],
        "model": source["model"],
        "loss": source["loss"],
        "optimizer": source["optimizer"],
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
    }
    if batch_size is not None:
        config["batch_size"] = batch_size
    if test_split is not None:
        config["test_split"] = test_split
    if "task" in source:
        config["task"] = source["task"]
    if "accuracy_tolerance" in source:
        config["accuracy_tolerance"] = source["accuracy_tolerance"]
    if normalize != "none":
        config["normalize"] = normalize

    _print_config_summary(config)

    exp_dir = experiment_dir(new_name, root=root)
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "experiment.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    success(f"Saved experiment config to {config_path}")
    console.print()
    return config_path
