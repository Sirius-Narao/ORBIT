import json
import questionary

from orbit.cli.commands.copy import copy_experiment


def fake_prompts(monkeypatch, texts, select=None):
    """
    select=None stands in for the user accepting the select prompt's
    pre-filled default (the source's value), same way the canned text
    answers below re-type the source's values to "accept" them.
    """
    texts = iter(texts)

    class FakeAnswer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    monkeypatch.setattr(questionary, "text", lambda *a, **k: FakeAnswer(next(texts)))
    monkeypatch.setattr(
        questionary,
        "select",
        lambda *a, **k: FakeAnswer(select if select is not None else k.get("default")),
    )


def write_source_config(root, name):
    exp_dir = root / name
    exp_dir.mkdir(parents=True)

    config = {
        "name": name,
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
        "epochs": 300,
        "seed": 1,
    }

    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)

    return exp_dir


def test_copy_experiment_overrides_hyperparams_and_keeps_model(tmp_path, monkeypatch):
    write_source_config(tmp_path, "source_exp")
    fake_prompts(monkeypatch, texts=["copied_exp", "3.0", "8", "500", "", "999"])

    config_path = copy_experiment("source_exp", root=tmp_path)

    assert config_path == tmp_path / "copied_exp" / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)

    assert config["name"] == "copied_exp"
    assert config["dataset"] == "xor"
    assert config["model"] == [
        {"type": "Linear", "in_features": 2, "neurons": 8},
        {"type": "Tanh"},
        {"type": "Linear", "neurons": 1},
        {"type": "Sigmoid"},
    ]
    assert config["loss"] == "MSE"
    assert config["optimizer"] == "SGD"
    assert config["learning_rate"] == 3.0
    assert config["batch_size"] == 8
    assert config["epochs"] == 500
    assert config["seed"] == 999


def test_copy_experiment_seed_defaults_to_keeping_the_source_seed(tmp_path, monkeypatch):
    write_source_config(tmp_path, "source_exp")
    # "1" here stands in for the user just hitting enter on the pre-filled
    # default - questionary would show the source's seed (1) already typed
    # into the field, so accepting it as-is sends back that same text.
    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "4", "300", "", "1"])

    config_path = copy_experiment("source_exp", root=tmp_path)

    with open(config_path) as f:
        config = json.load(f)

    assert config["seed"] == 1


def test_copy_experiment_blank_seed_gets_a_fresh_random_one(tmp_path, monkeypatch):
    write_source_config(tmp_path, "source_exp")
    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "4", "300", "", ""])

    config_path = copy_experiment("source_exp", root=tmp_path)

    with open(config_path) as f:
        config = json.load(f)

    assert config["seed"] != 1


def test_copy_experiment_blank_batch_size_omits_it(tmp_path, monkeypatch):
    write_source_config(tmp_path, "source_exp")
    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "", "300", "", "1"])

    config_path = copy_experiment("source_exp", root=tmp_path)

    with open(config_path) as f:
        config = json.load(f)

    assert "batch_size" not in config


def test_copy_experiment_preserves_task_when_present(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["task"] = "binary_classification"
    with open(config_path, "w") as f:
        json.dump(config, f)

    fake_prompts(monkeypatch, texts=["copied_exp", "3.0", "8", "500", "", "999"])

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert copy_config["task"] == "binary_classification"


def test_copy_experiment_test_split_defaults_to_source_value(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["test_split"] = 0.3
    with open(config_path, "w") as f:
        json.dump(config, f)

    # "0.3" stands in for the user accepting the pre-filled default, same as
    # test_copy_experiment_seed_defaults_to_keeping_the_source_seed does for seed.
    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "4", "300", "0.3", "1"])

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert copy_config["test_split"] == 0.3


def test_copy_experiment_blank_test_split_omits_it(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["test_split"] = 0.3
    with open(config_path, "w") as f:
        json.dump(config, f)

    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "4", "300", "", "1"])

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert "test_split" not in copy_config


def test_copy_experiment_normalize_defaults_to_source_value(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["normalize"] = "minmax"
    with open(config_path, "w") as f:
        json.dump(config, f)

    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "4", "300", "", "1"])

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert copy_config["normalize"] == "minmax"


def test_copy_experiment_can_turn_normalize_off(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["normalize"] = "standard"
    with open(config_path, "w") as f:
        json.dump(config, f)

    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "4", "300", "", "1"], select="none")

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert "normalize" not in copy_config


def test_copy_experiment_missing_source(tmp_path, capsys):
    result = copy_experiment("does_not_exist", root=tmp_path)

    assert result is None
    assert "was not found" in capsys.readouterr().out
