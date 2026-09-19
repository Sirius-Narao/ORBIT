import json
import questionary

from orbit.cli.commands.copy import copy_experiment


def fake_prompts(monkeypatch, texts):
    texts = iter(texts)

    class FakeAnswer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    monkeypatch.setattr(questionary, "text", lambda *a, **k: FakeAnswer(next(texts)))


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
    fake_prompts(monkeypatch, texts=["copied_exp", "3.0", "8", "500", "999"])

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
    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "4", "300", "1"])

    config_path = copy_experiment("source_exp", root=tmp_path)

    with open(config_path) as f:
        config = json.load(f)

    assert config["seed"] == 1


def test_copy_experiment_blank_seed_gets_a_fresh_random_one(tmp_path, monkeypatch):
    write_source_config(tmp_path, "source_exp")
    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "4", "300", ""])

    config_path = copy_experiment("source_exp", root=tmp_path)

    with open(config_path) as f:
        config = json.load(f)

    assert config["seed"] != 1


def test_copy_experiment_blank_batch_size_omits_it(tmp_path, monkeypatch):
    write_source_config(tmp_path, "source_exp")
    fake_prompts(monkeypatch, texts=["copied_exp", "2.0", "", "300", "1"])

    config_path = copy_experiment("source_exp", root=tmp_path)

    with open(config_path) as f:
        config = json.load(f)

    assert "batch_size" not in config


def test_copy_experiment_missing_source(tmp_path, capsys):
    result = copy_experiment("does_not_exist", root=tmp_path)

    assert result is None
    assert "was not found" in capsys.readouterr().out
