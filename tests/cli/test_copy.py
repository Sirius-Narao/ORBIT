import json
import questionary

from orbit.cli.commands.copy import copy_experiment


GRAD_CLIP_PROMPT = "Gradient clipping (max norm, blank = off):"


def fake_prompts(monkeypatch, texts, selects=None, grad_clip=None):
    """
    selects maps a select prompt's message to the answer to pick; any
    select prompt not in it stands in for the user accepting the pre-filled
    default (the source's value), same way the canned text answers below
    re-type the source's values to "accept" them. The gradient-clipping
    prompt likewise accepts its pre-filled default unless grad_clip is given.
    """
    selects = selects or {}
    texts = iter(texts)

    class FakeAnswer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    def text(message, *a, **k):
        if message == GRAD_CLIP_PROMPT:
            return FakeAnswer(grad_clip if grad_clip is not None else k.get("default", ""))
        return FakeAnswer(next(texts))

    monkeypatch.setattr(questionary, "text", text)
    monkeypatch.setattr(
        questionary,
        "select",
        lambda message, *a, **k: FakeAnswer(selects.get(message, k.get("default"))),
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
    fake_prompts(monkeypatch, texts=["copied_exp", "0", "3.0", "8", "500", "", "999"])

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
    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "4", "300", "", "1"])

    config_path = copy_experiment("source_exp", root=tmp_path)

    with open(config_path) as f:
        config = json.load(f)

    assert config["seed"] == 1


def test_copy_experiment_blank_seed_gets_a_fresh_random_one(tmp_path, monkeypatch):
    write_source_config(tmp_path, "source_exp")
    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "4", "300", "", ""])

    config_path = copy_experiment("source_exp", root=tmp_path)

    with open(config_path) as f:
        config = json.load(f)

    assert config["seed"] != 1


def test_copy_experiment_blank_batch_size_omits_it(tmp_path, monkeypatch):
    write_source_config(tmp_path, "source_exp")
    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "", "300", "", "1"])

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

    fake_prompts(monkeypatch, texts=["copied_exp", "0", "3.0", "8", "500", "", "999"])

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
    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "4", "300", "0.3", "1"])

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

    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "4", "300", "", "1"])

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

    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "4", "300", "", "1"])

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

    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "4", "300", "", "1"], selects={"Normalize inputs?": "none"})

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert "normalize" not in copy_config


def test_copy_experiment_missing_source(tmp_path, capsys):
    result = copy_experiment("does_not_exist", root=tmp_path)

    assert result is None
    assert "was not found" in capsys.readouterr().out


def test_copy_experiment_carries_accuracy_tolerance_with_task(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["task"] = "regression_tolerance"
    config["accuracy_tolerance"] = 0.25
    with open(config_path, "w") as f:
        json.dump(config, f)

    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "4", "300", "", "1"])

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert copy_config["task"] == "regression_tolerance"
    assert copy_config["accuracy_tolerance"] == 0.25


def test_copy_experiment_can_switch_sgd_with_momentum_to_adam(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["momentum"] = 0.9
    with open(config_path, "w") as f:
        json.dump(config, f)

    # No momentum answer - Adam doesn't prompt for it.
    fake_prompts(
        monkeypatch,
        texts=["copied_exp", "0.01", "4", "300", "", "1"],
        selects={"Optimizer:": "Adam"},
    )

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert copy_config["optimizer"] == "Adam"
    assert copy_config["learning_rate"] == 0.01
    assert copy_config["seed"] == 1
    # momentum belongs to SGD - carrying it over would make load_experiment raise
    assert "momentum" not in copy_config


def test_copy_experiment_momentum_defaults_to_the_source_momentum(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config["momentum"] = 0.9
    with open(config_path, "w") as f:
        json.dump(config, f)

    defaults = []
    class FakeAnswer:
        def __init__(self, value):
            self.value = value
        def ask(self):
            return self.value
    texts = iter(["copied_exp", "0.9", "2.0", "", "4", "300", "", "1"])  # "" = grad clip off
    def fake_text(message, *a, **k):
        defaults.append((message, k.get("default")))
        return FakeAnswer(next(texts))
    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "select", lambda *a, **k: FakeAnswer(k.get("default")))

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    assert ("Momentum (0 = plain SGD):", "0.9") in defaults
    with open(copy_config_path) as f:
        assert json.load(f)["momentum"] == 0.9


def test_copy_experiment_keeps_adam_betas_and_eps(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config.update(optimizer="Adam", learning_rate=0.01, betas=[0.9, 0.99], eps=1e-7)
    with open(config_path, "w") as f:
        json.dump(config, f)

    fake_prompts(monkeypatch, texts=["copied_exp", "0.01", "4", "300", "", "1"])

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert copy_config["optimizer"] == "Adam"
    assert copy_config["betas"] == [0.9, 0.99]
    assert copy_config["eps"] == 1e-7


def test_copy_experiment_drops_adam_options_when_switching_to_sgd(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)
    config.update(optimizer="Adam", learning_rate=0.01, betas=[0.9, 0.99], eps=1e-7)
    with open(config_path, "w") as f:
        json.dump(config, f)

    fake_prompts(
        monkeypatch,
        texts=["copied_exp", "0", "2.0", "4", "300", "", "1"],
        selects={"Optimizer:": "SGD"},
    )

    copy_config_path = copy_experiment("source_exp", root=tmp_path)

    with open(copy_config_path) as f:
        copy_config = json.load(f)

    assert copy_config["optimizer"] == "SGD"
    assert "betas" not in copy_config
    assert "eps" not in copy_config
    assert "momentum" not in copy_config



def test_copy_experiment_prefills_and_keeps_the_source_grad_clip(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    config = json.loads(config_path.read_text())
    config["grad_clip"] = 2.0
    config_path.write_text(json.dumps(config))
    defaults = []

    class FakeAnswer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    texts = iter(["copied_exp", "0", "2.0", "4", "300", "", "1"])

    def fake_text(message, *a, **k):
        if message == GRAD_CLIP_PROMPT:
            defaults.append(k.get("default"))
            return FakeAnswer(k.get("default"))
        return FakeAnswer(next(texts))

    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "select", lambda *a, **k: FakeAnswer(k.get("default")))

    with open(copy_experiment("source_exp", root=tmp_path)) as f:
        copied = json.load(f)

    assert defaults == ["2.0"]
    assert copied["grad_clip"] == 2.0


def test_copy_experiment_can_turn_grad_clip_off(tmp_path, monkeypatch):
    exp_dir = write_source_config(tmp_path, "source_exp")
    config_path = exp_dir / "experiment.json"
    config = json.loads(config_path.read_text())
    config["grad_clip"] = 2.0
    config_path.write_text(json.dumps(config))
    fake_prompts(monkeypatch, texts=["copied_exp", "0", "2.0", "4", "300", "", "1"], grad_clip="")

    with open(copy_experiment("source_exp", root=tmp_path)) as f:
        assert "grad_clip" not in json.load(f)
