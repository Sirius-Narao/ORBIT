import json
import questionary
from orbit.cli.commands.new import create_experiment


def fake_prompts(monkeypatch, *, texts, selects):
    """
    questionary.text(...)/.select(...) return prompt objects with a .ask()
    method. Stub both factories to hand back canned answers in call order,
    so create_experiment() runs the same as if a person had used arrow
    keys + enter to pick each one.
    """
    texts = iter(texts)
    selects = iter(selects)

    class FakeAnswer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    monkeypatch.setattr(questionary, "text", lambda *a, **k: FakeAnswer(next(texts)))
    monkeypatch.setattr(questionary, "select", lambda *a, **k: FakeAnswer(next(selects)))


def test_create_experiment_writes_expected_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=[
            "xor_mlp_01",   # name
            "2",            # in_features
            "8",            # neurons (layer 1)
            "1",            # neurons (layer 2)
            "2.0",          # learning_rate
            "4",            # batch_size
            "3000",         # epochs
        ],
        selects=[
            "xor",      # dataset
            "Linear",   # layer 1 type
            "Tanh",     # layer 2 type
            "Linear",   # layer 3 type
            "Sigmoid",  # layer 4 type
            "Done",     # finish model
            "MSE",      # loss
            "SGD",      # optimizer
        ],
    )

    config_path = create_experiment()

    assert config_path == tmp_path / "xor_mlp_01" / "experiment.json"
    with open(config_path) as f:
        config = json.load(f)

    assert config == {
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


def test_create_experiment_omits_batch_size_when_left_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_default_batch", "2", "1", "0.1", "", "10"],
        selects=["xor", "Linear", "Done", "MSE", "SGD"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert "batch_size" not in config
