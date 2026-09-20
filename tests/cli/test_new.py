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
            "8",            # neurons (layer 1)
            "1",            # neurons (layer 2)
            "2.0",          # learning_rate
            "4",            # batch_size
            "3000",         # epochs
            "",             # test_split (blank = no split)
            "42",           # seed
        ],
        selects=[
            "xor",      # dataset
            "Linear",   # layer 1 type
            "Tanh",     # layer 2 type
            "Linear",   # layer 3 type
            "Sigmoid",           # layer 4 type
            "Done",              # finish model
            "MSE",               # loss
            "No (not tracked)",  # track accuracy?
            "SGD",               # optimizer
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
        "seed": 42,
    }


def test_create_experiment_omits_batch_size_when_left_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_default_batch", "1", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert "batch_size" not in config


def test_create_experiment_fills_in_features_from_dataset_without_prompting(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    # Only 5 canned text answers: name, neurons, learning_rate, batch_size
    # (blank), epochs. There is deliberately no answer for in_features - if
    # create_experiment() still prompted for it, this would either raise
    # StopIteration or shift every later answer by one and fail below.
    fake_prompts(
        monkeypatch,
        texts=["xor_single_layer", "1", "0.05", "", "8", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["model"] == [{"type": "Linear", "in_features": 2, "neurons": 1}]

    captured = capsys.readouterr()
    assert "2 input feature(s)" in captured.out


def test_create_experiment_rejects_output_shape_mismatch_and_lets_user_fix_it(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    # First attempt ends in a Linear(2, 5) - xor only has 1 output feature,
    # so "Done" must be rejected instead of saving a broken config. The user
    # then adds a corrective Linear(5, 1) and finishes again, which should
    # succeed this time.
    fake_prompts(
        monkeypatch,
        texts=["bad_output_then_fixed", "5", "1", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "Linear", "Done", "MSE", "No (not tracked)", "SGD"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["model"] == [
        {"type": "Linear", "in_features": 2, "neurons": 5},
        {"type": "Linear", "neurons": 1},
    ]

    captured = capsys.readouterr()
    assert "Invalid model" in captured.out


def test_create_experiment_dataset_picker_includes_imported_datasets(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )
    # Stand in for an imported CSV dataset showing up in the picker, without
    # touching the real .orbits/datasets/ directory or DATASET_REGISTRY.
    monkeypatch.setattr(
        "orbit.cli.commands.new.list_dataset_names", lambda: ["xor", "housing"]
    )

    class FakeImportedDataset:
        input_shape = 3
        output_shape = 1

    monkeypatch.setattr(
        "orbit.cli.commands.new.build_dataset", lambda name: FakeImportedDataset()
    )

    fake_prompts(
        monkeypatch,
        texts=["housing_model", "1", "0.1", "", "50", "", ""],
        selects=["housing", "Linear", "Done", "MSE", "No (not tracked)", "SGD"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["dataset"] == "housing"
    assert config["model"] == [{"type": "Linear", "in_features": 3, "neurons": 1}]


def test_create_experiment_sets_task_when_accuracy_tracking_chosen(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_binary", "8", "1", "2.0", "4", "3000", "", "42"],
        selects=[
            "xor", "Linear", "Tanh", "Linear", "Sigmoid", "Done",
            "MSE", "binary_classification", "SGD",
        ],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["task"] == "binary_classification"


def test_create_experiment_includes_test_split_when_given(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_with_split", "1", "0.1", "", "10", "0.2", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["test_split"] == 0.2


def test_create_experiment_omits_task_when_not_tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_no_tracking", "1", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert "task" not in config
