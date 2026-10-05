import json
import questionary
from orbit.cli.commands.new import create_experiment


from orbit.cli.commands.new import PATIENCE_PROMPT, VALIDATION_SPLIT_PROMPT

GRAD_CLIP_PROMPT = "Gradient clipping (max norm, blank = off):"


def fake_prompts(monkeypatch, *, texts, selects, grad_clip="", validation_split="", patience=""):
    """
    questionary.text(...)/.select(...) return prompt objects with a .ask()
    method. Stub both factories to hand back canned answers in call order,
    so create_experiment() runs the same as if a person had used arrow
    keys + enter to pick each one. The gradient-clipping prompt is answered
    separately (grad_clip, blank = off by default), so the positional texts
    lists don't all have to account for it - and so are the validation split
    and early stopping prompts.
    """
    texts = iter(texts)
    selects = iter(selects)

    class FakeAnswer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    by_message = {
        GRAD_CLIP_PROMPT: grad_clip,
        VALIDATION_SPLIT_PROMPT: validation_split,
        PATIENCE_PROMPT: patience,
    }
    monkeypatch.setattr(
        questionary, "text",
        lambda message, *a, **k: FakeAnswer(by_message[message] if message in by_message else next(texts)),
    )
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
            "0",            # momentum (SGD)
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
            "none",              # normalize inputs?
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
        texts=["xor_default_batch", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
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

    # Only 7 canned text answers: name, neurons, momentum, learning_rate,
    # batch_size (blank), epochs, test_split, seed. There is deliberately no
    # answer for in_features - if create_experiment() still prompted for it,
    # this would either raise StopIteration or shift every later answer by
    # one and fail below.
    fake_prompts(
        monkeypatch,
        texts=["xor_single_layer", "1", "0", "0.05", "", "8", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
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
        texts=["bad_output_then_fixed", "5", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
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
        num_classes = None

    monkeypatch.setattr(
        "orbit.cli.commands.new.build_dataset", lambda name: FakeImportedDataset()
    )

    fake_prompts(
        monkeypatch,
        texts=["housing_model", "1", "0", "0.1", "", "50", "", ""],
        selects=["housing", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
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
        texts=["xor_binary", "8", "1", "0", "2.0", "4", "3000", "", "42"],
        selects=[
            "xor", "Linear", "Tanh", "Linear", "Sigmoid", "Done",
            "MSE", "binary_classification", "SGD", "none",
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
        texts=["xor_with_split", "1", "0", "0.1", "", "10", "0.2", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["test_split"] == 0.2


def test_create_experiment_sets_normalize_when_chosen(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_normalized", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "standard"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["normalize"] == "standard"


def test_create_experiment_omits_normalize_when_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_raw", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert "normalize" not in config


def test_create_experiment_omits_task_when_not_tracked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_no_tracking", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert "task" not in config


def test_create_experiment_prompts_for_tolerance_with_regression_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        # name, neurons, tolerance, momentum, learning_rate, batch_size, epochs, test_split, seed
        texts=["xor_tol", "1", "0.25", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "regression_tolerance", "SGD", "none"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["task"] == "regression_tolerance"
    assert config["accuracy_tolerance"] == 0.25


def test_create_experiment_regression_r2_does_not_prompt_for_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    # No tolerance answer in texts - an extra prompt would shift every later
    # answer (learning_rate would get "10") or raise StopIteration.
    fake_prompts(
        monkeypatch,
        texts=["xor_r2", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "regression_r2", "SGD", "none"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["task"] == "regression_r2"
    assert config["learning_rate"] == 0.1
    assert "accuracy_tolerance" not in config


def test_create_experiment_writes_momentum_when_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_momentum", "1", "0.9", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["optimizer"] == "SGD"
    assert config["momentum"] == 0.9


def test_create_experiment_omits_momentum_when_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    fake_prompts(
        monkeypatch,
        texts=["xor_plain_sgd", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert "momentum" not in config


def test_create_experiment_adam_skips_momentum_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name
    )

    # No momentum answer - Adam must not ask for one, or "0.01" would be
    # consumed as momentum and every later answer would shift by one.
    fake_prompts(
        monkeypatch,
        texts=["xor_adam", "1", "0.01", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "Adam", "none"],
    )

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)

    assert config["optimizer"] == "Adam"
    assert config["learning_rate"] == 0.01
    assert "momentum" not in config


def test_format_optimizer():
    from orbit.cli.commands.new import _format_optimizer

    assert _format_optimizer({"optimizer": "SGD"}) == "SGD"
    assert _format_optimizer({"optimizer": "SGD", "momentum": 0.9}) == "SGD (momentum 0.9)"
    assert _format_optimizer({"optimizer": "Adam"}) == "Adam"
    assert (
        _format_optimizer({"optimizer": "Adam", "betas": [0.9, 0.99], "eps": 1e-07})
        == "Adam (betas 0.9, 0.99, eps 1e-07)"
    )



def test_create_experiment_writes_grad_clip_when_given(tmp_path, monkeypatch):
    monkeypatch.setattr("orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name)
    fake_prompts(
        monkeypatch,
        texts=["xor_clipped", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
        grad_clip="1.5",
    )

    with open(create_experiment()) as f:
        config = json.load(f)

    assert config["grad_clip"] == 1.5


def test_create_experiment_omits_grad_clip_when_zero(tmp_path, monkeypatch):
    monkeypatch.setattr("orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name)
    fake_prompts(
        monkeypatch,
        texts=["xor_unclipped", "1", "0", "0.1", "", "10", "", ""],
        selects=["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"],
        grad_clip="0",
    )

    with open(create_experiment()) as f:
        assert "grad_clip" not in json.load(f)


def test_config_summary_shows_grad_clip(capsys):
    from orbit.cli.commands.new import _print_config_summary

    base = {
        "name": "x", "dataset": "xor", "model": [{"type": "Linear", "in_features": 2, "neurons": 1}],
        "loss": "MSE", "optimizer": "SGD", "learning_rate": 0.1, "epochs": 1,
    }
    _print_config_summary(dict(base, grad_clip=2.5))
    _print_config_summary(base)

    out = capsys.readouterr().out
    assert "Grad clip" in out
    assert "2.5" in out
    assert "off" in out


def test_create_experiment_prefills_prompts_from_settings_defaults(tmp_path, monkeypatch):
    from orbit.settings import load_settings, save_settings, set_value

    settings = load_settings()
    for key, value in [
        ("defaults.optimizer", "Adam"),
        ("defaults.adam_learning_rate", "0.005"),
        ("defaults.batch_size", "16"),
        ("defaults.epochs", "250"),
        ("defaults.test_split", "0.25"),
        ("defaults.normalize", "standard"),
        ("defaults.grad_clip", "2"),
    ]:
        settings = set_value(settings, key, value)
    save_settings(settings)

    monkeypatch.setattr("orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name)
    defaults_seen = {}

    class Answer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    def fake_text(message, default="", **kwargs):
        defaults_seen[message] = default
        # Accept the pre-filled value, like pressing enter.
        canned = {"Experiment name:": "prefilled", "  neurons (size of this layer's output):": "1",
                  "Seed (blank = random):": "7"}
        return Answer(canned.get(message, default))

    layer_answers = iter(["Linear", "Done"])

    def fake_select(message, choices, default=None, **kwargs):
        defaults_seen[message] = default
        if message == "Add a layer:":
            return Answer(next(layer_answers))
        canned = {"Dataset:": "xor", "Loss:": "MSE", "Track accuracy?": "No (not tracked)"}
        return Answer(canned.get(message, default))

    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "select", fake_select)

    config_path = create_experiment()

    with open(config_path) as f:
        config = json.load(f)
    assert defaults_seen["Optimizer:"] == "Adam"
    assert defaults_seen["Normalize inputs?"] == "standard"
    assert config["optimizer"] == "Adam"
    assert config["learning_rate"] == 0.005
    assert config["batch_size"] == 16
    assert config["epochs"] == 250
    assert config["test_split"] == 0.25
    assert config["normalize"] == "standard"
    assert config["grad_clip"] == 2.0


def test_create_experiment_without_settings_keeps_the_old_blank_defaults(tmp_path, monkeypatch):
    # No settings file: test split and grad clip stay blank (off), as before
    # settings existed.
    monkeypatch.setattr("orbit.cli.commands.new.experiment_dir", lambda name: tmp_path / name)
    seen = {}

    class Answer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    def fake_text(message, default="", **kwargs):
        seen[message] = default
        return Answer({"Experiment name:": "x", "  neurons (size of this layer's output):": "1"}.get(message, default))

    selects = iter(["xor", "Linear", "Done", "MSE", "No (not tracked)", "SGD", "none"])
    monkeypatch.setattr(questionary, "text", fake_text)
    monkeypatch.setattr(questionary, "select", lambda *a, **k: Answer(next(selects)))

    create_experiment()

    assert seen["Test split fraction (0-1, blank = no split):"] == ""
    assert seen[GRAD_CLIP_PROMPT] == ""
    assert seen["Learning rate:"] == "0.1"
