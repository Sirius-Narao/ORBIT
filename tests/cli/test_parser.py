import pathlib

import pytest
import orbit.cli.parser as parser_module


def test_init_dispatches_to_init_project(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "init_project", lambda: calls.append("called"))
    monkeypatch.setattr("sys.argv", ["orbit", "init"])

    parser_module.main()

    assert calls == ["called"]


def test_new_dispatches_to_create_experiment(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "create_experiment", lambda: calls.append("called"))
    monkeypatch.setattr("sys.argv", ["orbit", "new"])

    parser_module.main()

    assert calls == ["called"]


def test_run_dispatches_to_run_experiment_with_name(monkeypatch, capsys):
    calls = []

    class FakeResults:
        final_loss = 0.1234
        loss_history = [1.0, 0.5, 0.1234]  # a healthy, clearly-converging run
        test_loss = None
        test_accuracy = None

    def fake_run_experiment(name):
        calls.append(name)
        return FakeResults()

    monkeypatch.setattr(parser_module, "run_experiment", fake_run_experiment)
    monkeypatch.setattr("sys.argv", ["orbit", "run", "xor_mlp_01"])

    parser_module.main()

    assert calls == ["xor_mlp_01"]
    assert "0.1234" in capsys.readouterr().out


def test_run_reports_test_loss_and_accuracy_when_present(monkeypatch, capsys):
    class FakeResults:
        final_loss = 0.1234
        loss_history = [1.0, 0.5, 0.1234]
        test_loss = 0.2345
        test_accuracy = 0.875
        hyperparams = {}

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "split_model"])

    parser_module.main()

    out = capsys.readouterr().out
    assert "Test loss: 0.2345" in out
    assert "Test accuracy: 87.50%" in out


def test_run_omits_test_loss_line_when_absent(monkeypatch, capsys):
    class FakeResults:
        final_loss = 0.1234
        loss_history = [1.0, 0.5, 0.1234]
        test_loss = None
        test_accuracy = None

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "no_split_model"])

    parser_module.main()

    assert "Test loss" not in capsys.readouterr().out


def test_run_warns_when_loss_barely_improves(monkeypatch, capsys):
    class FakeResults:
        final_loss = 0.2389
        loss_history = [0.2448] * 10 + [0.2389]  # ~2% improvement - a stall
        test_loss = None
        test_accuracy = None

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "underpowered_model"])

    parser_module.main()

    assert "Warning" in capsys.readouterr().out


def test_run_does_not_warn_on_a_healthy_run(monkeypatch, capsys):
    class FakeResults:
        final_loss = 0.05
        loss_history = [1.0, 0.5, 0.2, 0.05]  # well over the 5% threshold
        test_loss = None
        test_accuracy = None

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "healthy_model"])

    parser_module.main()

    assert "Warning" not in capsys.readouterr().out


def test_run_does_not_warn_on_single_epoch_run(monkeypatch, capsys):
    # Not enough history to judge "improvement" at all - must not crash on
    # a one-element loss_history (would divide fine, but there's no trend).
    class FakeResults:
        final_loss = 0.5
        loss_history = [0.5]
        test_loss = None
        test_accuracy = None

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "one_epoch_model"])

    parser_module.main()

    assert "Warning" not in capsys.readouterr().out


def test_train_dispatches_to_train_experiment_with_name(monkeypatch, capsys):
    calls = []

    class FakeResults:
        final_loss = 0.1234
        loss_history = [1.0, 0.5, 0.1234]

    def fake_train_experiment(name):
        calls.append(name)
        return FakeResults()

    monkeypatch.setattr(parser_module, "train_experiment", fake_train_experiment)
    monkeypatch.setattr("sys.argv", ["orbit", "train", "xor_mlp_01"])

    parser_module.main()

    assert calls == ["xor_mlp_01"]
    assert "0.1234" in capsys.readouterr().out


def test_train_warns_when_loss_barely_improves(monkeypatch, capsys):
    class FakeResults:
        final_loss = 0.2389
        loss_history = [0.2448] * 10 + [0.2389]  # ~2% improvement - a stall

    monkeypatch.setattr(parser_module, "train_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "train", "underpowered_model"])

    parser_module.main()

    assert "Warning" in capsys.readouterr().out


def test_test_dispatches_to_test_experiment_with_name(monkeypatch, capsys):
    calls = []

    class FakeResults:
        test_loss = 0.2345
        test_accuracy = 0.875
        hyperparams = {}

    def fake_test_experiment(name):
        calls.append(name)
        return FakeResults()

    monkeypatch.setattr(parser_module, "test_experiment", fake_test_experiment)
    monkeypatch.setattr("sys.argv", ["orbit", "test", "split_model"])

    parser_module.main()

    assert calls == ["split_model"]
    out = capsys.readouterr().out
    assert "Test loss: 0.2345" in out
    assert "Test accuracy: 87.50%" in out


def test_test_prints_nothing_extra_when_test_experiment_returns_none(monkeypatch, capsys):
    monkeypatch.setattr(parser_module, "test_experiment", lambda name: None)
    monkeypatch.setattr("sys.argv", ["orbit", "test", "not_trained_model"])

    parser_module.main()

    assert "Test loss" not in capsys.readouterr().out


def test_import_dispatches_to_import_dataset_with_args(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "import_dataset",
        lambda csv_path, name=None, target_columns=None: calls.append(
            (csv_path, name, target_columns)
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["orbit", "import", "data.csv", "--name", "housing", "--target", "price"],
    )

    parser_module.main()

    assert calls == [("data.csv", "housing", ["price"])]


def test_import_dispatches_with_defaults_when_optional_flags_omitted(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "import_dataset",
        lambda csv_path, name=None, target_columns=None: calls.append(
            (csv_path, name, target_columns)
        ),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "import", "data.csv"])

    parser_module.main()

    assert calls == [("data.csv", None, None)]


def test_plotloss_dispatches_to_plot_experiment_with_name(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "plot_experiment",
        lambda name, log_scale=False: calls.append((name, log_scale)),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "plotloss", "xor_mlp_01"])

    parser_module.main()

    assert calls == [("xor_mlp_01", False)]


def test_plotloss_dispatches_with_logscale_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "plot_experiment",
        lambda name, log_scale=False: calls.append((name, log_scale)),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "plotloss", "xor_mlp_01", "--logscale"])

    parser_module.main()

    assert calls == [("xor_mlp_01", True)]


def test_plot_dispatches_with_names(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "plot_experiments",
        lambda names, is_all=False, log_scale=False, metrics=None: calls.append(
            (names, is_all, log_scale, metrics)
        ),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "plot", "exp_a", "exp_b"])

    parser_module.main()

    assert calls == [(["exp_a", "exp_b"], False, False, None)]


def test_plot_dispatches_with_all_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "plot_experiments",
        lambda names, is_all=False, log_scale=False, metrics=None: calls.append(
            (names, is_all, log_scale, metrics)
        ),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "plot", "--all"])

    parser_module.main()

    assert calls == [([], True, False, None)]


def test_plot_dispatches_with_logscale_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "plot_experiments",
        lambda names, is_all=False, log_scale=False, metrics=None: calls.append(
            (names, is_all, log_scale, metrics)
        ),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "plot", "exp_a", "--logscale"])

    parser_module.main()

    assert calls == [(["exp_a"], False, True, None)]


def test_plot_dispatches_with_metrics_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "plot_experiments",
        lambda names, is_all=False, log_scale=False, metrics=None: calls.append(
            (names, is_all, log_scale, metrics)
        ),
    )
    monkeypatch.setattr(
        "sys.argv", ["orbit", "plot", "exp_a", "--metrics", "loss", "test_accuracy"]
    )

    parser_module.main()

    assert calls == [(["exp_a"], False, False, ["loss", "test_accuracy"])]


def test_plot_rejects_unknown_metric_choice(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["orbit", "plot", "exp_a", "--metrics", "bogus"])

    try:
        parser_module.main()
        assert False, "expected SystemExit"
    except SystemExit:
        pass

    assert "invalid choice" in capsys.readouterr().err


def test_compare_dispatches_with_plotloss_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "compare_experiments",
        lambda names, is_all=False, plot_loss=False, log_scale=False, by=None, test_only=False: calls.append(
            (names, is_all, plot_loss, log_scale)
        ),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "compare", "exp_a", "exp_b", "--plotloss"])

    parser_module.main()

    assert calls == [(["exp_a", "exp_b"], False, True, False)]


def test_compare_dispatches_with_plotloss_and_logscale_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(
        parser_module,
        "compare_experiments",
        lambda names, is_all=False, plot_loss=False, log_scale=False, by=None, test_only=False: calls.append(
            (names, is_all, plot_loss, log_scale)
        ),
    )
    monkeypatch.setattr("sys.argv", ["orbit", "compare", "exp_a", "exp_b", "--plotloss", "--logscale"])

    parser_module.main()

    assert calls == [(["exp_a", "exp_b"], False, True, True)]


def test_missing_command_opens_the_repl(monkeypatch):
    calls = []
    monkeypatch.setattr("orbit.cli.repl.repl", lambda: calls.append("called"))
    monkeypatch.setattr("sys.argv", ["orbit"])

    parser_module.main()

    assert calls == ["called"]


def test_run_without_name_configures_then_runs(monkeypatch, capsys):
    calls = []

    class FakeResults:
        final_loss = 0.0456
        loss_history = [1.0, 0.5, 0.0456]  # a healthy, clearly-converging run
        test_loss = None
        test_accuracy = None

    def fake_create_experiment():
        calls.append("create_experiment")
        return pathlib.Path(".orbits/experiments/auto_named/experiment.json")

    def fake_run_experiment(name):
        calls.append(("run_experiment", name))
        return FakeResults()

    monkeypatch.setattr(parser_module, "create_experiment", fake_create_experiment)
    monkeypatch.setattr(parser_module, "run_experiment", fake_run_experiment)
    monkeypatch.setattr("sys.argv", ["orbit", "run"])

    parser_module.main()

    assert calls == ["create_experiment", ("run_experiment", "auto_named")]
    assert "0.0456" in capsys.readouterr().out


def test_run_reports_test_r2_as_a_plain_number(monkeypatch, capsys):
    class FakeResults:
        final_loss = 0.1234
        loss_history = [1.0, 0.5, 0.1234]
        test_loss = 0.2345
        test_accuracy = 0.8123
        hyperparams = {"task": "regression_r2"}

    monkeypatch.setattr(parser_module, "run_experiment", lambda name: FakeResults())
    monkeypatch.setattr("sys.argv", ["orbit", "run", "regression_model"])

    parser_module.main()

    out = capsys.readouterr().out
    assert "Test R²: 0.8123" in out
    assert "%" not in out


@pytest.mark.parametrize(
    "argv, fn_name, expected",
    [
        (["create", "sw", "--base", "b"], "create_sweep", (("sw", "b"), {})),
        (["start", "sw"], "start_sweep", (("sw",), {})),
        (["status", "sw"], "sweep_status", (("sw",), {})),
        (["compare", "sw", "--by", "test_loss"], "compare_sweep", (("sw",), {"by": ["test_loss"], "all_metrics": False, "test_only": False})),
        (
            ["compare", "sw", "--by", "test_loss", "test_accuracy"],
            "compare_sweep",
            (("sw",), {"by": ["test_loss", "test_accuracy"], "all_metrics": False, "test_only": False}),
        ),
        (["compare", "sw", "--by"], "compare_sweep", (("sw",), {"by": [], "all_metrics": False, "test_only": False})),
        (["compare", "sw"], "compare_sweep", (("sw",), {"by": None, "all_metrics": False, "test_only": False})),
        (["compare", "sw", "--all"], "compare_sweep", (("sw",), {"by": None, "all_metrics": True, "test_only": False})),
        (["compare", "sw", "--test"], "compare_sweep", (("sw",), {"by": None, "all_metrics": False, "test_only": True})),
        (["export", "sw", "--output", "x.csv"], "export_sweep", (("sw",), {"output": "x.csv"})),
        (
            ["plot", "sw", "--logscale", "--metrics", "loss"],
            "plot_sweep",
            (("sw",), {"log_scale": True, "metrics": ["loss"]}),
        ),
    ],
)
def test_sweep_subcommands_dispatch(monkeypatch, argv, fn_name, expected):
    calls = []
    monkeypatch.setattr(parser_module, fn_name, lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr("sys.argv", ["orbit", "sweep", *argv])

    parser_module.main()

    assert calls == [expected]


@pytest.mark.parametrize(
    "argv, expected_by",
    [
        (["compare", "a", "b"], None),
        (["compare", "a", "b", "--by", "test_loss", "test_accuracy", "duration_seconds"],
         ["test_loss", "test_accuracy", "duration_seconds"]),
        (["compare", "--all", "--by"], []),
    ],
)
def test_compare_dispatches_by_metrics(monkeypatch, argv, expected_by):
    calls = []
    monkeypatch.setattr(parser_module, "compare_experiments", lambda *a, **k: calls.append(k["by"]))
    monkeypatch.setattr("sys.argv", ["orbit", *argv])

    parser_module.main()

    assert calls == [expected_by]


def test_sweep_compare_all_and_by_are_mutually_exclusive(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["orbit", "sweep", "compare", "sw", "--all", "--by", "test_loss"])

    with pytest.raises(SystemExit):
        parser_module.main()

    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("argv, expected", [(["list"], False), (["list", "--runs"], True)])
def test_list_dispatches_show_runs(monkeypatch, argv, expected):
    calls = []
    monkeypatch.setattr(parser_module, "list_experiments", lambda **k: calls.append(k))
    monkeypatch.setattr("sys.argv", ["orbit", *argv])

    parser_module.main()

    assert calls == [{"show_runs": expected}]


@pytest.mark.parametrize("argv, expected_yes", [(["delete", "sw"], False), (["delete", "sw", "--yes"], True)])
def test_sweep_delete_dispatches(monkeypatch, argv, expected_yes):
    calls = []
    monkeypatch.setattr(parser_module, "delete_sweep", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setattr("sys.argv", ["orbit", "sweep", *argv])

    parser_module.main()

    assert calls == [(("sw",), {"yes": expected_yes})]


def test_compare_dispatches_test_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "compare_experiments", lambda *a, **k: calls.append(k))
    monkeypatch.setattr("sys.argv", ["orbit", "compare", "--all", "--test"])

    parser_module.main()

    assert calls[0]["test_only"] is True
    assert calls[0]["by"] is None


@pytest.mark.parametrize(
    "argv",
    [
        ["compare", "--all", "--test", "--by", "test_loss"],
        ["sweep", "compare", "sw", "--test", "--by", "test_loss"],
        ["sweep", "compare", "sw", "--test", "--all"],
    ],
)
def test_test_flag_is_exclusive_with_by_and_all(monkeypatch, capsys, argv):
    monkeypatch.setattr("sys.argv", ["orbit", *argv])

    with pytest.raises(SystemExit):
        parser_module.main()

    assert "not allowed with argument" in capsys.readouterr().err


def test_network_dispatches_with_its_options(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "network_experiment", lambda name, **kwargs: calls.append((name, kwargs)))
    monkeypatch.setattr("sys.argv", ["orbit", "network", "xor", "--sample", "2", "--output", "net.png", "--show"])

    parser_module.main()

    assert calls == [("xor", {"sample": 2, "output": "net.png", "show": True})]


def test_network_show_is_refused_inside_the_repl(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(parser_module, "network_experiment", lambda name, **kwargs: calls.append(kwargs["show"]))
    args = parser_module.build_parser().parse_args(["network", "xor", "--show"])

    parser_module._dispatch(args, in_repl=True)

    assert calls == [False]
    assert "not available in the REPL" in capsys.readouterr().out


def test_animate_dispatches_with_its_options(monkeypatch):
    calls = []
    monkeypatch.setattr(parser_module, "animate_experiment", lambda name, **kwargs: calls.append((name, kwargs)))
    monkeypatch.setattr("sys.argv", ["orbit", "animate", "xor", "--mp4", "--fps", "5", "--logscale"])

    parser_module.main()

    assert calls == [("xor", {
        "sample": None, "video_format": "mp4", "fps": 5, "output": None, "log_scale": True, "show": False,
    })]
