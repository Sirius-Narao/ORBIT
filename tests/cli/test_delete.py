from orbit.cli.commands.delete import delete_experiments


def test_delete_experiments_removes_named_experiment(tmp_path, capsys):
    (tmp_path / "exp_a").mkdir(parents=True)
    (tmp_path / "exp_b").mkdir(parents=True)

    delete_experiments("exp_a", root=tmp_path)

    assert not (tmp_path / "exp_a").exists()
    assert (tmp_path / "exp_b").exists()
    assert "exp_a" in capsys.readouterr().out


def test_delete_experiments_missing_name_leaves_experiments_untouched(tmp_path, capsys):
    (tmp_path / "exp_a").mkdir(parents=True)

    delete_experiments("does_not_exist", root=tmp_path)

    assert (tmp_path / "exp_a").exists()
    assert "was not found" in capsys.readouterr().out


def test_delete_experiments_all_removes_every_experiment(tmp_path, capsys):
    (tmp_path / "exp_a").mkdir(parents=True)
    (tmp_path / "exp_b").mkdir(parents=True)

    delete_experiments(is_all=True, root=tmp_path)

    assert not (tmp_path / "exp_a").exists()
    assert not (tmp_path / "exp_b").exists()
    assert "2" in capsys.readouterr().out


def test_delete_experiments_empty_dir(tmp_path, capsys):
    delete_experiments("anything", root=tmp_path)

    assert "No experiments found" in capsys.readouterr().out


def test_delete_experiments_missing_root(tmp_path, capsys):
    delete_experiments("anything", root=tmp_path / "does_not_exist")

    assert "No experiments found" in capsys.readouterr().out


# --- sweeps ----------------------------------------------------------------------

import json


def write_sweep(project, sweep_name, run_names):
    sweep_dir = project / "sweeps" / sweep_name
    sweep_dir.mkdir(parents=True)
    runs = [{"name": run, "params": {}} for run in run_names]
    (sweep_dir / "sweep.json").write_text(json.dumps({"name": sweep_name, "runs": runs}))


def test_delete_a_sweep_run_warns_it_will_show_as_missing(tmp_path, capsys):
    experiments = tmp_path / "experiments"
    (experiments / "sw_001").mkdir(parents=True)
    write_sweep(tmp_path, "sw", ["sw_001"])

    delete_experiments("sw_001", root=experiments)
    out = capsys.readouterr().out

    assert not (experiments / "sw_001").exists()
    assert "belongs to sweep sw" in out
    assert "orbit sweep delete sw" in out
    # the sweep manifest itself is left alone
    assert (tmp_path / "sweeps" / "sw" / "sweep.json").exists()


def test_delete_a_standalone_experiment_has_no_sweep_warning(tmp_path, capsys):
    experiments = tmp_path / "experiments"
    (experiments / "exp_a").mkdir(parents=True)

    delete_experiments("exp_a", root=experiments)

    assert "belongs to sweep" not in capsys.readouterr().out


def test_delete_all_also_removes_sweeps(tmp_path, capsys):
    experiments = tmp_path / "experiments"
    (experiments / "sw_001").mkdir(parents=True)
    (experiments / "exp_a").mkdir(parents=True)
    write_sweep(tmp_path, "sw", ["sw_001"])
    write_sweep(tmp_path, "sw2", [])

    delete_experiments(is_all=True, root=experiments)
    out = capsys.readouterr().out

    assert list(experiments.iterdir()) == []
    assert not (tmp_path / "sweeps" / "sw").exists()
    assert not (tmp_path / "sweeps" / "sw2").exists()
    assert "Deleted 2 experiment(s). Deleted 2 sweep(s)." in out


def test_delete_all_with_only_sweeps_left_still_clears_them(tmp_path):
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    write_sweep(tmp_path, "sw", ["sw_001"])

    delete_experiments(is_all=True, root=experiments)

    assert not (tmp_path / "sweeps" / "sw").exists()


def test_delete_defaults_to_the_sweeps_dir_next_to_root(tmp_path):
    """Never the fixed .orbits/sweeps - a root elsewhere must not reach the real project."""
    from orbit.sweeps.config import sweeps_root_for

    assert sweeps_root_for(tmp_path / "experiments") == tmp_path / "sweeps"
