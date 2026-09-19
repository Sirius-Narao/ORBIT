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
