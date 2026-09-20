from orbit.cli.commands.init import init_project


def test_init_project_creates_experiments_and_datasets_dirs(tmp_path):
    root = tmp_path / "myproj"
    experiments_root = root / "experiments"
    datasets_root = root / "datasets"

    init_project(experiments_root=experiments_root, datasets_root=datasets_root)

    assert experiments_root.is_dir()
    assert datasets_root.is_dir()


def test_init_project_reports_success_on_first_run(tmp_path, capsys):
    root = tmp_path / "myproj"

    init_project(experiments_root=root / "experiments", datasets_root=root / "datasets")

    assert "Initialized" in capsys.readouterr().out


def test_init_project_warns_when_already_initialized(tmp_path, capsys):
    root = tmp_path / "myproj"
    experiments_root = root / "experiments"
    datasets_root = root / "datasets"

    init_project(experiments_root=experiments_root, datasets_root=datasets_root)
    capsys.readouterr()

    init_project(experiments_root=experiments_root, datasets_root=datasets_root)

    assert "already initialized" in capsys.readouterr().out.lower()


def test_init_project_does_not_clobber_existing_experiments(tmp_path):
    root = tmp_path / "myproj"
    experiments_root = root / "experiments"
    datasets_root = root / "datasets"
    experiments_root.mkdir(parents=True)
    (experiments_root / "existing_experiment").mkdir()

    init_project(experiments_root=experiments_root, datasets_root=datasets_root)

    assert (experiments_root / "existing_experiment").is_dir()
