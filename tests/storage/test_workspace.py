import pathlib

from orbit.settings import load_settings, save_settings, set_value
from orbit.storage import dataset_dir, experiment_dir
from orbit.storage.workspace import comparisons_root, datasets_root, experiments_root, sweeps_root


def test_roots_default_to_dot_orbits_without_a_config():
    assert experiments_root() == pathlib.Path(".orbits/experiments")
    assert datasets_root() == pathlib.Path(".orbits/datasets")
    assert sweeps_root() == pathlib.Path(".orbits/sweeps")
    assert comparisons_root() == pathlib.Path(".orbits/comparisons")


def test_an_explicit_root_wins(tmp_path):
    assert experiments_root(tmp_path) == tmp_path


def test_roots_follow_the_configured_workspace_at_call_time(tmp_path):
    # Resolved when called, not frozen at import time, so changing the
    # settings is picked up by functions that were already imported.
    save_settings(set_value(load_settings(), "workspace", str(tmp_path / "ws")))

    assert experiment_dir("a") == tmp_path / "ws" / "experiments" / "a"
    assert dataset_dir("d") == tmp_path / "ws" / "datasets" / "d"
    assert sweeps_root() == tmp_path / "ws" / "sweeps"
