import json
import pathlib

from orbit.storage.workspace import experiments_root, sweeps_root
from orbit.sweeps.config import (
    list_sweep_names,
    sweep_membership,
    sweeps_root_for,
)


def write_manifest(root, name, runs):
    d = root / name
    d.mkdir(parents=True)
    (d / "sweep.json").write_text(json.dumps({"name": name, "runs": [{"name": r, "params": {}} for r in runs]}))


def test_list_sweep_names_only_counts_dirs_with_a_manifest(tmp_path):
    write_manifest(tmp_path, "b_sweep", [])
    write_manifest(tmp_path, "a_sweep", [])
    (tmp_path / "plots_only").mkdir()
    (tmp_path / "stray.txt").write_text("x")

    assert list_sweep_names(tmp_path) == ["a_sweep", "b_sweep"]


def test_list_sweep_names_missing_root(tmp_path):
    assert list_sweep_names(tmp_path / "nope") == []


def test_sweep_membership_maps_runs_to_their_sweep(tmp_path):
    write_manifest(tmp_path, "s1", ["s1_001", "s1_002"])
    write_manifest(tmp_path, "s2", ["s2_001"])

    assert sweep_membership(tmp_path) == {"s1_001": "s1", "s1_002": "s1", "s2_001": "s2"}


def test_sweeps_root_for_matches_the_default_layout():
    assert sweeps_root_for(experiments_root()) == sweeps_root()
