from orbit.storage import (
    save_dataset_manifest,
    load_dataset_manifest,
    dataset_exists,
    list_imported_dataset_names,
)


def test_save_then_load_manifest_round_trips(tmp_path):
    manifest = {"input_columns": ["sqft", "bedrooms"], "output_columns": ["price"]}

    save_dataset_manifest("housing", manifest, root=tmp_path)
    loaded = load_dataset_manifest("housing", root=tmp_path)

    assert loaded == manifest


def test_save_creates_parent_directories(tmp_path):
    root = tmp_path / "nested" / "datasets"
    assert not root.exists()

    save_dataset_manifest("housing", {"input_columns": ["x"], "output_columns": ["y"]}, root=root)

    assert (root / "housing" / "dataset.json").exists()


def test_dataset_exists_true_after_save(tmp_path):
    assert not dataset_exists("housing", root=tmp_path)

    save_dataset_manifest("housing", {"input_columns": ["x"], "output_columns": ["y"]}, root=tmp_path)

    assert dataset_exists("housing", root=tmp_path)


def test_dataset_exists_false_for_unknown_name(tmp_path):
    assert not dataset_exists("unknown", root=tmp_path)


def test_list_imported_dataset_names_reflects_disk_state(tmp_path):
    assert list_imported_dataset_names(root=tmp_path) == []

    save_dataset_manifest("housing", {"input_columns": ["x"], "output_columns": ["y"]}, root=tmp_path)
    save_dataset_manifest("iris", {"input_columns": ["x"], "output_columns": ["y"]}, root=tmp_path)

    assert list_imported_dataset_names(root=tmp_path) == ["housing", "iris"]


def test_list_imported_dataset_names_empty_when_root_missing(tmp_path):
    missing_root = tmp_path / "does_not_exist"

    assert list_imported_dataset_names(root=missing_root) == []
