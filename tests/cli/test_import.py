import json

import pytest
import questionary

import orbit.cli.commands.import_dataset as import_dataset_module
from orbit.cli.commands.import_dataset import import_dataset
from orbit.storage.datasets import save_dataset_manifest as real_save_dataset_manifest


def patch_dataset_storage(monkeypatch, tmp_path):
    """
    import_dataset() writes through dataset_dir()/save_dataset_manifest()
    imported into its own module namespace - patch both there (not the
    real .orbits/datasets/ location) to redirect writes under tmp_path,
    the same way tests/cli/test_new.py patches experiment_dir.
    """
    root = tmp_path / "datasets"
    monkeypatch.setattr(import_dataset_module, "dataset_dir", lambda name: root / name)
    monkeypatch.setattr(
        import_dataset_module,
        "save_dataset_manifest",
        lambda name, manifest: real_save_dataset_manifest(name, manifest, root=root),
    )
    return root


def write_csv(path, text):
    path.write_text(text)
    return path


def test_import_dataset_with_explicit_target_needs_no_prompt(tmp_path, monkeypatch):
    root = patch_dataset_storage(monkeypatch, tmp_path)
    csv_path = write_csv(tmp_path / "housing.csv", "sqft,bedrooms,price\n1000,2,200000\n1500,3,300000\n")

    result_dir = import_dataset(str(csv_path), name="housing", target_columns=["price"])

    assert result_dir == root / "housing"
    manifest = json.loads((result_dir / "dataset.json").read_text())
    assert manifest == {"input_columns": ["sqft", "bedrooms"], "output_columns": ["price"]}
    assert (result_dir / "data.csv").read_text() == csv_path.read_text()


def test_import_dataset_defaults_name_to_file_stem(tmp_path, monkeypatch):
    root = patch_dataset_storage(monkeypatch, tmp_path)
    csv_path = write_csv(tmp_path / "iris.csv", "a,b,c\n1,2,3\n")

    result_dir = import_dataset(str(csv_path), target_columns=["c"])

    assert result_dir == root / "iris"


def test_import_dataset_prompts_for_target_when_not_given(tmp_path, monkeypatch):
    patch_dataset_storage(monkeypatch, tmp_path)
    csv_path = write_csv(tmp_path / "housing.csv", "sqft,bedrooms,price\n1000,2,200000\n")

    class FakeAnswer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    monkeypatch.setattr(questionary, "checkbox", lambda *a, **k: FakeAnswer(["price"]))

    result_dir = import_dataset(str(csv_path), name="housing")

    manifest = json.loads((result_dir / "dataset.json").read_text())
    assert manifest["output_columns"] == ["price"]
    assert manifest["input_columns"] == ["sqft", "bedrooms"]


def test_import_dataset_rejects_unknown_target_column(tmp_path, monkeypatch):
    patch_dataset_storage(monkeypatch, tmp_path)
    csv_path = write_csv(tmp_path / "housing.csv", "sqft,bedrooms,price\n1000,2,200000\n")

    with pytest.raises(ValueError):
        import_dataset(str(csv_path), name="housing", target_columns=["not_a_column"])


def test_import_dataset_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_dataset(str(tmp_path / "does_not_exist.csv"), name="x", target_columns=["y"])


def test_import_dataset_rejects_all_columns_as_target(tmp_path, monkeypatch):
    patch_dataset_storage(monkeypatch, tmp_path)
    csv_path = write_csv(tmp_path / "housing.csv", "sqft,price\n1000,200000\n")

    with pytest.raises(ValueError):
        import_dataset(str(csv_path), name="housing", target_columns=["sqft", "price"])
