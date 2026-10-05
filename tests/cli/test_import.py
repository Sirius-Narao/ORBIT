import json

import numpy as np

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


# --- column types (imported, then loaded back through build_dataset) -----------

@pytest.fixture
def workspace(tmp_path):
    """A real workspace via the settings file, so import and build_dataset share it."""
    from orbit.settings import load_settings, save_settings, set_value

    save_settings(set_value(load_settings(), "workspace", str(tmp_path / "ws")))
    return tmp_path / "ws"


def test_categorical_input_and_target_round_trip(tmp_path, workspace):
    from orbit.core.config import build_dataset

    csv_path = write_csv(tmp_path / "pets.csv", "weight,color,label\n4,black,cat\n30,brown,dog\n5,brown,cat\n")

    import_dataset(str(csv_path), target_columns=["label"], assume_yes=True)
    dataset = build_dataset("pets")

    manifest = json.loads((workspace / "datasets" / "pets" / "dataset.json").read_text())
    assert manifest["columns"]["color"] == {"type": "categorical", "categories": ["black", "brown"]}
    assert dataset.num_classes == 2
    assert dataset.input_shape == 3  # weight + one-hot(black, brown)
    x, y = dataset[1]
    assert list(x) == [30.0, 0.0, 1.0]
    assert y.tolist() == [1]  # dog


def test_text_column_becomes_word_counts(tmp_path, workspace):
    from orbit.core.config import build_dataset

    csv_path = write_csv(
        tmp_path / "reviews.csv",
        "review,score\n"
        "\"a great great movie, truly\",1\n"
        "\"a dull and boring movie\",0\n",
    )

    import_dataset(str(csv_path), target_columns=["score"], vocab_size=3, assume_yes=True)
    dataset = build_dataset("reviews")

    # counts: a 2, great 2, movie 2 (then and/boring/dull/truly 1) -> top 3 alphabetically among ties
    manifest = json.loads((workspace / "datasets" / "reviews" / "dataset.json").read_text())
    assert manifest["columns"]["review"]["vocabulary"] == ["a", "great", "movie"]
    assert dataset.X.tolist() == [[1, 2, 1], [1, 0, 1]]


def test_image_column_is_decoded_once_at_import(tmp_path, workspace):
    from PIL import Image

    from orbit.core.config import build_dataset

    images = tmp_path / "imgs"
    images.mkdir()
    Image.fromarray(np.full((8, 8), 255, dtype=np.uint8)).save(images / "white.png")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "black.png")
    csv_path = write_csv(tmp_path / "shades.csv", "path,label\nimgs/white.png,1\nimgs/black.png,0\n")

    import_dataset(str(csv_path), target_columns=["label"], image_size=4, assume_yes=True)
    # Training must not need the original files anymore.
    for path in images.iterdir():
        path.unlink()
    dataset = build_dataset("shades")

    assert (workspace / "datasets" / "shades" / "features.npz").is_file()
    assert dataset.input_shape == 16
    assert np.allclose(dataset.X[0], 1.0)
    assert np.allclose(dataset.X[1], 0.0)


def test_non_numeric_columns_are_asked_about_with_the_detected_default(tmp_path, workspace, monkeypatch):
    asked = []

    class Answer:
        def __init__(self, value):
            self.value = value

        def ask(self):
            return self.value

    def fake_select(message, choices, default=None, **kwargs):
        asked.append((message.split("'")[1], default))
        return Answer("text")

    monkeypatch.setattr(questionary, "select", fake_select)
    csv_path = write_csv(tmp_path / "d.csv", "x,color,y\n1,red,0\n2,blue,1\n")

    import_dataset(str(csv_path), target_columns=["y"])

    assert asked == [("color", "categorical")]
    manifest = json.loads((workspace / "datasets" / "d" / "dataset.json").read_text())
    assert manifest["columns"]["color"]["type"] == "text"


def test_column_type_override_turns_a_numeric_label_into_classes(tmp_path, workspace):
    from orbit.core.config import build_dataset

    csv_path = write_csv(tmp_path / "digits.csv", "p0,p1,digit\n0,1,7\n1,0,2\n1,1,10\n")

    import_dataset(str(csv_path), target_columns=["digit"], column_types={"digit": "categorical"})
    dataset = build_dataset("digits")

    assert dataset.num_classes == 3
    assert dataset.Y[:, 0].tolist() == [1, 0, 2]  # categories sorted numerically: 2, 7, 10


def test_numeric_class_label_gets_a_hint(tmp_path, workspace, capsys):
    csv_path = write_csv(tmp_path / "d.csv", "a,label\n1,0\n2,1\n3,2\n")

    import_dataset(str(csv_path), target_columns=["label"])

    assert "--column-type label=categorical" in capsys.readouterr().out


@pytest.mark.parametrize("column_types, target, message", [
    ({"x": "nope"}, ["y"], "Unknown column type"),
    ({"missing": "text"}, ["y"], "isn't in the CSV"),
    ({"color": "numeric"}, ["y"], "can't be numeric"),
    ({"color": "text"}, ["color"], "targets must be numeric or categorical"),
    ({"color": "categorical"}, ["color", "y"], "only target column"),
])
def test_invalid_column_type_choices_are_rejected(tmp_path, workspace, column_types, target, message):
    csv_path = write_csv(tmp_path / "d.csv", "x,color,y\n1,red,0\n2,blue,1\n")

    with pytest.raises(ValueError, match=message):
        import_dataset(str(csv_path), target_columns=target, column_types=column_types, assume_yes=True)


def test_old_numeric_manifest_still_loads(tmp_path, workspace):
    from orbit.core.config import build_dataset

    csv_path = write_csv(tmp_path / "plain.csv", "a,b\n1,2\n3,4\n")
    import_dataset(str(csv_path), target_columns=["b"])

    manifest = json.loads((workspace / "datasets" / "plain" / "dataset.json").read_text())
    assert "columns" not in manifest
    assert build_dataset("plain").X.tolist() == [[1.0], [3.0]]
