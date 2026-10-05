import pathlib
import json
from typing import Optional
from orbit.storage.workspace import datasets_root

# Decoded image columns, written by orbit import next to data.csv: one
# (rows, height*width) array per image column, keyed by column name.
FEATURES_FILENAME = "features.npz"


def dataset_dir(name: str, root: Optional[pathlib.Path] = None) -> pathlib.Path:
    root = datasets_root(root)
    return root/name

def save_dataset_manifest(name: str, manifest: dict, root: Optional[pathlib.Path] = None) -> None:
    root = datasets_root(root)
    path = dataset_dir(name, root=root) / "dataset.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

def load_dataset_manifest(name: str, root: Optional[pathlib.Path] = None) -> dict:
    root = datasets_root(root)
    path = dataset_dir(name, root=root) / "dataset.json"
    with open(path) as f:
        return json.load(f)

def dataset_exists(name: str, root: Optional[pathlib.Path] = None) -> bool:
    root = datasets_root(root)
    return dataset_dir(name, root=root).is_dir()

def list_imported_dataset_names(root: Optional[pathlib.Path] = None) -> list:
    root = datasets_root(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
