import pathlib
import json

DATASETS_ROOT = pathlib.Path(".orbits/datasets")

def dataset_dir(name: str, root: pathlib.Path = DATASETS_ROOT) -> pathlib.Path:
    return root/name

def save_dataset_manifest(name: str, manifest: dict, root: pathlib.Path = DATASETS_ROOT) -> None:
    path = dataset_dir(name, root=root) / "dataset.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

def load_dataset_manifest(name: str, root: pathlib.Path = DATASETS_ROOT) -> dict:
    path = dataset_dir(name, root=root) / "dataset.json"
    with open(path) as f:
        return json.load(f)

def dataset_exists(name: str, root: pathlib.Path = DATASETS_ROOT) -> bool:
    return dataset_dir(name, root=root).is_dir()

def list_imported_dataset_names(root: pathlib.Path = DATASETS_ROOT) -> list:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())
