import csv
import pathlib
import shutil
from typing import Optional

import questionary

from orbit.storage import dataset_dir, save_dataset_manifest
from orbit.ui import PROMPT_STYLE, success

# --- top-level command --------------------------------------------------------

def import_dataset(
    csv_path: str,
    name: Optional[str] = None,
    target_columns: Optional[list] = None,
) -> pathlib.Path:
    """
    Register a CSV file as a named dataset under .orbits/datasets/<name>/,
    so it can be picked in orbit new / referenced in a hand-edited
    experiment.json the same way "xor" is today. The CSV itself is copied
    alongside the manifest (not referenced by its original path), so the
    dataset stays reproducible even if the source file moves or changes.
    """
    csv_path = pathlib.Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No such CSV file: {csv_path}")

    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), None)

    if header is None or len(header) < 2:
        raise ValueError(
            f"{csv_path} needs a header row with at least 2 columns "
            "(at least 1 input and 1 output)"
        )

    if name is None:
        name = csv_path.stem

    if target_columns is None:
        target_columns = questionary.checkbox(
            "Select the output/target column(s):", choices=header, style=PROMPT_STYLE
        ).ask()

    if not target_columns:
        raise ValueError("At least one output/target column must be selected")

    unknown = [c for c in target_columns if c not in header]
    if unknown:
        raise ValueError(f"Not a column in {csv_path}: {unknown!r}")

    input_columns = [c for c in header if c not in target_columns]
    if not input_columns:
        raise ValueError("At least one input column must remain (not selected as output)")

    with open(csv_path, newline="") as f:
        row_count = sum(1 for _ in csv.reader(f)) - 1  # minus the header row

    exp_dir = dataset_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv_path, exp_dir / "data.csv")

    save_dataset_manifest(name, {
        "input_columns": input_columns,
        "output_columns": target_columns,
    })

    success(
        f"Imported {name!r}: {len(input_columns)} input feature(s), "
        f"{len(target_columns)} output feature(s), {row_count} row(s)"
    )
    return exp_dir
