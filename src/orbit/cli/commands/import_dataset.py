import csv
import pathlib
import shutil
from typing import Dict, Optional

import numpy as np
import questionary

from orbit.core.columns import (
    COLUMN_TYPES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_VOCAB_SIZE,
    column_width,
    detect_column_type,
    encode_images,
    fit_column,
    is_numeric,
)
from orbit.storage import FEATURES_FILENAME, dataset_dir, save_dataset_manifest
from orbit.ui import PROMPT_STYLE, info, success

# A numeric target with at most this many distinct integer values is
# probably a class label (e.g. a digit 0-9), so import suggests the flag that
# makes it one.
CLASS_LABEL_HINT_MAX_VALUES = 20

# --- top-level command --------------------------------------------------------

def import_dataset(
    csv_path: str,
    name: Optional[str] = None,
    target_columns: Optional[list] = None,
    column_types: Optional[Dict[str, str]] = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
    assume_yes: bool = False,
) -> pathlib.Path:
    """
    Register a CSV file as a named dataset under <workspace>/datasets/<name>/,
    so it can be picked in orbit new / referenced in a hand-edited
    experiment.json the same way "xor" is today. The CSV itself is copied
    alongside the manifest (not referenced by its original path), so the
    dataset stays reproducible even if the source file moves or changes.

    Every column gets a type (numeric, categorical, text, image - see
    core/columns.py): numeric when every cell is a number, otherwise a
    detected guess the user confirms or changes per column. column_types
    ({column: type}, from --column-type) sets types up front - including
    turning a numeric column into categorical, e.g. a 0-9 digit label - and
    assume_yes accepts the detected types without asking. What each type
    needs to encode a cell (categories, vocabulary, image size) is stored in
    dataset.json, and image columns are decoded once into features.npz.
    """
    csv_path = pathlib.Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No such CSV file: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    if header is None or len(header) < 2:
        raise ValueError(
            f"{csv_path} needs a header row with at least 2 columns "
            "(at least 1 input and 1 output)"
        )
    if not rows:
        raise ValueError(f"{csv_path} has a header but no data rows")
    if any(len(row) != len(header) for row in rows):
        raise ValueError(f"{csv_path}: every row needs {len(header)} cells, like the header")

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

    values = {column: [row[i] for row in rows] for i, column in enumerate(header)}
    types = _column_types(header, values, column_types or {}, assume_yes)
    _check_targets(target_columns, types)

    specs = {}
    for column in header:
        spec = fit_column(types[column], values[column], vocab_size=vocab_size, image_size=image_size)
        if spec is not None:
            specs[column] = spec

    exp_dir = dataset_dir(name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv_path, exp_dir / "data.csv")

    images = {
        column: encode_images(values[column], csv_path.parent, spec["size"], column).astype(np.float32)
        for column, spec in specs.items() if spec["type"] == "image"
    }
    if images:
        np.savez_compressed(exp_dir / FEATURES_FILENAME, **images)

    manifest = {"input_columns": input_columns, "output_columns": target_columns}
    if specs:
        manifest["columns"] = specs
    save_dataset_manifest(name, manifest)

    n_inputs = sum(column_width(specs.get(c)) for c in input_columns)
    target_spec = specs.get(target_columns[0])
    if len(target_columns) == 1 and target_spec is not None:
        outputs = f"{len(target_spec['categories'])} classes"
    else:
        outputs = f"{len(target_columns)} output feature(s)"
    success(f"Imported {name!r}: {n_inputs} input feature(s), {outputs}, {len(rows)} row(s)")
    _hint_class_label(target_columns, types, values)
    return exp_dir


def _column_types(header, values, overrides, assume_yes) -> Dict[str, str]:
    """
    Each column's type: an override wins; otherwise the detected type, which
    the user confirms or changes for each non-numeric column (numeric needs
    no question - its cells are already numbers).
    """
    unknown = [c for c in overrides if c not in header]
    if unknown:
        raise ValueError(f"--column-type names a column that isn't in the CSV: {unknown!r}")
    for column, column_type in overrides.items():
        if column_type not in COLUMN_TYPES:
            raise ValueError(
                f"Unknown column type {column_type!r} for {column!r} (valid: {', '.join(COLUMN_TYPES)})"
            )
        if column_type == "numeric" and not is_numeric(values[column]):
            raise ValueError(f"Column {column!r} can't be numeric: it has non-numeric cells")

    types = {}
    for column in header:
        if column in overrides:
            types[column] = overrides[column]
            continue
        detected = detect_column_type(values[column])
        if detected != "numeric" and not assume_yes:
            choices = [t for t in COLUMN_TYPES if t != "numeric"]
            detected = questionary.select(
                f"Column {column!r} isn't numeric (e.g. {_example(values[column])}). Treat it as:",
                choices=choices, default=detected, style=PROMPT_STYLE,
            ).ask()
        types[column] = detected
    return types


def _example(column_values) -> str:
    example = next((v for v in column_values if v.strip()), "")
    return repr(example if len(example) <= 40 else example[:37] + "...")


def _check_targets(target_columns, types) -> None:
    for column in target_columns:
        if types[column] in ("text", "image"):
            raise ValueError(
                f"Target column {column!r} is {types[column]} - targets must be numeric or categorical"
            )
    if len(target_columns) > 1 and any(types[c] == "categorical" for c in target_columns):
        raise ValueError("A categorical target must be the only target column")


def _hint_class_label(target_columns, types, values) -> None:
    if len(target_columns) != 1 or types[target_columns[0]] != "numeric":
        return
    column = target_columns[0]
    distinct = {float(v) for v in values[column]}
    if 2 < len(distinct) <= CLASS_LABEL_HINT_MAX_VALUES and all(v.is_integer() for v in distinct):
        info(
            f"Tip: {column!r} looks like a class label ({len(distinct)} distinct whole numbers). "
            f"To train a classifier on it, re-import with --column-type {column}=categorical"
        )
