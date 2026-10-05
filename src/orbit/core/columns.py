"""
Column types for imported CSV datasets, and how each becomes numbers.

ORBIT's models are MLPs: every sample must be a fixed-length vector of
floats. A CSV cell can hold other things, so each column has a type that
says how to turn its cells into numbers:

  numeric      the cell parsed as a float                           1 value
  categorical  one-hot over the column's categories                 k values
               ("red"/"green"/"blue" -> [0,1,0] for "green")
  text         bag of words: how often each vocabulary word         V values
               appears in the cell (word order is lost)
  image        a path to an image file (relative to the CSV),       H*W values
               loaded as grayscale, resized to H x W, scaled to [0, 1]

A CSV can't contain an image itself - it's plain text - so an image column
holds file paths. (A CSV that already has one numeric column per pixel, like
the common MNIST CSV, is just numeric columns and needs nothing special.)

What each type needs to encode a cell (the category list, the vocabulary,
the image size) is decided once by orbit import and stored in the dataset's
dataset.json under "columns", so every later load encodes identically. A
column missing from "columns" is numeric, which keeps manifests written
before column types existed loading exactly as before. Image pixels are
decoded once at import and stored in features.npz, so training never
re-reads the image files.

Note: categories and vocabulary are taken from the whole file at import,
before any train/test split. That's a mild, accepted leak (it reveals which
words/categories exist, not anything about the targets), unlike the
"normalize" statistics, which are fit on the training split only.

A categorical *target* column becomes class indices (see TensorDataset's
num_classes) rather than one-hot vectors - the format CrossEntropy and
accuracy_multiclass expect.
"""
import collections
import pathlib
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

COLUMN_TYPES = ("numeric", "categorical", "text", "image")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
DEFAULT_VOCAB_SIZE = 1000
DEFAULT_IMAGE_SIZE = 28

_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lowercase words and numbers: "It's GREAT, 10/10" -> ["it's", "great", "10", "10"]."""
    return _TOKEN.findall(text.lower())


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_numeric(values: Sequence[str]) -> bool:
    return all(_is_float(v) for v in values)


def detect_column_type(values: Sequence[str]) -> str:
    """
    A best guess, which orbit import shows and lets the user change:
    every cell a number -> numeric; every cell a file name with an image
    extension -> image (a missing file is then reported when the images are
    loaded, rather than silently guessing another type); free text (several
    words per cell on average, or mostly-unique multi-word cells) -> text;
    anything else -> categorical.
    """
    values = list(values)
    if is_numeric(values):
        return "numeric"
    non_empty = [v for v in values if v.strip()]
    if non_empty and all(v.strip().lower().endswith(IMAGE_EXTENSIONS) for v in non_empty):
        return "image"
    word_counts = [len(tokenize(v)) for v in non_empty]
    mean_words = sum(word_counts) / len(word_counts) if word_counts else 0
    unique_ratio = len(set(non_empty)) / len(non_empty) if non_empty else 0
    if mean_words >= 3 or (unique_ratio > 0.5 and mean_words >= 2):
        return "text"
    return "categorical"


# --- fitting (orbit import) ----------------------------------------------------

def fit_categories(values: Sequence[str]) -> List[str]:
    """Sorted distinct values - numerically when they're all numbers ("2" < "10")."""
    unique = set(values)
    if is_numeric(unique):
        return sorted(unique, key=float)
    return sorted(unique)


def fit_vocabulary(values: Sequence[str], vocab_size: int = DEFAULT_VOCAB_SIZE) -> List[str]:
    """The vocab_size most frequent words; ties broken alphabetically so it's deterministic."""
    counts = collections.Counter(word for value in values for word in tokenize(value))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:vocab_size]]


def fit_column(column_type: str, values: Sequence[str], vocab_size: int = DEFAULT_VOCAB_SIZE,
               image_size: int = DEFAULT_IMAGE_SIZE) -> Optional[dict]:
    """The "columns" entry orbit import stores for a column (None for numeric)."""
    if column_type == "numeric":
        return None
    if column_type == "categorical":
        return {"type": "categorical", "categories": fit_categories(values)}
    if column_type == "text":
        return {"type": "text", "vocabulary": fit_vocabulary(values, vocab_size)}
    if column_type == "image":
        return {"type": "image", "size": [image_size, image_size]}
    raise ValueError(f"Unknown column type {column_type!r} (valid: {', '.join(COLUMN_TYPES)})")


def load_image(path: pathlib.Path, size: Sequence[int]) -> np.ndarray:
    """Grayscale, resized to size = (height, width), flattened, scaled to [0, 1]."""
    from PIL import Image  # lazy: only image imports need Pillow loaded

    height, width = size
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((width, height)), dtype=np.float32)
    return (pixels / 255.0).reshape(-1)


def encode_images(values: Sequence[str], base_dir: pathlib.Path, size: Sequence[int],
                  column: str) -> np.ndarray:
    rows = []
    for value in values:
        path = base_dir / value
        if not path.is_file():
            raise ValueError(f"Column {column!r}: image file not found: {path}")
        rows.append(load_image(path, size))
    return np.stack(rows)


# --- encoding (every load) -----------------------------------------------------

def column_width(spec: Optional[dict]) -> int:
    if spec is None:
        return 1
    if spec["type"] == "categorical":
        return len(spec["categories"])
    if spec["type"] == "text":
        return len(spec["vocabulary"])
    if spec["type"] == "image":
        return spec["size"][0] * spec["size"][1]
    raise ValueError(f"Unknown column type {spec['type']!r}")


def encode_column(column: str, spec: Optional[dict], values: Sequence[str],
                  images: Optional[np.ndarray] = None) -> np.ndarray:
    """One column's cells as an (N, column_width(spec)) float array."""
    values = list(values)
    if spec is None:
        try:
            return np.array([float(v) for v in values]).reshape(-1, 1)
        except ValueError:
            bad = next(v for v in values if not _is_float(v))
            raise ValueError(
                f"Column {column!r} has a non-numeric value ({bad!r}) - re-import the "
                f"dataset with a column type for it, e.g. --column-type {column}=categorical"
            )

    if spec["type"] == "categorical":
        index = {category: i for i, category in enumerate(spec["categories"])}
        encoded = np.zeros((len(values), len(index)))
        for row, value in enumerate(values):
            if value not in index:
                raise ValueError(f"Column {column!r}: unknown category {value!r}")
            encoded[row, index[value]] = 1.0
        return encoded

    if spec["type"] == "text":
        index = {word: i for i, word in enumerate(spec["vocabulary"])}
        encoded = np.zeros((len(values), len(index)))
        for row, value in enumerate(values):
            for word in tokenize(value):
                if word in index:
                    encoded[row, index[word]] += 1.0
        return encoded

    if spec["type"] == "image":
        if images is None:
            raise ValueError(f"Column {column!r}: decoded images missing (features.npz) - re-import the dataset")
        return np.asarray(images, dtype=float)

    raise ValueError(f"Column {column!r}: unknown column type {spec['type']!r}")


def class_indices(column: str, spec: dict, values: Sequence[str]) -> np.ndarray:
    """A categorical target column as integer class indices, shape (N,)."""
    index = {category: i for i, category in enumerate(spec["categories"])}
    try:
        return np.array([index[v] for v in values])
    except KeyError as e:
        raise ValueError(f"Column {column!r}: unknown category {e.args[0]!r}")


def feature_names(manifest: dict) -> List[str]:
    """
    One name per model input, in order: a numeric column keeps its name,
    the others expand - color=red, word:great, pixels[3,4].
    """
    columns: Dict[str, dict] = manifest.get("columns", {})
    names = []
    for column in manifest["input_columns"]:
        spec = columns.get(column)
        if spec is None:
            names.append(column)
        elif spec["type"] == "categorical":
            names.extend(f"{column}={category}" for category in spec["categories"])
        elif spec["type"] == "text":
            names.extend(f"{column}:{word}" for word in spec["vocabulary"])
        elif spec["type"] == "image":
            height, width = spec["size"]
            names.extend(f"{column}[{r},{c}]" for r in range(height) for c in range(width))
    return names
