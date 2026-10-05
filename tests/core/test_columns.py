import numpy as np
import pytest

from orbit.core.columns import (
    class_indices,
    column_width,
    detect_column_type,
    encode_column,
    feature_names,
    fit_categories,
    fit_column,
    fit_vocabulary,
    load_image,
    tokenize,
)


def write_image(path, pixels):
    from PIL import Image

    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(path)
    return path


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("It's GREAT, 10/10!") == ["it's", "great", "10", "10"]


@pytest.mark.parametrize("values, expected", [
    (["1", "2.5", "-3e2"], "numeric"),
    (["red", "green", "red", "blue"], "categorical"),
    (["loved this movie a lot", "not good at all", "would watch again"], "text"),
])
def test_detect_column_type(values, expected):
    assert detect_column_type(values) == expected


def test_detect_image_paths():
    assert detect_column_type(["imgs/a.png", "b.JPG"]) == "image"


def test_missing_image_file_is_reported(tmp_path):
    from orbit.core.columns import encode_images

    with pytest.raises(ValueError, match="image file not found"):
        encode_images(["missing.png"], tmp_path, (2, 2), "img")


def test_categories_sort_numerically_when_numbers():
    assert fit_categories(["10", "2", "1", "2"]) == ["1", "2", "10"]
    assert fit_categories(["b", "a", "b"]) == ["a", "b"]


def test_vocabulary_keeps_most_frequent_words_with_alphabetical_ties():
    # counts: good 3, bad 2, film 1, zebra 1 -> top 3 = good, bad, film
    values = ["good good bad", "bad good", "zebra film"]

    assert fit_vocabulary(values, vocab_size=3) == ["good", "bad", "film"]


def test_categorical_one_hot():
    spec = {"type": "categorical", "categories": ["blue", "green", "red"]}

    encoded = encode_column("color", spec, ["red", "blue"])

    assert encoded.tolist() == [[0, 0, 1], [1, 0, 0]]


def test_unknown_category_raises():
    spec = {"type": "categorical", "categories": ["a"]}

    with pytest.raises(ValueError, match="unknown category"):
        encode_column("c", spec, ["b"])


def test_text_counts_vocabulary_words_and_ignores_others():
    spec = {"type": "text", "vocabulary": ["good", "bad"]}

    encoded = encode_column("review", spec, ["good, very GOOD", "bad", "meh"])

    assert encoded.tolist() == [[2, 0], [0, 1], [0, 0]]


def test_numeric_column_with_text_names_the_fix():
    with pytest.raises(ValueError, match="--column-type x=categorical"):
        encode_column("x", None, ["1", "oops"])


def test_load_image_grayscale_resized_and_scaled(tmp_path):
    path = write_image(tmp_path / "img.png", [[0, 255], [255, 0]])

    pixels = load_image(path, (2, 2))

    assert pixels.shape == (4,)
    assert np.allclose(pixels, [0, 1, 1, 0])


def test_column_widths():
    assert column_width(None) == 1
    assert column_width(fit_column("categorical", ["a", "b", "a"])) == 2
    assert column_width(fit_column("image", [], image_size=5)) == 25


def test_class_indices_follow_category_order():
    spec = {"type": "categorical", "categories": ["cat", "dog"]}

    assert class_indices("label", spec, ["dog", "cat", "dog"]).tolist() == [1, 0, 1]


def test_feature_names_expand_typed_columns():
    manifest = {
        "input_columns": ["size", "color", "review", "img"],
        "output_columns": ["y"],
        "columns": {
            "color": {"type": "categorical", "categories": ["blue", "red"]},
            "review": {"type": "text", "vocabulary": ["good"]},
            "img": {"type": "image", "size": [1, 2]},
        },
    }

    assert feature_names(manifest) == [
        "size", "color=blue", "color=red", "review:good", "img[0,0]", "img[0,1]",
    ]
