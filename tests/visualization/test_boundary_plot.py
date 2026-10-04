import numpy as np
import pytest

from orbit.visualization.boundary import (
    BoundaryData,
    animate_boundary,
    boundary_grid,
    class_labels,
    is_classification,
    plot_boundary,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_boundary_grid_pads_the_data_range():
    X = np.array([[0.0, 0.0], [1.0, 2.0]])

    xx, yy, points = boundary_grid(X, resolution=5, margin=0.5)

    # x range 1 -> pad 0.5; y range 2 -> pad 1.
    assert np.isclose(xx.min(), -0.5) and np.isclose(xx.max(), 1.5)
    assert np.isclose(yy.min(), -1.0) and np.isclose(yy.max(), 3.0)
    assert points.shape == (25, 2)


def test_boundary_grid_rejects_more_than_two_features():
    with pytest.raises(ValueError):
        boundary_grid(np.zeros((3, 3)))


def test_class_labels_from_one_hot_and_from_indices():
    assert list(class_labels(np.array([[0, 1, 0], [1, 0, 0]]), 3)) == [1, 0]
    assert list(class_labels(np.array([[2], [0]]), 3)) == [2, 0]


def test_is_classification_detects_binary_targets_without_a_task():
    assert is_classification(BoundaryData(np.zeros((2, 2)), np.array([[0.0], [1.0]])), 1)
    assert not is_classification(BoundaryData(np.zeros((2, 2)), np.array([[0.2], [3.0]])), 1)
    assert not is_classification(
        BoundaryData(np.zeros((2, 2)), np.array([[0.0], [1.0]]), task="regression_r2"), 1
    )


@pytest.mark.parametrize("Y, predict", [
    (np.array([[0.0], [1.0], [1.0], [0.0]]), lambda p: 1 / (1 + np.exp(-(p[:, :1] - p[:, 1:])))),  # binary
    (np.array([[0.5], [2.0], [3.5], [1.0]]), lambda p: p[:, :1] * 2),                               # regression
    (np.array([[0], [1], [2], [1]]), lambda p: np.column_stack([p[:, 0], p[:, 1], -p.sum(axis=1)])),  # 3 classes
])
def test_plot_boundary_saves_a_png_for_each_kind_of_output(tmp_path, Y, predict):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    data = BoundaryData(X[:3], Y[:3], X[3:], Y[3:])
    xx, yy, points = boundary_grid(X, resolution=20)

    output = plot_boundary(xx, yy, predict(points), data, "test", tmp_path / "b.png")

    assert output.read_bytes()[:8] == PNG_MAGIC


def test_animate_boundary_writes_one_frame_per_snapshot(tmp_path):
    from PIL import Image

    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    data = BoundaryData(X, np.array([[0.0], [1.0]]))
    xx, yy, points = boundary_grid(X, resolution=10)
    frames = [(epoch, 1 / (1 + np.exp(-epoch * (points[:, :1] - 0.5)))) for epoch in range(3)]

    output = animate_boundary(xx, yy, frames, data, "test", tmp_path / "b.gif")

    with Image.open(output) as gif:
        assert gif.n_frames == 3
