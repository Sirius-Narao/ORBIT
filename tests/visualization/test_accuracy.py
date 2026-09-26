import pytest

from orbit.core import Results
from orbit.visualization import plot_accuracy, plot_accuracy_comparison


def test_plot_accuracy_saves_a_png_file(tmp_path):
    results = Results(
        name="xor_test", final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234],
        accuracy_history=[0.5, 0.75, 0.9],
    )
    output_path = tmp_path / "accuracy.png"

    returned = plot_accuracy(results, output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_accuracy_raises_on_empty_history():
    results = Results(name="empty_test", final_loss=0.0, loss_history=[0.1])

    with pytest.raises(ValueError):
        plot_accuracy(results, "unused.png")


def test_plot_accuracy_comparison_saves_a_png_file(tmp_path):
    results_a = Results(
        name="exp_a", final_loss=0.2, loss_history=[0.5, 0.3, 0.2], accuracy_history=[0.5, 0.7, 0.8],
    )
    results_b = Results(
        name="exp_b", final_loss=0.1, loss_history=[0.4, 0.2, 0.1], accuracy_history=[0.6, 0.8, 0.95],
    )
    output_path = tmp_path / "comparison.png"

    returned = plot_accuracy_comparison([results_a, results_b], output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_accuracy_comparison_raises_on_empty_list():
    with pytest.raises(ValueError):
        plot_accuracy_comparison([], "unused.png")


def test_plot_accuracy_log_scale_saves_a_png_file(tmp_path):
    results = Results(
        name="xor_test", final_loss=0.009, loss_history=[0.9, 0.09, 0.009],
        accuracy_history=[0.5, 0.8, 0.99],
    )
    output_path = tmp_path / "accuracy.png"

    returned = plot_accuracy(results, output_path, log_scale=True)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_accuracy_axis_label_by_task():
    from orbit.visualization.accuracy import accuracy_axis_label

    r2 = Results(name="a", final_loss=0.1, loss_history=[0.1], hyperparams={"task": "regression_r2"})
    clf = Results(name="b", final_loss=0.1, loss_history=[0.1], hyperparams={"task": "binary_classification"})
    old = Results(name="c", final_loss=0.1, loss_history=[0.1])  # results.json predating "task"

    assert accuracy_axis_label([r2]) == "R²"
    assert accuracy_axis_label([clf, old]) == "Accuracy"
    assert accuracy_axis_label([r2, clf]) == "Accuracy / R²"


def test_plot_accuracy_saves_a_png_for_an_r2_run(tmp_path):
    results = Results(
        name="r2_run", final_loss=0.1, loss_history=[0.5, 0.1],
        hyperparams={"task": "regression_r2"}, accuracy_history=[-0.4, 0.8],
    )
    output_path = tmp_path / "accuracy.png"

    plot_accuracy(results, output_path)

    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
