import pytest

from orbit.core import Results
from orbit.visualization import plot_test_loss, plot_test_accuracy


def test_plot_test_loss_saves_a_png_file_for_one_result(tmp_path):
    results = Results(name="exp_a", final_loss=0.1, loss_history=[0.5, 0.3, 0.1], test_loss=0.42)
    output_path = tmp_path / "test_loss.png"

    returned = plot_test_loss([results], output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_test_loss_saves_a_png_file_for_several_results(tmp_path):
    results_list = [
        Results(name=f"exp_{i}", final_loss=0.1 * i, loss_history=[0.5, 0.3, 0.1 * i], test_loss=0.1 * i)
        for i in range(1, 4)
    ]
    output_path = tmp_path / "test_loss.png"

    returned = plot_test_loss(results_list, output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_test_loss_raises_on_empty_list():
    with pytest.raises(ValueError):
        plot_test_loss([], "unused.png")


def test_plot_test_loss_log_scale_saves_a_png_file(tmp_path):
    results = Results(name="exp_a", final_loss=0.1, loss_history=[0.5, 0.3, 0.1], test_loss=0.42)
    output_path = tmp_path / "test_loss.png"

    returned = plot_test_loss([results], output_path, log_scale=True)

    assert returned == output_path
    assert output_path.exists()


def test_plot_test_accuracy_saves_a_png_file(tmp_path):
    results = Results(
        name="exp_a", final_loss=0.1, loss_history=[0.5, 0.3, 0.1], test_accuracy=0.9,
    )
    output_path = tmp_path / "test_accuracy.png"

    returned = plot_test_accuracy([results], output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_test_accuracy_raises_on_empty_list():
    with pytest.raises(ValueError):
        plot_test_accuracy([], "unused.png")
