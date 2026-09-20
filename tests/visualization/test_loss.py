import pytest

from orbit.core import Results
from orbit.visualization import plot_loss, plot_loss_comparison


def test_plot_loss_saves_a_png_file(tmp_path):
    results = Results(name="xor_test", final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])
    output_path = tmp_path / "loss.png"

    returned = plot_loss(results, output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_loss_raises_on_empty_history():
    results = Results(name="empty_test", final_loss=0.0, loss_history=[])

    with pytest.raises(ValueError):
        plot_loss(results, "unused.png")


def test_plot_loss_comparison_saves_a_png_file(tmp_path):
    results_a = Results(name="exp_a", final_loss=0.2, loss_history=[0.5, 0.3, 0.2])
    results_b = Results(name="exp_b", final_loss=0.1, loss_history=[0.4, 0.2, 0.1])
    output_path = tmp_path / "comparison.png"

    returned = plot_loss_comparison([results_a, results_b], output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_loss_comparison_raises_on_empty_list():
    with pytest.raises(ValueError):
        plot_loss_comparison([], "unused.png")


def test_plot_loss_log_scale_saves_a_png_file(tmp_path):
    results = Results(name="xor_test", final_loss=0.009, loss_history=[0.9, 0.09, 0.009])
    output_path = tmp_path / "loss.png"

    returned = plot_loss(results, output_path, log_scale=True)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_loss_comparison_log_scale_saves_a_png_file(tmp_path):
    results_a = Results(name="exp_a", final_loss=0.009, loss_history=[0.9, 0.09, 0.009])
    results_b = Results(name="exp_b", final_loss=0.02, loss_history=[0.8, 0.1, 0.02])
    output_path = tmp_path / "comparison.png"

    returned = plot_loss_comparison([results_a, results_b], output_path, log_scale=True)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
