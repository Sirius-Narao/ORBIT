import pytest

from orbit.core import Results
from orbit.visualization import plot_gradient_norm, plot_gradient_norm_comparison


def test_plot_gradient_norm_saves_a_png_file(tmp_path):
    results = Results(
        name="xor_test", final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234],
        gradient_norm_history=[3.0, 1.5, 0.8],
    )
    output_path = tmp_path / "gradient_norm.png"

    returned = plot_gradient_norm(results, output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_gradient_norm_raises_on_empty_history():
    results = Results(name="empty_test", final_loss=0.0, loss_history=[0.1])

    with pytest.raises(ValueError):
        plot_gradient_norm(results, "unused.png")


def test_plot_gradient_norm_comparison_saves_a_png_file(tmp_path):
    results_a = Results(
        name="exp_a", final_loss=0.2, loss_history=[0.5, 0.3, 0.2], gradient_norm_history=[3.0, 1.5, 0.8],
    )
    results_b = Results(
        name="exp_b", final_loss=0.1, loss_history=[0.4, 0.2, 0.1], gradient_norm_history=[2.0, 1.0, 0.5],
    )
    output_path = tmp_path / "comparison.png"

    returned = plot_gradient_norm_comparison([results_a, results_b], output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_gradient_norm_comparison_raises_on_empty_list():
    with pytest.raises(ValueError):
        plot_gradient_norm_comparison([], "unused.png")


def test_plot_gradient_norm_log_scale_saves_a_png_file(tmp_path):
    results = Results(
        name="xor_test", final_loss=0.009, loss_history=[0.9, 0.09, 0.009],
        gradient_norm_history=[3.0, 1.5, 0.8],
    )
    output_path = tmp_path / "gradient_norm.png"

    returned = plot_gradient_norm(results, output_path, log_scale=True)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
