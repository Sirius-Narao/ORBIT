import pytest

from orbit.core import Results
from orbit.visualization import plot_loss


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
