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


def _bar_results(n):
    return [
        Results(name=f"sweep_run_{i:03d}", final_loss=0.1, loss_history=[0.5, 0.1], test_loss=0.01 * (i + 1))
        for i in range(n)
    ]


def _render_axes(monkeypatch):
    """Capture the Axes _bar_chart draws on, before it is closed."""
    import matplotlib.pyplot as plt

    captured = {}
    real_subplots = plt.subplots

    def spy(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["fig"], captured["ax"] = fig, ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", spy)
    monkeypatch.setattr(plt, "close", lambda fig: None)
    return captured


def test_few_bars_stay_vertical(tmp_path, monkeypatch):
    captured = _render_axes(monkeypatch)

    plot_test_loss(_bar_results(3), tmp_path / "few.png")

    ax = captured["ax"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["sweep_run_000", "sweep_run_001", "sweep_run_002"]
    assert ax.get_ylabel() == "Test Loss"


def test_many_bars_switch_to_horizontal_with_names_first_at_top(tmp_path, monkeypatch):
    captured = _render_axes(monkeypatch)

    plot_test_loss(_bar_results(18), tmp_path / "many.png")

    ax, fig = captured["ax"], captured["fig"]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[0] == "sweep_run_000" and labels[-1] == "sweep_run_017"
    assert ax.yaxis_inverted()  # first experiment drawn at the top
    assert ax.get_xlabel() == "Test Loss"
    # every bar is labeled with its value
    assert len(ax.texts) == 18
    # the figure grows taller than matplotlib's 4.8in default to fit the bars
    assert fig.get_figheight() > 4.8


def test_many_bars_log_scale_applies_to_the_value_axis(tmp_path, monkeypatch):
    captured = _render_axes(monkeypatch)

    plot_test_loss(_bar_results(10), tmp_path / "many_log.png", log_scale=True)

    assert captured["ax"].get_xscale() == "log"
