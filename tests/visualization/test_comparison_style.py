import matplotlib

matplotlib.use("Agg")
import matplotlib.image
import matplotlib.pyplot as plt

from orbit.core import Results
from orbit.visualization import plot_loss_comparison
from orbit.visualization.comparison_style import (
    INSIDE_LEGEND_MAX,
    add_legend,
    legend_columns,
    line_colors,
)


def make_results(n):
    return [
        Results(final_loss=0.1, loss_history=[1.0 / (i + 1), 0.5 / (i + 1)], name=f"run_{i:03d}")
        for i in range(n)
    ]


def test_line_colors_small_counts_keep_the_default_cycle():
    assert line_colors(3) == [None, None, None]
    assert line_colors(10) == [None] * 10


def test_line_colors_are_all_distinct_past_the_default_cycle():
    for n in (11, 20, 21, 50):
        colors = line_colors(n)
        assert len(colors) == n
        assert len(set(colors)) == n


def test_legend_columns_split_every_20_entries():
    assert legend_columns(7) == 1
    assert legend_columns(20) == 1
    assert legend_columns(21) == 2
    assert legend_columns(45) == 3


def _legend_for(n):
    fig, ax = plt.subplots()
    for i in range(n):
        ax.plot([0, 1], [i, i], label=str(i))
    legend = add_legend(ax, n)
    fig.canvas.draw()
    axes_box = ax.get_window_extent()
    legend_box = legend.get_window_extent()
    plt.close(fig)
    return axes_box, legend_box


def test_add_legend_stays_inside_for_few_lines():
    axes_box, legend_box = _legend_for(INSIDE_LEGEND_MAX)

    assert legend_box.x0 >= axes_box.x0 and legend_box.x1 <= axes_box.x1


def test_add_legend_moves_outside_the_axes_for_many_lines():
    axes_box, legend_box = _legend_for(18)

    # Entirely to the right of the plotting area, so it can't cover any curve.
    assert legend_box.x0 > axes_box.x1


def test_comparison_plot_with_many_lines_keeps_the_outside_legend_in_the_image(tmp_path):
    # Default figure is 640px wide; a cropped-off outside legend would leave
    # the saved PNG no wider than that, while bbox_inches="tight" grows it.
    path = plot_loss_comparison(make_results(18), tmp_path / "many.png")

    width = matplotlib.image.imread(path).shape[1]
    assert width > 640


def test_bars_go_horizontal_past_the_vertical_max():
    from orbit.visualization.comparison_style import VERTICAL_BARS_MAX, use_horizontal_bars

    assert not use_horizontal_bars(VERTICAL_BARS_MAX)
    assert use_horizontal_bars(VERTICAL_BARS_MAX + 1)


def test_bar_figure_height_grows_with_bar_count():
    from orbit.visualization.comparison_style import bar_figure_height

    assert bar_figure_height(3) == 4.8
    assert bar_figure_height(45) > bar_figure_height(18) > 4.8
