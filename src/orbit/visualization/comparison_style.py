"""
Shared styling for the multi-line *_comparison plots (loss, accuracy,
gradient norm), which `orbit compare --plotloss`, `orbit plot` and
`orbit sweep plot` all go through.

Small comparisons keep matplotlib's defaults - its 10-color cycle and an
in-plot legend - so their output looks the same as before this module
existed. Larger ones (a sweep easily has 10-50 runs) would otherwise get a
legend covering the curves and repeating colors, so they get:
- the legend moved outside the axes, to the right, split into columns so
  it never runs off the bottom of the figure;
- distinct colors: tab20 up to 20 lines, then evenly spaced samples of a
  continuous colormap (beyond ~20, no palette keeps every line distinct,
  but neighbors in run order - usually neighbors in the grid - stay close).
Callers must save with bbox_inches="tight" so the outside legend isn't
cropped off.

The scalar test-metric bar charts (test_metrics.py) have the same scaling
problem in a different form: rotated experiment names under vertical bars
overlap and get cropped past a handful of bars. Past VERTICAL_BARS_MAX they
switch to horizontal bars - names down the y-axis, readable at any count -
in a figure whose height grows with the number of bars (bar_figure_height).

matplotlib is imported lazily, matching the plot functions themselves.
"""
import math

INSIDE_LEGEND_MAX = 6
DEFAULT_CYCLE_MAX = 10
TAB20_MAX = 20
LEGEND_ROWS_PER_COLUMN = 20
VERTICAL_BARS_MAX = 6
BAR_HEIGHT_INCHES = 0.28


def line_colors(n: int) -> list:
    """One color per line; None means "use matplotlib's default cycle"."""
    if n <= DEFAULT_CYCLE_MAX:
        return [None] * n

    import matplotlib

    if n <= TAB20_MAX:
        cmap = matplotlib.colormaps["tab20"]
        return [cmap(i) for i in range(n)]
    cmap = matplotlib.colormaps["turbo"]
    return [cmap(i / (n - 1)) for i in range(n)]


def legend_columns(n: int) -> int:
    return math.ceil(n / LEGEND_ROWS_PER_COLUMN)


def add_legend(ax, n: int):
    if n <= INSIDE_LEGEND_MAX:
        return ax.legend()
    return ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        ncol=legend_columns(n),
        fontsize="small",
        frameon=False,
    )


def use_horizontal_bars(n: int) -> bool:
    return n > VERTICAL_BARS_MAX


def bar_figure_height(n: int) -> float:
    """Inches: matplotlib's default 4.8 until the bars need more room."""
    return max(4.8, BAR_HEIGHT_INCHES * n + 1.5)
