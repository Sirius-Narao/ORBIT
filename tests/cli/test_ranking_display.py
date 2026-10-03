import pytest
from rich.color import Color, ColorSystem

from orbit.cli.commands.ranking_display import (
    METRIC_COLORS,
    OVERALL_BEST_STYLE,
    choose_metrics,
    format_value,
    metric_cell,
)
from orbit.core.ranking import RANK_METRICS


def test_every_rank_metric_has_a_color():
    assert set(METRIC_COLORS) == set(RANK_METRICS)


@pytest.mark.parametrize("system", [ColorSystem.TRUECOLOR, ColorSystem.EIGHT_BIT, ColorSystem.STANDARD])
def test_metric_colors_stay_distinct_from_each_other_and_green_when_downgraded(system):
    green = OVERALL_BEST_STYLE.split()[-1]
    colors = [green, *METRIC_COLORS.values()]

    downgraded = [Color.parse(c).downgrade(system) for c in colors]
    keys = [c.triplet if system == ColorSystem.TRUECOLOR else c.number for c in downgraded]

    assert len(set(keys)) == len(colors)


def test_format_value_per_metric():
    assert format_value(None, "test_loss") == "-"
    assert format_value(0.0000108, "final_loss") == "1.08e-05"
    assert format_value(0.9, "test_accuracy", "binary_classification") == "90.00%"
    assert format_value(0.7266, "accuracy", "regression_r2") == "0.7266"
    assert format_value(1.5, "duration_seconds") == "1.50s"


def test_metric_cell_marks_and_colors_the_best():
    best = metric_cell(0.1, "test_loss", None, True)
    other = metric_cell(0.2, "test_loss", None, False)

    assert best.plain.endswith("★")
    assert METRIC_COLORS["test_loss"] in str(best.style)
    assert other.plain == "0.2"


def test_choose_metrics():
    assert choose_metrics(None, default=["final_loss"]) == ["final_loss"]
    assert choose_metrics(None, default=None) is None
    assert choose_metrics(["test_loss", "test_loss", "accuracy"], default=None) == ["test_loss", "accuracy"]
