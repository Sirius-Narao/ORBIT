import math

import pytest

from orbit.core.ranking import best_indices, metric_ranks, overall_ranking, usable_metrics


def test_metric_ranks_losses_lower_is_better():
    rows = [{"test_loss": 0.3}, {"test_loss": 0.1}, {"test_loss": 0.2}]

    assert metric_ranks(rows, "test_loss") == [3, 1, 2]


def test_metric_ranks_accuracy_higher_is_better():
    rows = [{"test_accuracy": 0.7}, {"test_accuracy": 0.9}, {"test_accuracy": 0.8}]

    assert metric_ranks(rows, "test_accuracy") == [3, 1, 2]


def test_metric_ranks_ties_share_the_average_rank():
    """0.1, 0.1, 0.3 -> the two 0.1s span ranks 1 and 2, so both get 1.5."""
    rows = [{"final_loss": 0.1}, {"final_loss": 0.3}, {"final_loss": 0.1}]

    assert metric_ranks(rows, "final_loss") == [1.5, 3, 1.5]


def test_metric_ranks_missing_values_get_no_rank():
    rows = [{"test_loss": None}, {"test_loss": 0.2}, {"test_loss": float("nan")}, {}]

    assert metric_ranks(rows, "test_loss") == [None, 1, None, None]


def test_metric_ranks_unknown_metric_raises():
    with pytest.raises(ValueError):
        metric_ranks([{}], "bogus")


def test_best_indices_includes_ties():
    rows = [{"final_loss": 0.1}, {"final_loss": 0.3}, {"final_loss": 0.1}]

    assert best_indices(rows, "final_loss") == {0, 2}
    assert best_indices([{"final_loss": None}], "final_loss") == set()


def test_overall_ranking_averages_ranks_across_metrics():
    """
               test_loss  test_accuracy  duration   ranks      avg
      a          0.10        0.80          5.0      1, 2, 3    2.0
      b          0.20        0.90          1.0      2, 1, 1    1.333  <- best overall
      c          0.30        0.70          2.0      3, 3, 2    2.667
    a is best on test_loss alone, yet b wins overall.
    """
    rows = [
        {"name": "a", "test_loss": 0.10, "test_accuracy": 0.80, "duration_seconds": 5.0},
        {"name": "b", "test_loss": 0.20, "test_accuracy": 0.90, "duration_seconds": 1.0},
        {"name": "c", "test_loss": 0.30, "test_accuracy": 0.70, "duration_seconds": 2.0},
    ]

    ranking = overall_ranking(rows, ["test_loss", "test_accuracy", "duration_seconds"])

    assert [e["row"]["name"] for e in ranking] == ["b", "a", "c"]
    assert math.isclose(ranking[0]["avg_rank"], 4 / 3)
    assert ranking[1]["ranks"] == {"test_loss": 1, "test_accuracy": 2, "duration_seconds": 3}
    assert all(e["complete"] for e in ranking)


def test_overall_ranking_breaks_avg_ties_by_metric_order():
    rows = [
        {"name": "a", "test_loss": 0.2, "test_accuracy": 0.9},  # ranks 2, 1
        {"name": "b", "test_loss": 0.1, "test_accuracy": 0.8},  # ranks 1, 2
    ]

    assert [e["row"]["name"] for e in overall_ranking(rows, ["test_loss", "test_accuracy"])] == ["b", "a"]
    assert [e["row"]["name"] for e in overall_ranking(rows, ["test_accuracy", "test_loss"])] == ["a", "b"]


def test_overall_ranking_puts_incomplete_rows_last():
    """
    c has the best test_loss but no test_accuracy - an average over one
    metric is not comparable to an average over two, so it goes after a/b.
    """
    rows = [
        {"name": "a", "test_loss": 0.3, "test_accuracy": 0.7},
        {"name": "b", "test_loss": 0.2, "test_accuracy": 0.9},
        {"name": "c", "test_loss": 0.1, "test_accuracy": None},
        {"name": "d"},
    ]

    ranking = overall_ranking(rows, ["test_loss", "test_accuracy"])

    assert [e["row"]["name"] for e in ranking] == ["b", "a", "c", "d"]
    assert not ranking[2]["complete"]
    assert ranking[3]["avg_rank"] is None


def test_overall_ranking_single_metric_is_plain_ranking():
    rows = [{"final_loss": 0.3}, {"final_loss": 0.1}, {"final_loss": 0.2}]

    ranking = overall_ranking(rows, ["final_loss"])

    assert [e["index"] for e in ranking] == [1, 2, 0]


def test_usable_metrics_drops_metrics_nobody_has():
    rows = [{"final_loss": 0.1, "test_loss": None}, {"final_loss": 0.2}]

    assert usable_metrics(rows, ["final_loss", "test_loss"]) == (["final_loss"], ["test_loss"])
