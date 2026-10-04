import numpy as np
from rich.console import Console

from orbit.nn import Sequential
from orbit.nn.layers.linear import Linear
from orbit.nn.activations import ReLU, Tanh
from orbit.visualization.terminal import MAX_CELLS, REFRESH_SECONDS, SPARK_CHARS, WatchView, sparkline


def render_text(view) -> str:
    console = Console(record=True, width=160, color_system="truecolor")
    console.print(view.render())
    return console.export_text()


def test_sparkline_maps_the_lowest_and_highest_values_to_the_end_blocks():
    line = sparkline([0.1, 0.01, 0.001])

    # Log scale: three evenly spaced decades -> bottom, middle, top blocks.
    assert line == SPARK_CHARS[-1] + SPARK_CHARS[4] + SPARK_CHARS[0]


def test_sparkline_handles_flat_and_empty_input():
    assert sparkline([0.5, 0.5]) == SPARK_CHARS[0] * 2
    assert sparkline([]) == ""
    assert sparkline([float("nan")]) == ""


def test_sparkline_keeps_only_the_last_values():
    assert len(sparkline(list(range(1, 200)), width=10)) == 10


def test_render_shows_every_layer_epoch_and_loss():
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 1))
    view = WatchView(model, np.random.randn(5, 2), epochs=10, name="xor", live=False)
    view(3, model, 0.125)

    text = render_text(view)

    assert "xor" in text
    assert "3/10" in text
    assert "0.125" in text
    for label in ("Input (2)", "Linear 4 · Tanh", "Output 1"):
        assert label in text
    assert "mean |w|" in text


def test_render_truncates_wide_layers():
    model = Sequential(Linear(3, 50), ReLU(), Linear(50, 1))
    view = WatchView(model, np.random.randn(8, 3), epochs=1, live=False)

    text = render_text(view)

    assert f"+{50 - MAX_CELLS}" in text


def test_callback_throttles_redraws_but_always_draws_the_last_epoch():
    model = Sequential(Linear(2, 1))
    now = [0.0]
    view = WatchView(model, np.zeros((1, 2)), epochs=5, clock=lambda: now[0], live=False)

    for epoch in range(1, 5):
        view(epoch, model, 1.0)  # all at t=0: only the first one is due
    assert view.renders == 1

    now[0] = REFRESH_SECONDS
    view(5, model, 0.5)
    assert view.renders == 2
    assert view.losses == [1.0, 1.0, 1.0, 1.0, 0.5]
