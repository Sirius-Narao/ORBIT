import numpy as np
import pytest

from orbit.core import Results
from orbit.settings import load_settings, save_settings, set_value
from orbit.visualization import plot_loss
from orbit.visualization.theme import (
    THEMES,
    resolve_theme,
    set_theme_override,
    theme_color,
    theme_context,
)


@pytest.fixture(autouse=True)
def no_override():
    set_theme_override(None)
    yield
    set_theme_override(None)


def hex_to_rgb(color):
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def corner_pixel(path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").getpixel((1, 1))


def results():
    return Results(name="r", final_loss=0.1, loss_history=[0.5, 0.3, 0.1])


def test_without_settings_the_theme_is_light():
    assert resolve_theme() == "light"


def test_settings_choose_the_theme():
    save_settings(set_value(load_settings(), "display.theme", "dark"))

    assert resolve_theme() == "dark"


def test_cli_override_beats_settings_and_explicit_beats_both():
    save_settings(set_value(load_settings(), "display.theme", "dark"))
    set_theme_override("light")

    assert resolve_theme() == "light"
    assert resolve_theme("dark") == "dark"


def test_unknown_theme_is_rejected():
    with pytest.raises(ValueError):
        set_theme_override("neon")


def test_theme_color_follows_the_active_context():
    with theme_context("dark"):
        assert theme_color("accent") == THEMES["dark"]["roles"]["accent"]
        with theme_context("light"):
            assert theme_color("accent") == THEMES["light"]["roles"]["accent"]
        assert theme_color("accent") == THEMES["dark"]["roles"]["accent"]


def test_theme_context_restores_matplotlib_settings():
    import matplotlib

    before = matplotlib.rcParams["axes.facecolor"]
    with theme_context("dark"):
        assert matplotlib.rcParams["axes.facecolor"] == THEMES["dark"]["rc"]["axes.facecolor"]
    assert matplotlib.rcParams["axes.facecolor"] == before


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_saved_png_background_matches_the_theme(tmp_path, theme):
    path = plot_loss(results(), tmp_path / "loss.png", theme=theme)

    assert corner_pixel(path) == hex_to_rgb(THEMES[theme]["roles"]["surface"])


def test_png_uses_the_settings_theme_by_default(tmp_path):
    save_settings(set_value(load_settings(), "display.theme", "dark"))

    path = plot_loss(results(), tmp_path / "loss.png")

    assert corner_pixel(path) == hex_to_rgb(THEMES["dark"]["roles"]["surface"])


def test_animation_frames_use_the_theme_background(tmp_path):
    from PIL import Image

    from orbit.nn import Sequential
    from orbit.nn.activations import Tanh
    from orbit.nn.introspection import forward_trace, network_structure, weights_of
    from orbit.nn.layers.linear import Linear
    from orbit.visualization.animation import Frame, animate_training

    model = Sequential(Linear(2, 3), Tanh(), Linear(3, 1))
    trace = forward_trace(model, np.random.randn(4, 2))
    frames = [Frame(0, weights_of(model), [column.mean(axis=0) for column in trace])]

    path = animate_training(network_structure(model), frames, [0.5], "t", tmp_path / "a.gif", theme="dark")

    with Image.open(path) as gif:
        pixel = gif.convert("RGB").getpixel((1, 1))
    # GIF quantizes to a 256-color palette, so allow a small difference.
    expected = hex_to_rgb(THEMES["dark"]["roles"]["surface"])
    assert all(abs(a - b) <= 8 for a, b in zip(pixel, expected))
