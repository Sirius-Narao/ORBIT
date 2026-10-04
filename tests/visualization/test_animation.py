import numpy as np
import pytest

from orbit.nn import Sequential
from orbit.nn.layers.linear import Linear
from orbit.nn.activations import Tanh
from orbit.nn.introspection import forward_trace, network_structure, weights_of
from orbit.visualization.animation import Frame, VideoWriterError, animate_training, ffmpeg_path, make_writer


def make_frames(count=3):
    model = Sequential(Linear(2, 4), Tanh(), Linear(4, 1))
    X = np.random.randn(5, 2)
    frames = []
    for epoch in range(count):
        for param in model.parameters():
            param.data = param.data * 1.1
        trace = forward_trace(model, X)
        frames.append(Frame(epoch, weights_of(model), [column.mean(axis=0) for column in trace]))
    return network_structure(model), frames


def test_animate_training_writes_a_gif_with_one_frame_per_snapshot(tmp_path):
    from PIL import Image

    structure, frames = make_frames(3)
    output_path = tmp_path / "training.gif"

    returned = animate_training(structure, frames, [0.5, 0.3], "test", output_path)

    assert returned == output_path
    with Image.open(output_path) as gif:
        assert gif.n_frames == 3


def test_animate_training_raises_without_frames(tmp_path):
    structure, _ = make_frames(1)

    with pytest.raises(ValueError):
        animate_training(structure, [], [0.5], "test", tmp_path / "x.gif")


@pytest.mark.skipif(ffmpeg_path() is None, reason="no ffmpeg available")
def test_animate_training_writes_an_mp4(tmp_path):
    structure, frames = make_frames(2)
    output_path = tmp_path / "training.mp4"

    animate_training(structure, frames, [0.5, 0.3], "test", output_path, video_format="mp4", log_scale=True)

    assert output_path.stat().st_size > 0


def test_make_writer_explains_how_to_get_ffmpeg(monkeypatch):
    monkeypatch.setattr("orbit.visualization.animation.ffmpeg_path", lambda: None)

    with pytest.raises(VideoWriterError, match=r"\[video\]"):
        make_writer("mp4")


def test_make_writer_rejects_an_unknown_format():
    with pytest.raises(ValueError):
        make_writer("avi")
