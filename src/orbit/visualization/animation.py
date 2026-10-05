"""
Animations of training: the network diagram redrawn at every recorded
weight snapshot, next to the loss curve with a marker at that epoch.

Output is a file - GIF by default (Pillow, already a matplotlib
dependency), or MP4, which needs an ffmpeg binary (see ffmpeg_path).
"""
import pathlib
import shutil
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from orbit.nn.introspection import LayerInfo
from orbit.visualization.backend import can_show, pyplot
from orbit.visualization.network import add_network_colorbars, draw_network, network_figure_size
from orbit.visualization.theme import theme_color, themed

VIDEO_FORMATS = ("gif", "mp4")
DEFAULT_FPS = 10


class VideoWriterError(RuntimeError):
    pass


@dataclass
class Frame:
    epoch: int
    weights: Sequence[Tuple[np.ndarray, np.ndarray]]
    activations: Sequence[np.ndarray]


def ffmpeg_path() -> Optional[str]:
    """
    An ffmpeg executable for MP4 output: one on PATH, else the binary the
    optional imageio-ffmpeg package bundles (pip install -e .[video]), else
    None.
    """
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_writer(video_format: str, fps: int = DEFAULT_FPS):
    import matplotlib
    from matplotlib.animation import FFMpegWriter, PillowWriter

    if video_format == "gif":
        return PillowWriter(fps=fps)
    if video_format == "mp4":
        path = ffmpeg_path()
        if path is None:
            raise VideoWriterError(
                "MP4 output needs ffmpeg: install it (so 'ffmpeg' is on PATH) "
                "or run 'pip install -e .[video]' - or leave out --mp4 for a GIF."
            )
        matplotlib.rcParams["animation.ffmpeg_path"] = path
        # yuv420p keeps the file playable in browsers/Windows' own player.
        return FFMpegWriter(fps=fps, extra_args=["-pix_fmt", "yuv420p"])
    raise ValueError(f"Unknown video format {video_format!r} (expected one of {VIDEO_FORMATS})")


# GIFs get big fast; 80 dpi keeps a 100-frame animation of a small network
# to a few MB while staying readable.
ANIMATION_DPI = 80


def save_animation(fig, update, frame_count: int, output_path: pathlib.Path, video_format: str,
                   fps: int, show: bool, plt) -> pathlib.Path:
    """
    Render update(i) for every frame to output_path, then optionally play
    it in a window. The writer is driven by hand rather than through
    FuncAnimation.save(), which draws every frame twice (an idle redraw,
    then the grab) - half the render time for the same file.
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(video_format, fps)

    with writer.saving(fig, str(output_path), dpi=ANIMATION_DPI):
        for index in range(frame_count):
            update(index)
            writer.grab_frame()

    if show and can_show(plt):
        from matplotlib.animation import FuncAnimation
        animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, repeat=True)  # noqa: F841 - must stay referenced while shown
        plt.show()
    plt.close(fig)
    return output_path


@themed
def animate_training(
    structure: List[LayerInfo],
    frames: List[Frame],
    loss_history: Sequence[float],
    title: str,
    output_path: pathlib.Path,
    value_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    weight_scales: Optional[Sequence[float]] = None,
    ranked_nodes: Optional[Sequence[Sequence[int]]] = None,
    video_format: str = "gif",
    fps: int = DEFAULT_FPS,
    log_scale: bool = False,
    show: bool = False,
) -> pathlib.Path:
    """
    Network diagram (left) at each frame's weights next to the full loss
    curve (right) with a marker at the frame's epoch.

    value_ranges/weight_scales should be computed over *every* frame, so a
    node or edge changing color means the model changed, not that the
    scale was renormalized under it; likewise ranked_nodes should be one
    fixed choice (e.g. from the final weights) so wide columns don't
    reshuffle between frames.
    """
    if not frames:
        raise ValueError("No frames to animate.")

    plt = pyplot(show)

    net_width, net_height = network_figure_size(structure)
    fig = plt.figure(figsize=(net_width + 5.0, max(net_height, 5.0)))
    net_ax = fig.add_axes([0.02, 0.14, 0.58, 0.76])
    loss_ax = fig.add_axes([0.68, 0.22, 0.29, 0.6])
    add_network_colorbars(fig, net_ax, caxes=(fig.add_axes([0.06, 0.06, 0.22, 0.02]),
                                             fig.add_axes([0.34, 0.06, 0.22, 0.02])))

    epochs = np.arange(1, len(loss_history) + 1)
    loss_ax.plot(epochs, loss_history, color=theme_color("accent"), linewidth=1.5)
    if log_scale:
        loss_ax.set_yscale("log")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss" + (" (log scale)" if log_scale else ""))
    loss_ax.set_title("Training loss", fontsize=10)
    epoch_line = loss_ax.axvline(0, color=theme_color("muted"), linewidth=1, linestyle="--")
    (epoch_marker,) = loss_ax.plot([], [], "o", color=theme_color("highlight"), markersize=7)
    fig.suptitle(title, fontsize=12)

    def update(index):
        frame = frames[index]
        net_ax.clear()
        draw_network(
            net_ax, structure, frame.weights, frame.activations,
            value_ranges=value_ranges, weight_scales=weight_scales, ranked_nodes=ranked_nodes,
        )
        net_ax.set_title(f"Epoch {frame.epoch}", fontsize=11)
        epoch_line.set_xdata([frame.epoch, frame.epoch])
        # Epoch 0 is before training: there's no loss value to mark yet.
        if 1 <= frame.epoch <= len(loss_history):
            epoch_marker.set_data([frame.epoch], [loss_history[frame.epoch - 1]])
        else:
            epoch_marker.set_data([], [])

    return save_animation(fig, update, len(frames), output_path, video_format, fps, show, plt)
