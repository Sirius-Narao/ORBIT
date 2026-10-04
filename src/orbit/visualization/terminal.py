"""
A live terminal view of a network while it trains (orbit run/train --watch):
each layer as a row of colored cells, one per neuron, recolored as the
model learns, plus each layer's mean |weight| and a loss sparkline.

It's a Trainer on_epoch_end callback that redraws a rich Live display, at
most REFRESH_SECONDS apart - an XOR epoch takes well under a millisecond,
so redrawing every epoch would cost far more than the training itself.
"""
import time
from typing import Callable, List, Optional

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.text import Text

from orbit.nn.introspection import forward_trace, network_structure, weights_of
from orbit.ui import console
from orbit.visualization.network import ACTIVATION_CMAP, column_label, rank_nodes

REFRESH_SECONDS = 0.1
# A row this wide or narrower shows every neuron; a wider one shows its
# most active units (rank_nodes) and "+k" for the rest.
MAX_CELLS = 32
SPARKLINE_WIDTH = 60
SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _hex(rgba) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(channel * 255)) for channel in rgba[:3]))


def sparkline(values: List[float], width: int = SPARKLINE_WIDTH) -> str:
    """
    The last `width` values as block characters, low to high. Plotted on a
    log scale when every value is positive (losses usually are), since a
    loss falling from 0.25 to 0.001 would otherwise flatten into one line
    after the first few epochs.
    """
    values = [v for v in values[-width:] if v is not None and np.isfinite(v)]
    if not values:
        return ""
    data = np.array(values, dtype=float)
    if np.all(data > 0):
        data = np.log10(data)
    low, high = data.min(), data.max()
    if high - low < 1e-12:
        return SPARK_CHARS[0] * len(data)
    levels = np.round((data - low) / (high - low) * (len(SPARK_CHARS) - 1)).astype(int)
    return "".join(SPARK_CHARS[level] for level in levels)


class WatchView:
    """
    Use as a context manager around training, passing the instance as the
    on_epoch_end callback:

        with WatchView(model, X, epochs, name) as view:
            experiment.run(on_epoch_end=view)

    X is the batch the activations are computed on (the training set);
    node colors are its mean activation per neuron, normalized per layer.
    """

    def __init__(self, model, X: np.ndarray, epochs: int, name: str = "",
                 clock: Callable[[], float] = time.monotonic, live: bool = True):
        import matplotlib
        self.model = model
        self.X = np.asarray(X, dtype=float)
        self.epochs = epochs
        self.name = name
        self.structure = network_structure(model)
        self.clock = clock
        self.cmap = matplotlib.colormaps[ACTIVATION_CMAP]
        self.losses: List[float] = []
        self.epoch = 0
        self.last_render: Optional[float] = None
        self.renders = 0
        self._live = Live(console=console, auto_refresh=False, transient=False) if live else None

    def __enter__(self):
        if self._live is not None:
            self._live.start()
            self._live.update(self.render(), refresh=True)
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            self._live.stop()
        return False

    def __call__(self, epoch: int, model, loss: float) -> None:
        self.epoch = epoch
        self.losses.append(float(loss))
        now = self.clock()
        due = self.last_render is None or now - self.last_render >= REFRESH_SECONDS
        if due or epoch == self.epochs:
            self.last_render = now
            self.renders += 1
            if self._live is not None:
                self._live.update(self.render(), refresh=True)

    # --- rendering -------------------------------------------------------

    def _row(self, column: int, values: np.ndarray, ranked: List[int], weight_size: Optional[float],
             label_width: int) -> Text:
        size = len(values)
        shown = sorted(ranked[:MAX_CELLS]) if size > MAX_CELLS else list(range(size))
        low, high = float(values.min()), float(values.max())
        span = high - low if high - low > 1e-12 else 1.0

        row = Text(column_label(self.structure, column).ljust(label_width) + "  ")
        for node in shown:
            row.append("██", style=_hex(self.cmap((values[node] - low) / span)))
        if size > len(shown):
            row.append(f" +{size - len(shown)}", style="#888888")
        if weight_size is not None:
            row.append(f"   mean |w| {weight_size:.3g}", style="#888888")
        return row

    def render(self):
        trace = forward_trace(self.model, self.X)
        means = [column.mean(axis=0) for column in trace]
        ranked = rank_nodes(trace)
        weights = weights_of(self.model)
        label_width = max(len(column_label(self.structure, c)) for c in range(len(trace)))

        rows = [
            self._row(
                column, means[column], ranked[column],
                None if column == 0 else float(np.mean(np.abs(weights[column - 1][0]))),
                label_width,
            )
            for column in range(len(trace))
        ]

        loss = self.losses[-1] if self.losses else float("nan")
        header = Text.assemble(
            ("Epoch ", "bold"), (f"{self.epoch}/{self.epochs}", "bold #ffeab0"),
            ("   loss ", "bold"), (f"{loss:.4g}", "bold #b0ffb3"),
        )
        loss_line = Text.assemble(("Loss  ", "#888888"), (sparkline(self.losses), "#b0ffb3"))
        legend = Text("Neuron color: mean activation over the training set, per layer (dark = low, bright = high)",
                      style="#888888")

        return Panel(
            Group(header, ProgressBar(total=max(self.epochs, 1), completed=self.epoch, width=50),
                  Text(""), *rows, Text(""), loss_line, legend),
            title=f"[bold #ffeab0]{self.name or 'Training'}[/]",
            border_style="#ffeab0",
            expand=False,
        )
