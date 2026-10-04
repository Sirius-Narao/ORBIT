from orbit.ui import warning


def pyplot(show: bool = False):
    """
    Import matplotlib.pyplot with the right backend: Agg (save-to-file only)
    by default, or TkAgg when the caller wants an interactive window too.

    --show is only ever passed outside the REPL (cli/parser.py refuses it
    there), so the TkAgg switch happens in a fresh, short-lived process
    rather than flipping backends mid-session. If Tk isn't available, falls
    back to Agg with a warning - the file is still saved either way.
    """
    import matplotlib

    if show:
        try:
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            warning("Could not open a window (Tk is not available) - saving the file only.")

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def can_show(plt) -> bool:
    """True when pyplot ended up on an interactive backend."""
    return plt.get_backend().lower() != "agg"
