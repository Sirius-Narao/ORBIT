import copy
import itertools
from typing import Optional

from orbit.core.config import OPTIMIZER_OPTIONS, ALL_OPTIMIZER_OPTIONS
from orbit.sweeps.config import ARCHITECTURE_FIELDS


def _normalize_config(config: dict) -> dict:
    """
    Drop settings that can't take effect, so the config loads cleanly and
    equivalent combinations compare equal:
    - optimizer options that belong to a different optimizer (e.g.
      "momentum" once the optimizer is Adam - load_experiment would raise);
    - "normalize": "none", which means the same as the key being absent;
    - "grad_clip": 0, likewise (no clipping).
    """
    allowed = OPTIMIZER_OPTIONS.get(config.get("optimizer"), ())
    for option in ALL_OPTIMIZER_OPTIONS:
        if option not in allowed:
            config.pop(option, None)
    if config.get("normalize") == "none":
        config.pop("normalize")
    if config.get("grad_clip") == 0:
        config.pop("grad_clip")
    return config


# --- architecture ---------------------------------------------------------------
# hidden_width / hidden_depth / activation are "virtual" sweep fields: they
# never become experiment.json keys, they rewrite its "model" list. A model
# is read as  [hidden part][head]:
#   head   = the last Linear plus any activations after it (e.g. Linear(->1),
#            Sigmoid). Kept exactly as is, so the output width and output
#            activation the dataset/task need are never touched.
#   hidden = everything before the head: Linear layers (their widths) and
#            the activations between them.

def _split_model(model: list) -> tuple:
    linear_positions = [i for i, layer in enumerate(model) if layer["type"] == "Linear"]
    if not linear_positions:
        raise ValueError("Model has no Linear layer to sweep the architecture of")
    head_start = linear_positions[-1]
    return model[:head_start], model[head_start:]


def describe_architecture(model: list) -> dict:
    """
    The base values of the architecture fields: the first hidden Linear's
    width (None without hidden layers), the number of hidden Linear layers,
    and the first hidden activation (None if the hidden part has none).
    """
    hidden, _ = _split_model(model)
    widths = [layer["neurons"] for layer in hidden if layer["type"] == "Linear"]
    activations = [layer["type"] for layer in hidden if layer["type"] != "Linear"]
    return {
        "hidden_width": widths[0] if widths else None,
        "hidden_depth": len(widths),
        "activation": activations[0] if activations else None,
    }


def apply_architecture(
    model: list,
    hidden_width: Optional[int] = None,
    hidden_depth: Optional[int] = None,
    activation: Optional[str] = None,
) -> list:
    """
    Rebuild model as hidden_depth x [Linear(hidden_width), activation] followed
    by the original head. Any argument left as None keeps the base model's
    own value (describe_architecture) - e.g. sweeping only the activation
    keeps the base's depth and width. A tapered base (128 -> 64) has no single
    width; its first hidden width is used.

    in_features (required on the first layer only) moves to whichever layer
    is now first: the first hidden Linear, or the head's Linear at depth 0.
    """
    base = describe_architecture(model)
    width = hidden_width if hidden_width is not None else base["hidden_width"]
    depth = hidden_depth if hidden_depth is not None else base["hidden_depth"]
    act = activation if activation is not None else base["activation"]

    if depth > 0 and width is None:
        raise ValueError(
            "The base model has no hidden layer, so there is no width to reuse - "
            "also vary hidden_width when varying hidden_depth"
        )

    in_features = model[0]["in_features"]
    _, head = _split_model(model)

    layers = []
    for _ in range(depth):
        layers.append({"type": "Linear", "neurons": width})
        if act is not None:
            layers.append({"type": act})
    for layer in copy.deepcopy(head):
        layer.pop("in_features", None)
        layers.append(layer)
    # in_features right after "type", matching hand-written configs
    layers[0] = {"type": "Linear", "in_features": in_features, "neurons": layers[0]["neurons"]}
    return layers


def _effective_architecture(arch: dict) -> dict:
    """At depth 0 there are no hidden layers, so width/activation can't matter."""
    if arch.get("hidden_depth") == 0:
        return {k: v for k, v in arch.items() if k == "hidden_depth"}
    return arch


# --- grid expansion ---------------------------------------------------------------

def expand_grid(sweep_name: str, base_config: dict, grid: dict) -> list:
    """
    Expand a grid ({field: [values, ...]}) into one experiment config per
    combination, applied on top of a copy of base_config.

    Returns a list of (run_name, params, config) tuples, in itertools.product
    order over the grid's keys. params holds only the grid values that
    actually took effect after normalizing - e.g. a momentum value is
    dropped from an Adam run's params, and the duplicates that produces
    (SGD/Adam x momentum [0, 0.9] -> Adam appears twice) are removed, so
    that grid yields 3 runs, not 4. The same goes for architecture: at
    hidden_depth 0, hidden_width/activation are dropped (depth [0, 1] x
    width [8, 16] -> 3 runs).

    Raises ValueError if an architecture combination can't be built (see
    apply_architecture).
    """
    keys = list(grid)
    seen = set()
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        arch = _effective_architecture({k: v for k, v in combo.items() if k in ARCHITECTURE_FIELDS})

        config = copy.deepcopy(base_config)
        config.update({k: v for k, v in combo.items() if k not in ARCHITECTURE_FIELDS})
        config = _normalize_config(config)
        if arch:
            config["model"] = apply_architecture(config["model"], **arch)

        params = {}
        for k in keys:
            if k in ARCHITECTURE_FIELDS:
                if k in arch:
                    params[k] = arch[k]
            elif k in config:
                params[k] = config[k]

        fingerprint = tuple(sorted((k, repr(v)) for k, v in params.items()))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        combos.append((params, config))

    width = max(3, len(str(len(combos))))
    runs = []
    for i, (params, config) in enumerate(combos, start=1):
        run_name = f"{sweep_name}_{i:0{width}d}"
        config["name"] = run_name
        runs.append((run_name, params, config))
    return runs
