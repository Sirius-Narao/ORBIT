import copy
import itertools

from orbit.core.config import OPTIMIZER_OPTIONS, ALL_OPTIMIZER_OPTIONS


def _normalize_config(config: dict) -> dict:
    """
    Drop settings that can't take effect, so the config loads cleanly and
    equivalent combinations compare equal:
    - optimizer options that belong to a different optimizer (e.g.
      "momentum" once the optimizer is Adam - load_experiment would raise);
    - "normalize": "none", which means the same as the key being absent.
    """
    allowed = OPTIMIZER_OPTIONS.get(config.get("optimizer"), ())
    for option in ALL_OPTIMIZER_OPTIONS:
        if option not in allowed:
            config.pop(option, None)
    if config.get("normalize") == "none":
        config.pop("normalize")
    return config


def expand_grid(sweep_name: str, base_config: dict, grid: dict) -> list:
    """
    Expand a grid ({field: [values, ...]}) into one experiment config per
    combination, applied on top of a copy of base_config.

    Returns a list of (run_name, params, config) tuples, in itertools.product
    order over the grid's keys. params holds only the grid values that
    actually took effect after normalizing - e.g. a momentum value is
    dropped from an Adam run's params, and the duplicates that produces
    (SGD/Adam x momentum [0, 0.9] -> Adam appears twice) are removed, so
    that grid yields 3 runs, not 4.
    """
    keys = list(grid)
    seen = set()
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        config = copy.deepcopy(base_config)
        config.update(dict(zip(keys, values)))
        config = _normalize_config(config)
        params = {k: config[k] for k in keys if k in config}

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
