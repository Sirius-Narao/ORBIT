# Changelog

## v0.2.0

### Added
- **Settings and a global workspace.** `orbit init [path]` creates the workspace (`<path>/.orbits/`) and a settings file, `~/.orbit/config.toml` (or `$ORBIT_CONFIG`), that points every `orbit` command to it from any directory. `init` also asks for your default hyperparameters and display preferences. `orbit config` shows the settings, and `orbit config set <key> <value>` changes one.
- **Default hyperparameters.** `orbit new` pre-fills the optimizer, learning rate, momentum, normalization, gradient clipping, batch size, epochs and test split from the settings. The values you accept are written into `experiment.json`, so changing the settings never changes existing experiments.
- **Dark and light themes** for every plot and animation, set by `display.theme` or per command with `--theme dark|light`. `animate` and `boundary` get `--gif`/`--mp4` and `--fps`, which default to `display.animation_format`/`display.fps`.
- **More column types in `orbit import`.** Categorical columns (one-hot, or classes when used as the target), text (bag-of-words, `--vocab-size`) and image paths (grayscale, resized to `--image-size`, decoded once at import). Types are detected automatically, then asked about per column, or set with `--column-type COLUMN=TYPE` and `--yes`.
- **Built-in toy datasets:** `moons`, `circles` (binary), `spirals` and `blobs` (3 classes). They are all 2-D, so `orbit boundary` can show what the model learned.
- **Validation split and early stopping.** `"validation_split"` measures a held-out set after every epoch. `"patience"` stops training once the validation loss stops improving and keeps the best epoch's weights. `orbit plot` draws validation curves as dashed lines.

### Fixed
- `CrossEntropy` and multiclass accuracy gave wrong values on imported datasets, because they mishandled `(N, 1)` class-index targets.

### Changed
- `orbit init` now takes an optional path and asks for settings. Use `--yes` to keep the defaults.
- Pillow is now a declared dependency (it was already installed with matplotlib). Python 3.11+ is required, for `tomllib`.

## v0.1.0

First release: the NumPy autograd engine, MLP layers, losses, SGD/Adam, declarative experiments, sweeps and the black-box visualizations (`network`, `animate`, `boundary`, `health`, `--watch`).
