# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ORBIT is

ORBIT (Open Research & Benchmarking Intelligence Toolkit) is a **terminal-first AI/research toolkit**, not a PyTorch clone. The end vision: the user configures an experiment (dataset, model architecture, training setup) through an interactive CLI, and ORBIT handles dataset loading, model construction, training, and result storage — the user should rarely hand-write a training loop or manually call `loss.backward()` / `optimizer.step()`. See the project's stated vision for the full CLI/UX design (an arrow-key model-builder questionnaire, `orbit import <dataset>`, `orbit experiment new`, etc.) if asked to build toward it.

**Current reality**: the core tensor/autograd engine, the NN building blocks (`Module`, `Parameter`, `Linear`, `Sequential`, activations, losses), a minibatch data pipeline (`Dataset`/`DataLoader`), a batched training loop (`Optimizer`/`SGD`, `Trainer`), and an `Experiment`/`Results` orchestration layer all exist and are tested end-to-end (see `tests/integration/test_xor.py` for the full Tensor → Module → Linear → Activation → Loss → autograd → Optimizer round trip, and `tests/core/test_experiment.py` for the `Experiment`/`Results` layer on top of it). Everything under `cli/`, `storage/`, `sweeps/`, `research/`, `visualization/`, `simulation/`, `core/config.py`, and `core/metrics.py` is still an empty stub file (0 bytes) sketching the intended architecture — do not assume any of it is implemented, and do not build out large swaths of it speculatively. Implement only what the current task needs, matching the style of the existing engine code.

**Roadmap**: the original 5-step plan — (1) `Dataset`/`DataLoader`, (2) batch `Trainer.fit` off a `DataLoader`, (3) a `Sequential` container in `nn/model.py`, (4) plain metric functions in `core/metrics.py`, (5) `Experiment`/`Results` objects bundling model+data+hyperparams+history — is done. Next real phase, in order: `storage/` (persist an `Experiment`'s config + a `Results` object to disk — this is the first storage/reproducibility layer, and unblocks `tests/integration/test_reproducibility.py` which is currently an empty stub), then `cli/` (the arrow-key model builder, `orbit experiment new`), then `sweeps/`, `visualization/`, `research/` — each layers on the ones before it, so don't jump ahead.

## Commands

Package is installed in editable mode in `.venv` (Windows venv at `.venv/Scripts/python.exe`).

```powershell
# Run the full test suite
.venv\Scripts\python.exe -m pytest

# Verbose
.venv\Scripts\python.exe -m pytest -v

# Run a single test file
.venv\Scripts\python.exe -m pytest tests/core/test_autograd.py -v

# Run a single test
.venv\Scripts\python.exe -m pytest tests/core/test_autograd.py::test_matrix_multiplication_backward -v

# Reinstall the package after dependency/packaging changes
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

There is no lint/format tooling configured yet (no ruff/black/flake8 config present) and no `pytest.ini`/`conftest.py` — pytest runs on defaults, discovering `tests/`.

## Architecture

### The autograd engine (`src/orbit/core/`)

This is the heart of the project and the most mathematically load-bearing code.

- **`tensor.py`** — `Tensor` wraps a NumPy array plus graph metadata (`operation`, `parents`, `requires_grad`, `grad`). Every operator (`__add__`, `__mul__`, `__matmul__`, `.relu()`, `.softmax()`, etc.) builds a new output `Tensor` and attaches a closure to `out._backward` that knows how to route gradients to that op's specific parents. `.backward()` just delegates to `autograd.backward()`.
- **`autograd.py`** — the actual differentiation engine, deliberately kept simple and heavily comment-documented (treat the comments as intentional teaching material, not clutter to strip out). Key pieces:
  - `build_topological_order(root)` — DFS over `parents` to produce a topological ordering of the graph.
  - `backward(root, gradient=None)` — seeds `root.grad` (defaults to `1` for a scalar) and walks the topological order in reverse, calling each node's `_backward`.
  - `accumulate_gradient(tensor, gradient)` — adds (not overwrites) into `tensor.grad`, since a tensor may be reused in multiple graph paths (e.g. `y = x * x`).
  - One `backward_<op>` function per operation (add/sub/mul/div/matmul/pow/neg/sum/mean/relu/sigmoid/tanh/softmax, plus a fused `backward_cross_entropy` for softmax+CCE).
- **Known, intentional limitations of v0.1**: backward rules generally assume compatible shapes (no broadcasting) — this is documented, not a bug to silently "fix" by adding broadcasting logic unless asked. The one exception is `backward_add`, which *does* un-broadcast via the `unbroadcast(gradient, shape)` helper (sums a gradient back down to a parent's original shape when forward-pass broadcasting stretched it) — this exists specifically so `Linear`'s `x @ weight + bias` works with a batched input and a 1D bias. `backward_matmul`, `backward_multiply`, etc. still assume both operands already have compatible/equal shapes going in.
- `tensor.py` imports from `autograd.py` with a `try/except ImportError` fallback (relative import, then absolute `orbit.core.autograd`) — preserve this pattern if editing those imports, it's there to let files be run standalone via `python -m orbit.core.tensor` as well as imported as a package.

### The NN layer (`src/orbit/nn/`)

- **`parameter.py`** — `Parameter(Tensor)`, identical to `Tensor` except `requires_grad=True` by default. Marks "this tensor is a trainable weight."
- **`module.py`** — `Module` base class, PyTorch-inspired: `register_parameter`/`register_module` build `_params`/`_modules` dicts; `parameters()`/`named_parameters()`/`modules()`/`named_modules()` recurse through the tree; `forward()` must be overridden by subclasses; `__call__` dispatches to `forward()`; `zero_grad()`, `train()`/`eval()` propagate through submodules.
- **`layers/linear.py`** — `Linear(Module)`: `x @ weight + bias`. It uses `register_parameter` for both `weight` and `bias`, consistent with the `Module` registration pattern used elsewhere (e.g. in tests' `MLPDummy`), so both show up in `.parameters()`.
- **`model.py`** — `Sequential(Module)`: `__init__(self, *layers)` registers each positional layer under its index (`register_module(str(i), layer)`); `forward` just loops `self._modules.values()` in order and threads `x` through each layer, relying on `dict` preserving insertion order (Python 3.7+) to keep that order correct. Lets simple stacks (e.g. `Sequential(Linear(2,8), Tanh(), Linear(8,1), Sigmoid())`) skip writing a one-off `Module` subclass.
- **`activations/`** — `Activation(Module)` base class + `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, each a thin wrapper calling the corresponding `Tensor` method (`x.relu()`, etc.) and storing `self.output`.
- **`losses/`** — plain (non-`Module`) `Loss` base class with `forward(y_pred, y_true)`; `MSE` composes existing `Tensor` ops (`(y_pred - y_true) ** 2).mean()`) so it gets autograd for free; `CrossEntropy` instead computes softmax+NLL directly in NumPy and wires a hand-derived fused backward rule (`backward_cross_entropy`) for numerical stability — these are two different, both-valid strategies for adding a loss (compose existing autograd ops vs. hand-derive a fused gradient), pick whichever fits a new loss's numerical stability needs.
- Note: `layers/__init__.py` and `losses/__init__.py` are empty (no re-exports), unlike `activations/__init__.py` and `optimizers/__init__.py` which do export their public classes. Import `Linear`/`MSE`/`CrossEntropy` via their full submodule path (`orbit.nn.layers.linear`, `orbit.nn.losses.mse`, ...) until that's made consistent.

### The data pipeline (`src/orbit/core/dataset.py`, `src/orbit/core/dataloader.py`)

- **`dataset.py`** — `Dataset` is a plain interface (no `abc`, matching the rest of the codebase): `__len__`/`__getitem__` both raise `NotImplementedError`. Contract: `__getitem__` returns one `(x, y)` sample as raw numpy (or scalars), **not** wrapped in `Tensor` — batching/`Tensor`-wrapping is `DataLoader`'s job, not `Dataset`'s. `TensorDataset(Dataset)` is the one concrete implementation: wraps two arrays (accepts numpy, lists, or `Tensor`s — unwraps `.data` if given a `Tensor`), validates matching row counts (`ValueError` if not), and indexes both in lockstep.
- **`dataloader.py`** — `DataLoader(dataset, batch_size=32, shuffle=True)`. `__len__` is `math.ceil(len(dataset) / batch_size)`. `__iter__` is a **generator** (has to be, not a method that builds and returns a list) — that's what makes every fresh `for X_batch, Y_batch in dataloader:` loop (i.e. every epoch) reshuffle independently rather than replaying the same order forever. Per batch: gathers `dataset[i]` for each index in the chunk, `np.stack`s the x's and y's separately, wraps both as `Tensor` — this is the one place in the pipeline where raw numpy becomes a `Tensor`. The last batch of an epoch is naturally short when the dataset doesn't divide evenly by `batch_size`; no special-casing needed, plain slicing handles it.

### The training layer (`src/orbit/nn/optimizers/`, `src/orbit/nn/training/`)

- **`optimizers/optimizer.py`** — `Optimizer` base class: `__init__(self, parameters, lr=0.01)` stores the parameter list and learning rate; `step()` raises `NotImplementedError`; `zero_grad()` sets every parameter's `.grad` to `None`. `Module.zero_grad()` does the same thing independently — call whichever one you already have a handle on.
- **`optimizers/sgd.py`** — `SGD(Optimizer)`: `step()` does plain `param.data -= self.lr * param.grad` for every parameter whose `.grad` is not `None`, skipping the rest.
- **`training/trainer.py`** — `Trainer.fit(model, loss_fn, optimizer, dataloader, epochs, verbose=False, log_every=100)` now consumes a `DataLoader`: per epoch it loops every batch doing `model.zero_grad()` → `loss.backward()` → `optimizer.step()` (zero_grad happens **per batch**, not per epoch, so gradients never leak across batches), and accumulates a `batch_size`-weighted running sum to compute a true epoch-average loss (a plain float/numpy scalar, not a `Tensor` — you can't return a single `Tensor` representing an epoch once several batches' graphs have each already been consumed by their own `.backward()` call). `fit` returns that final epoch's average; `Trainer` also keeps `self.history`, one averaged-loss entry appended per epoch, for anything that wants the full curve (e.g. `Experiment`/`Results` below).

### Experiment orchestration (`src/orbit/core/experiment.py`, `src/orbit/core/results.py`)

- **`results.py`** — `Results` is a plain, dependency-free data container (no imports from the rest of `orbit`, deliberately, since it's meant to be the thing a future `storage/` layer serializes): `name`, `final_loss`, `loss_history`, `hyperparams` (defaults to `{}`).
- **`experiment.py`** — `Experiment(model, loss_fn, optimizer, dataloader, epochs, name=None, verbose=False, log_every=100)` stores all of that plus its own `Trainer()` instance. `run(self)` (no extra args — everything needed is already on `self` from `__init__`) calls `self.trainer.fit(...)` and packages the result into a `Results`, pulling `loss_history` off `self.trainer.history` and building `hyperparams` from values already sitting on the objects it holds (`optimizer.lr`, `dataloader.batch_size`, `loss_fn.name`) rather than duplicating any of that state itself.

- **`metrics.py`** — plain functions, not classes (no autograd needed on final predictions): `accuracy(y_pred, y_true, threshold=0.5)` for binary classification, `accuracy_multiclass(y_pred, y_true)` using `np.argmax(y_pred.data, axis=-1)` vs. integer class labels.

### Everything else

`cli/`, `storage/`, `sweeps/`, `research/`, `visualization/`, `simulation/`, and `core/config.py` are empty placeholder files reflecting the target package layout from the project vision. Treat their presence as a hint about where new code should eventually live, not as existing behavior. `simulation/` in particular has no design notes yet — don't assume its purpose, ask if a task touches it.

### Known gaps / test coverage

- `tests/nn/test_linear.py`, `tests/nn/test_losses.py` exist as 0-byte stub files — `Linear` and both losses currently have no dedicated unit tests (only indirect coverage via `test_xor.py`).
- Reproducibility (`tests/integration/test_reproducibility.py`) is proven at the pipeline level (seed `np.random.seed(...)` before constructing a model/dataloader → identical `Results`, survives a `storage/` save/load round-trip) but not at the "reconstruct a `Model` from a saved file" level — that needs architecture serialization, which doesn't exist yet.

## Conventions to follow

- Match the existing style: heavy docstring/comment explanations of the *math* in `core/` (why a gradient rule is what it is), sparser comments in straightforward `nn/` wrapper code.
- New `Tensor` operations follow the established pattern: build the output `Tensor` with `operation=` and `parents=` set, then attach `out._backward = lambda res=out, ...: backward_<op>(res, ...)`, and add the corresponding `backward_<op>` function in `autograd.py`.
- New losses/activations/layers follow the existing base-class shape (`Loss`, `Activation`, `Module`) in their respective `__init__.py`-exported package.
- Test philosophy (see `tests/core/test_autograd.py`): small, single-purpose tests, each with a comment deriving the expected gradient by hand before asserting `np.allclose(...)`. Follow this for new autograd rules — a numerical/hand-derived check, not just a smoke test.
- Don't add broadcasting, shape-inference magic, or other generalizations beyond what the current op needs unless asked — v0.1 intentionally assumes compatible shapes. `backward_add`'s `unbroadcast` is the sole, deliberate exception (needed for `Linear`'s bias term); it is not a precedent for generalizing other ops without being asked.
