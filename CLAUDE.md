# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ORBIT is

ORBIT (Open Research & Benchmarking Intelligence Toolkit) is a **terminal-first AI/research toolkit**, not a PyTorch clone. The end vision: the user configures an experiment (dataset, model architecture, training setup) through an interactive CLI, and ORBIT handles dataset loading, model construction, training, and result storage — the user should rarely hand-write a training loop or manually call `loss.backward()` / `optimizer.step()`. See the project's stated vision for the full CLI/UX design (an arrow-key model-builder questionnaire, `orbit import <dataset>`, `orbit experiment new`, etc.) if asked to build toward it.

**Current reality**: the core tensor/autograd engine, the NN building blocks (`Module`, `Parameter`, `Linear`, activations, losses), and a first training loop (`Optimizer`/`SGD`, `Trainer`) exist and are tested end-to-end (see `tests/integration/test_xor.py`, a full Tensor → Module → Linear → Activation → Loss → autograd → Optimizer round trip). Everything under `cli/`, `storage/`, `sweeps/`, `research/`, `visualization/`, `simulation/`, `core/dataset.py`, `core/dataloader.py`, `core/config.py`, `core/experiment.py`, `core/metrics.py`, `core/results.py`, and `nn/model.py` is an empty stub file (0 bytes) sketching the intended architecture — do not assume any of it is implemented, and do not build out large swaths of it speculatively. Implement only what the current task needs, matching the style of the existing engine code.

**Roadmap** (what to build next, in order — each step depends on the previous one): (1) `Dataset`/`DataLoader` in `core/dataset.py` / `core/dataloader.py` so training isn't hardcoded to full-batch tensors, (2) update `Trainer.fit` to consume a `DataLoader` and report per-epoch average loss, (3) a `Sequential` container in `nn/model.py` so simple architectures don't require a hand-written `Module` subclass, (4) plain metric functions in `core/metrics.py` (e.g. `accuracy`), (5) `Experiment`/`Results` objects in `core/experiment.py` / `core/results.py` bundling model+data+hyperparams+history. After that: `storage/` (persist experiments), `cli/` (the arrow-key model builder, `orbit experiment new`), `sweeps/`, `visualization/`, `research/` — each of these layers on the ones before it, so don't jump ahead.

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
- **`activations/`** — `Activation(Module)` base class + `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, each a thin wrapper calling the corresponding `Tensor` method (`x.relu()`, etc.) and storing `self.output`.
- **`losses/`** — plain (non-`Module`) `Loss` base class with `forward(y_pred, y_true)`; `MSE` composes existing `Tensor` ops (`(y_pred - y_true) ** 2).mean()`) so it gets autograd for free; `CrossEntropy` instead computes softmax+NLL directly in NumPy and wires a hand-derived fused backward rule (`backward_cross_entropy`) for numerical stability — these are two different, both-valid strategies for adding a loss (compose existing autograd ops vs. hand-derive a fused gradient), pick whichever fits a new loss's numerical stability needs.
- Note: `layers/__init__.py` and `losses/__init__.py` are empty (no re-exports), unlike `activations/__init__.py` and `optimizers/__init__.py` which do export their public classes. Import `Linear`/`MSE`/`CrossEntropy` via their full submodule path (`orbit.nn.layers.linear`, `orbit.nn.losses.mse`, ...) until that's made consistent.

### The training layer (`src/orbit/nn/optimizers/`, `src/orbit/nn/training/`)

- **`optimizers/optimizer.py`** — `Optimizer` base class: `__init__(self, parameters, lr=0.01)` stores the parameter list and learning rate; `step()` raises `NotImplementedError`; `zero_grad()` sets every parameter's `.grad` to `None`. `Module.zero_grad()` does the same thing independently — call whichever one you already have a handle on.
- **`optimizers/sgd.py`** — `SGD(Optimizer)`: `step()` does plain `param.data -= self.lr * param.grad` for every parameter whose `.grad` is not `None`, skipping the rest.
- **`training/trainer.py`** — `Trainer.fit(model, loss_fn, optimizer, X, Y, epochs, verbose=False, log_every=100)` is a minimal **full-batch-only** loop: per epoch it does one forward pass over the entire `X`/`Y`, `model.zero_grad()`, `loss.backward()`, `optimizer.step()`, no minibatching/`DataLoader` support yet (see Roadmap above — that's the next thing to add here).

### Everything else

`cli/`, `storage/`, `sweeps/`, `research/`, `visualization/`, `simulation/`, and the rest of `core/`/`nn/` not listed above (`core/dataset.py`, `core/dataloader.py`, `core/config.py`, `core/experiment.py`, `core/metrics.py`, `core/results.py`, `nn/model.py`) are empty placeholder files reflecting the target package layout from the project vision. Treat their presence as a hint about where new code should eventually live, not as existing behavior. `simulation/` in particular has no design notes yet — don't assume its purpose, ask if a task touches it.

### Known gaps / test coverage

- `tests/nn/test_linear.py`, `tests/nn/test_losses.py`, `tests/core/test_dataloader.py`, `tests/integration/test_reproducibility.py` exist as 0-byte stub files — `Linear` and both losses currently have no dedicated unit tests (only indirect coverage via `test_xor.py`).
- `tests/integration/test_xor.py` calls `test_xor_converges()` at module level in addition to pytest discovering it as a test function, so it currently runs twice per `pytest` invocation. Not a correctness bug today, but don't copy this pattern into new test files.

## Conventions to follow

- Match the existing style: heavy docstring/comment explanations of the *math* in `core/` (why a gradient rule is what it is), sparser comments in straightforward `nn/` wrapper code.
- New `Tensor` operations follow the established pattern: build the output `Tensor` with `operation=` and `parents=` set, then attach `out._backward = lambda res=out, ...: backward_<op>(res, ...)`, and add the corresponding `backward_<op>` function in `autograd.py`.
- New losses/activations/layers follow the existing base-class shape (`Loss`, `Activation`, `Module`) in their respective `__init__.py`-exported package.
- Test philosophy (see `tests/core/test_autograd.py`): small, single-purpose tests, each with a comment deriving the expected gradient by hand before asserting `np.allclose(...)`. Follow this for new autograd rules — a numerical/hand-derived check, not just a smoke test.
- Don't add broadcasting, shape-inference magic, or other generalizations beyond what the current op needs unless asked — v0.1 intentionally assumes compatible shapes. `backward_add`'s `unbroadcast` is the sole, deliberate exception (needed for `Linear`'s bias term); it is not a precedent for generalizing other ops without being asked.
