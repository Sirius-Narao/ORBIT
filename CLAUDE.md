# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ORBIT is

ORBIT (Open Research & Benchmarking Intelligence Toolkit) is a **terminal-first AI/research toolkit**, not a PyTorch clone. The end vision: the user configures an experiment (dataset, model architecture, training setup) through an interactive CLI, and ORBIT handles dataset loading, model construction, training, and result storage — the user should rarely hand-write a training loop or manually call `loss.backward()` / `optimizer.step()`. See the project's stated vision for the full CLI/UX design (an arrow-key model-builder questionnaire, `orbit import <dataset>`, `orbit experiment new`, etc.) if asked to build toward it.

**Current reality**: only the core tensor/autograd engine and the first NN building blocks exist and are tested (see "What actually works today" below). Everything under `cli/`, `experiment.py`, `storage/`, `sweeps/`, `research/`, `visualization/`, `dataset.py`, `dataloader.py`, `model.py`, `trainer.py`, `optimizer.py` is an empty stub file (0 bytes) sketching the intended architecture — do not assume any of it is implemented, and do not build out large swaths of it speculatively. Implement only what the current task needs, matching the style of the existing engine code.

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
- **`layers/linear.py`** — `Linear(Module)`: `x @ weight + bias`. Note it currently does *not* use `register_parameter`/`register_module` (assigns `self.weight`/`self.bias` directly as plain attributes) — inconsistent with the `Module` registration pattern used elsewhere (e.g. in tests' `MLPDummy`). Be aware of this when composing `Linear` into larger modules, since un-registered attributes won't show up in `.parameters()`.
- **`activations/`** — `Activation(Module)` base class + `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, each a thin wrapper calling the corresponding `Tensor` method (`x.relu()`, etc.) and storing `self.output`.
- **`losses/`** — plain (non-`Module`) `Loss` base class with `forward(y_pred, y_true)`; `MSE` composes existing `Tensor` ops (`(y_pred - y_true) ** 2).mean()`) so it gets autograd for free; `CrossEntropy` instead computes softmax+NLL directly in NumPy and wires a hand-derived fused backward rule (`backward_cross_entropy`) for numerical stability — these are two different, both-valid strategies for adding a loss (compose existing autograd ops vs. hand-derive a fused gradient), pick whichever fits a new loss's numerical stability needs.

### Everything else

`cli/`, `storage/`, `sweeps/`, `research/`, `visualization/`, and most of `core/`/`nn/` beyond what's listed above are empty placeholder files reflecting the target package layout from the project vision. Treat their presence as a hint about where new code should eventually live, not as existing behavior.

## Conventions to follow

- Match the existing style: heavy docstring/comment explanations of the *math* in `core/` (why a gradient rule is what it is), sparser comments in straightforward `nn/` wrapper code.
- New `Tensor` operations follow the established pattern: build the output `Tensor` with `operation=` and `parents=` set, then attach `out._backward = lambda res=out, ...: backward_<op>(res, ...)`, and add the corresponding `backward_<op>` function in `autograd.py`.
- New losses/activations/layers follow the existing base-class shape (`Loss`, `Activation`, `Module`) in their respective `__init__.py`-exported package.
- Test philosophy (see `tests/core/test_autograd.py`): small, single-purpose tests, each with a comment deriving the expected gradient by hand before asserting `np.allclose(...)`. Follow this for new autograd rules — a numerical/hand-derived check, not just a smoke test.
- Don't add broadcasting, shape-inference magic, or other generalizations beyond what the current op needs unless asked — v0.1 intentionally assumes compatible shapes. `backward_add`'s `unbroadcast` is the sole, deliberate exception (needed for `Linear`'s bias term); it is not a precedent for generalizing other ops without being asked.
