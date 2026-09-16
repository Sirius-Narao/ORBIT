import numpy as np
from orbit.core.dataset import TensorDataset
from orbit.core.dataloader import DataLoader
from orbit.nn.layers.linear import Linear
from orbit.nn.losses.mse import MSE
from orbit.nn.optimizers import SGD
from orbit.core.experiment import Experiment
from orbit.storage.experiments import save_results, load_results


def build_and_run(seed):
    """
    Seeding before constructing anything makes the whole run reproducible:
    Linear's np.random.randn weight init and DataLoader's np.random.shuffle
    both draw from the same global numpy RNG stream, so fixing the seed
    once up front fixes every random draw made afterward - no need for a
    seed argument on Linear itself, and no need to hand-pin weights like
    test_trainer.py/test_experiment.py do.
    """
    np.random.seed(seed)

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    Y = np.array([[0.0], [2.0], [4.0], [6.0], [8.0], [10.0], [12.0]])
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = Linear(1, 1)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.01)

    experiment = Experiment(model, loss_fn, optimizer, dataloader, epochs=15, name=f"seed-{seed}")
    return experiment.run()


def test_seeded_runs_produce_identical_results():
    """
    Same seed -> same random weight init and same shuffle order every
    epoch -> the exact same sequence of floating-point operations, so the
    two runs' loss histories must match exactly, not just approximately.
    """
    results_a = build_and_run(seed=42)
    results_b = build_and_run(seed=42)

    assert results_a.loss_history == results_b.loss_history
    assert results_a.final_loss == results_b.final_loss


def test_different_seeds_diverge():
    """
    Guards against a broken/no-op seed: if np.random.seed stopped actually
    being applied somewhere, two "differently seeded" runs could still
    accidentally come out identical, and test_seeded_runs_produce_identical_results
    would pass for the wrong reason. This test confirms the seed is doing
    real work by checking that different seeds produce different results.
    """
    results_a = build_and_run(seed=1)
    results_b = build_and_run(seed=2)

    assert results_a.loss_history != results_b.loss_history


def test_save_reload_preserves_reproducibility(tmp_path):
    """
    A saved run and a freshly reproduced run (same seed) must agree after
    going through the JSON save/load round trip - this is what proves
    Results.to_dict()'s float() casting doesn't quietly lose precision.

    Scope note: this proves the training pipeline (random init + shuffling)
    is seed-reproducible, and that persisting a Results survives that
    comparison. It does NOT prove a saved Results file alone is enough to
    reconstruct and rerun the exact experiment later - that needs model/
    architecture serialization, which doesn't exist yet.
    """
    results_a = build_and_run(seed=7)
    save_results(results_a, tmp_path / "a.json")

    results_b = build_and_run(seed=7)
    loaded = load_results(tmp_path / "a.json")

    assert loaded.final_loss == results_b.final_loss
    assert loaded.loss_history == results_b.loss_history
