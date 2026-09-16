import numpy as np
from orbit.core import TensorDataset, DataLoader
from orbit.nn.layers import Linear
from orbit.nn.losses import MSE
from orbit.nn.optimizers import SGD
from orbit.nn.training import Trainer
from orbit.core.experiment import Experiment


def make_fixed_linear(weight, bias):
    """
    Linear(1, 1) with weight/bias pinned to known values, matching the
    fixture used in tests/nn/test_trainer.py so losses are hand-computable.
    """
    model = Linear(1, 1)
    model.weight.data = np.array([[weight]])
    model.bias.data = np.array([bias])
    return model


def make_dataloader(batch_size, shuffle=False):
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def test_run_returns_results_with_correct_length_history():
    """
    With lr=0 the weight never updates, so every one of the 5 epochs sees
    the same fixed model against the same (unshuffled) batches - the
    per-epoch loss must be 56/3 every time (see test_trainer.py's
    test_fit_weights_epoch_average_by_batch_size for the by-hand
    derivation of 56/3 for weight=2, X=[1,2,3], Y=[0,0,0], batch_size=2).
    """
    model = make_fixed_linear(weight=2.0, bias=0.0)
    dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.0)
    loss_fn = MSE()

    experiment = Experiment(model, loss_fn, optimizer, dataloader, epochs=5, name="fixed-weight")
    results = experiment.run()

    assert len(results.loss_history) == 5
    assert all(np.isclose(loss, 56 / 3) for loss in results.loss_history)
    assert np.isclose(results.final_loss, 56 / 3)


def test_run_captures_hyperparams():
    model = make_fixed_linear(weight=2.0, bias=0.0)
    dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.05)
    loss_fn = MSE()

    experiment = Experiment(model, loss_fn, optimizer, dataloader, epochs=3, name="hp-check")
    results = experiment.run()

    assert results.hyperparams == {
        "epochs": 3,
        "lr": 0.05,
        "batch_size": 2,
        "loss": "mse",
    }
    assert results.name == "hp-check"


def test_run_matches_calling_trainer_directly():
    """
    Experiment.run() should be a faithful wrapper around Trainer.fit():
    training an identical model/optimizer/dataloader combo directly
    through Trainer must produce the same final loss as going through
    Experiment.
    """
    dataloader_a = make_dataloader(batch_size=2)
    model_a = make_fixed_linear(weight=2.0, bias=0.0)
    optimizer_a = SGD(model_a.parameters(), lr=0.1)
    loss_a = Trainer().fit(model_a, MSE(), optimizer_a, dataloader_a, epochs=10)

    dataloader_b = make_dataloader(batch_size=2)
    model_b = make_fixed_linear(weight=2.0, bias=0.0)
    optimizer_b = SGD(model_b.parameters(), lr=0.1)
    experiment = Experiment(model_b, MSE(), optimizer_b, dataloader_b, epochs=10)
    results = experiment.run()

    assert np.isclose(results.final_loss, loss_a)
