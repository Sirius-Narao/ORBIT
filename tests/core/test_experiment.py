import numpy as np
from orbit.core import TensorDataset, DataLoader
from orbit.core.metrics import accuracy
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


def test_run_captures_duration_seconds(monkeypatch):
    model = make_fixed_linear(weight=2.0, bias=0.0)
    dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.05)
    loss_fn = MSE()

    times = iter([200.0, 200.25])
    monkeypatch.setattr("orbit.nn.training.trainer.time.time", lambda: next(times))

    experiment = Experiment(model, loss_fn, optimizer, dataloader, epochs=3, name="timed-run")
    results = experiment.run()

    assert results.duration_seconds == 0.25
    assert results.duration_seconds == experiment.trainer.duration_seconds


def test_run_captures_gradient_norm_history():
    model = make_fixed_linear(weight=2.0, bias=0.0)
    dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.05)
    loss_fn = MSE()

    experiment = Experiment(model, loss_fn, optimizer, dataloader, epochs=3, name="gradnorm-run")
    results = experiment.run()

    assert len(results.gradient_norm_history) == 3
    assert results.gradient_norm_history == experiment.trainer.gradient_norm_history


def test_run_captures_accuracy_history_when_accuracy_fn_given():
    model = make_fixed_linear(weight=2.0, bias=0.0)
    dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.05)
    loss_fn = MSE()

    experiment = Experiment(
        model, loss_fn, optimizer, dataloader, epochs=3, name="accuracy-run", accuracy_fn=accuracy
    )
    results = experiment.run()

    assert len(results.accuracy_history) == 3
    assert results.accuracy_history == experiment.trainer.accuracy_history


def test_run_leaves_test_loss_and_accuracy_none_without_test_dataloader():
    model = make_fixed_linear(weight=2.0, bias=0.0)
    dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.05)
    loss_fn = MSE()

    experiment = Experiment(model, loss_fn, optimizer, dataloader, epochs=3, name="no-test-split-run")
    results = experiment.run()

    assert results.test_loss is None
    assert results.test_accuracy is None


def test_run_evaluates_test_dataloader_after_training():
    """
    With lr=0 the weight never updates, so the test set's loss is fully
    hand-computable the same way test_trainer.py's fixtures are: a held-out
    single-row test batch (x=4.0, y=0.0) against the fixed weight=2.0 model
    gives prediction=8.0, MSE = (8.0-0.0)^2 = 64.0.
    """
    model = make_fixed_linear(weight=2.0, bias=0.0)
    train_dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.0)
    loss_fn = MSE()

    test_dataset = TensorDataset(np.array([[4.0]]), np.array([[0.0]]))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    experiment = Experiment(
        model, loss_fn, optimizer, train_dataloader, epochs=3, name="test-split-run",
        test_dataloader=test_dataloader,
    )
    results = experiment.run()

    assert np.isclose(results.test_loss, 64.0)


def test_run_test_accuracy_stays_none_without_accuracy_fn():
    model = make_fixed_linear(weight=2.0, bias=0.0)
    train_dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.0)
    loss_fn = MSE()

    test_dataset = TensorDataset(np.array([[4.0]]), np.array([[0.0]]))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    experiment = Experiment(
        model, loss_fn, optimizer, train_dataloader, epochs=3, name="test-split-no-accuracy",
        test_dataloader=test_dataloader,
    )
    results = experiment.run()

    assert results.test_accuracy is None


def test_run_does_not_mutate_model_during_test_evaluation():
    """
    Training happens first, then the test-set pass - the test pass itself
    must not further change the trained weights (it's forward-only).
    """
    model = make_fixed_linear(weight=2.0, bias=0.0)
    train_dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.1)
    loss_fn = MSE()

    test_dataset = TensorDataset(np.array([[4.0]]), np.array([[0.0]]))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    experiment = Experiment(
        model, loss_fn, optimizer, train_dataloader, epochs=3, name="test-split-no-mutation",
        test_dataloader=test_dataloader,
    )
    experiment.run()
    weight_after_run = model.weight.data.copy()

    # Re-running just the evaluate() call directly must not move the weight
    # any further.
    experiment.trainer.evaluate(model, loss_fn, test_dataloader)

    assert np.array_equal(model.weight.data, weight_after_run)


def test_run_skip_test_leaves_test_loss_and_accuracy_none_even_with_test_dataloader():
    model = make_fixed_linear(weight=2.0, bias=0.0)
    train_dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.0)
    loss_fn = MSE()

    test_dataset = TensorDataset(np.array([[4.0]]), np.array([[0.0]]))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    experiment = Experiment(
        model, loss_fn, optimizer, train_dataloader, epochs=3, name="skip-test-run",
        test_dataloader=test_dataloader,
    )
    results = experiment.run(skip_test=True)

    assert results.test_loss is None
    assert results.test_accuracy is None


def test_run_leaves_accuracy_history_none_by_default():
    model = make_fixed_linear(weight=2.0, bias=0.0)
    dataloader = make_dataloader(batch_size=2)
    optimizer = SGD(model.parameters(), lr=0.05)
    loss_fn = MSE()

    experiment = Experiment(model, loss_fn, optimizer, dataloader, epochs=3, name="no-accuracy-run")
    results = experiment.run()

    assert results.accuracy_history is None


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
