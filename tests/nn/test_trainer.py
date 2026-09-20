import numpy as np
from orbit.core import TensorDataset, DataLoader
from orbit.core.metrics import accuracy
from orbit.nn.layers import Linear
from orbit.nn.losses import MSE
from orbit.nn.optimizers import SGD
from orbit.nn.training import Trainer


def make_fixed_linear(weight, bias):
    """
    Linear(1, 1) with weight/bias pinned to known values, so predictions
    (and therefore losses) are fully hand-computable.
    """
    model = Linear(1, 1)
    model.weight.data = np.array([[weight]])
    model.bias.data = np.array([bias])
    return model


def test_fit_weights_epoch_average_by_batch_size():
    """
    3 samples, batch_size=2, shuffle=False -> batches of size [2, 1].

    With weight=2, bias=0, target=0:
        predictions = [2, 4, 6]   (for X = [1, 2, 3])
        squared errors = [4, 16, 36]

    Batch 1 (rows 0,1): MSE = mean(4, 16) = 10
    Batch 2 (row 2):    MSE = mean(36)    = 36

    Correct (sample-weighted) epoch average:
        (10*2 + 36*1) / 3 = 56/3 = 18.666...

    A naive average of the two batch losses would instead give
        (10 + 36) / 2 = 23,
    which is wrong because the batches aren't the same size. lr=0 so SGD
    never updates the weight, keeping every batch's loss computed against
    the same fixed model.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    avg_loss = Trainer().fit(model, loss_fn, optimizer, dataloader, epochs=1)

    assert np.isclose(avg_loss, 56 / 3)


def test_fit_returns_a_plain_number_not_a_tensor():
    """
    fit() computes the epoch average after several batches' losses have
    each already been consumed by backward(), so it can't return a single
    Tensor tied to the computation graph - it must return a plain scalar.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    avg_loss = Trainer().fit(model, loss_fn, optimizer, dataloader, epochs=1)

    assert not hasattr(avg_loss, "backward")


def test_fit_updates_parameters_via_optimizer():
    """
    Sanity check that Trainer actually drives the optimizer: with lr > 0
    and a non-zero gradient, the weight must move away from its initial
    value after fit().
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    initial_weight = model.weight.data.copy()

    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.1)

    Trainer().fit(model, loss_fn, optimizer, dataloader, epochs=1)

    assert not np.allclose(model.weight.data, initial_weight)


def test_fit_records_history_matching_returned_loss():
    """
    self.history should collect one entry per epoch, and the last entry
    must equal the value fit() returns.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    trainer = Trainer()
    avg_loss = trainer.fit(model, loss_fn, optimizer, dataloader, epochs=5)

    assert len(trainer.history) == 5
    assert trainer.history[-1] == avg_loss


def test_fit_zero_grads_before_each_batch():
    """
    zero_grad() must run before each batch's backward(), not once per
    epoch - otherwise gradients from earlier batches in the same epoch
    would accumulate into later batches' gradients instead of being
    replaced. With lr=0 (no updates) and 3 single-row batches, the
    weight's final gradient should match the gradient of the LAST batch
    alone, not the sum across all three batches.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    Trainer().fit(model, loss_fn, optimizer, dataloader, epochs=1)

    # Gradient of MSE = mean((w*x - y)^2) w.r.t. w, for a single sample,
    # is 2 * (w*x - y) * x. Last batch is row 2: x=3.0, y=0.0, w=2.0.
    x_last, y_last = 3.0, 0.0
    expected_grad = 2 * (2.0 * x_last - y_last) * x_last

    assert np.isclose(model.weight.grad.item(), expected_grad)


def test_fit_records_gradient_norm_history_length():
    """
    self.gradient_norm_history should collect one entry per epoch, mirroring
    self.history.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    trainer = Trainer()
    trainer.fit(model, loss_fn, optimizer, dataloader, epochs=5)

    assert len(trainer.gradient_norm_history) == 5


def test_fit_gradient_norm_matches_hand_derived_value():
    """
    Single batch covering all 3 samples (batch_size=3), weight=2, bias=0,
    targets all 0.

    predictions = w*x = [2, 4, 6]
    dL/dw = (2/N) * sum((pred_i - y_i) * x_i) = (2/3) * (2*1 + 4*2 + 6*3) = 56/3
    dL/db = (2/N) * sum(pred_i - y_i)         = (2/3) * (2 + 4 + 6)       = 8.0
    grad_norm = sqrt((56/3)^2 + 8.0^2)
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    trainer = Trainer()
    trainer.fit(model, loss_fn, optimizer, dataloader, epochs=1)

    expected_norm = np.sqrt((56 / 3) ** 2 + 8.0 ** 2)
    assert np.isclose(trainer.gradient_norm_history[-1], expected_norm)


def test_fit_gradient_norm_reflects_last_batch_only():
    """
    Same setup as test_fit_zero_grads_before_each_batch: batch_size=1, 3
    single-row batches, lr=0. The recorded norm for this epoch must match
    only the last batch's gradient (row 2: x=3.0, y=0.0, w=2.0, b=0.0), not
    an accumulation across all three batches.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    trainer = Trainer()
    trainer.fit(model, loss_fn, optimizer, dataloader, epochs=1)

    x_last, y_last = 3.0, 0.0
    expected_grad_w = 2 * (2.0 * x_last - y_last) * x_last
    expected_grad_b = 2 * (2.0 * x_last - y_last)
    expected_norm = np.sqrt(expected_grad_w ** 2 + expected_grad_b ** 2)

    assert np.isclose(trainer.gradient_norm_history[-1], expected_norm)


def test_fit_accuracy_history_stays_none_without_accuracy_fn():
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    trainer = Trainer()
    trainer.fit(model, loss_fn, optimizer, dataloader, epochs=5)

    assert trainer.accuracy_history is None


def test_fit_accuracy_history_is_batch_size_weighted():
    """
    Mirrors test_fit_weights_epoch_average_by_batch_size's batch-weighting
    logic, but for accuracy: batch_size=2 -> batches of size [2, 1]. A stub
    accuracy_fn returns 1.0 for the first batch and 0.0 for the second, so
    the correct sample-weighted epoch average is (1.0*2 + 0.0*1) / 3 = 2/3,
    not a naive (1.0 + 0.0) / 2 = 0.5.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    accuracies = iter([1.0, 0.0])
    stub_accuracy_fn = lambda y_pred, y_true: next(accuracies)

    trainer = Trainer()
    trainer.fit(model, loss_fn, optimizer, dataloader, epochs=1, accuracy_fn=stub_accuracy_fn)

    assert np.isclose(trainer.accuracy_history[-1], 2 / 3)


def test_fit_accuracy_history_matches_real_accuracy_function():
    """
    Integration check using the real core.metrics.accuracy function (not a
    stub), proving the Trainer -> accuracy_fn dispatch path works end to
    end. weight=1, bias=0 (identity): predictions = X = [0.2, 0.6, 0.9].
    Thresholded at 0.5: [0, 1, 1]. Targets: [0, 1, 0]. Matches at indices 0
    and 1 only -> accuracy = 2/3.
    """
    X = np.array([[0.2], [0.6], [0.9]])
    Y = np.array([[0.0], [1.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    model = make_fixed_linear(weight=1.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    trainer = Trainer()
    trainer.fit(model, loss_fn, optimizer, dataloader, epochs=1, accuracy_fn=accuracy)

    assert np.isclose(trainer.accuracy_history[-1], 2 / 3)


def test_evaluate_returns_batch_weighted_average_loss():
    """
    Mirrors test_fit_weights_epoch_average_by_batch_size's math, but via
    evaluate() on a single batch covering all 3 rows: predictions = [2,4,6]
    against targets [0,0,0] -> MSE = mean(4,16,36) = 56/3.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()

    avg_loss, avg_accuracy = Trainer().evaluate(model, loss_fn, dataloader)

    assert np.isclose(avg_loss, 56 / 3)
    assert avg_accuracy is None


def test_evaluate_does_not_mutate_model_parameters():
    """
    evaluate() is forward-pass only - no zero_grad()/backward()/step(), so
    weights must be bit-for-bit unchanged afterward, and no gradient should
    even get computed.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    initial_weight = model.weight.data.copy()
    initial_bias = model.bias.data.copy()
    loss_fn = MSE()

    Trainer().evaluate(model, loss_fn, dataloader)

    assert np.array_equal(model.weight.data, initial_weight)
    assert np.array_equal(model.bias.data, initial_bias)
    assert model.weight.grad is None
    assert model.bias.grad is None


def test_evaluate_computes_batch_weighted_accuracy():
    """
    Same batch-weighting shape as test_fit_accuracy_history_is_batch_size_weighted:
    batch_size=2 over 3 rows -> batches of size [2, 1]. Stub accuracy_fn
    returns 1.0 then 0.0, so the correct weighted average is
    (1.0*2 + 0.0*1) / 3 = 2/3, not a naive (1.0+0.0)/2 = 0.5.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()

    accuracies = iter([1.0, 0.0])
    stub_accuracy_fn = lambda y_pred, y_true: next(accuracies)

    _, avg_accuracy = Trainer().evaluate(model, loss_fn, dataloader, accuracy_fn=stub_accuracy_fn)

    assert np.isclose(avg_accuracy, 2 / 3)


def test_fit_records_duration_seconds(monkeypatch):
    """
    fit() wraps its whole dispatch (start, then end) in time.time(), so
    exactly two calls happen regardless of epoch count - stub them with
    fixed values to make the elapsed duration deterministic.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])

    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = make_fixed_linear(weight=2.0, bias=0.0)
    loss_fn = MSE()
    optimizer = SGD(model.parameters(), lr=0.0)

    times = iter([100.0, 100.5])
    monkeypatch.setattr("orbit.nn.training.trainer.time.time", lambda: next(times))

    trainer = Trainer()
    trainer.fit(model, loss_fn, optimizer, dataloader, epochs=5)

    assert trainer.duration_seconds == 0.5
