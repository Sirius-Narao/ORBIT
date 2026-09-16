import numpy as np
from orbit.core.dataset import TensorDataset
from orbit.core.dataloader import DataLoader
from orbit.nn.layers.linear import Linear
from orbit.nn.losses.mse import MSE
from orbit.nn.optimizers import SGD
from orbit.nn.training.trainer import Trainer


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
