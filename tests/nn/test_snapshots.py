import numpy as np

from orbit.nn import Sequential
from orbit.nn.layers import Linear
from orbit.nn.activations import Tanh
from orbit.nn.training.snapshots import MAX_SNAPSHOTS, SnapshotRecorder, parameter_count, snapshot_interval


def test_snapshot_interval_caps_the_number_of_snapshots():
    assert snapshot_interval(50) == 1
    assert snapshot_interval(3000) == 30
    assert 3000 // snapshot_interval(3000) <= MAX_SNAPSHOTS


def test_parameter_count():
    # (2*8 + 8) + (8*1 + 1) = 33
    assert parameter_count(Sequential(Linear(2, 8), Tanh(), Linear(8, 1))) == 33


def test_recorder_keeps_every_nth_epoch_plus_epoch_zero_and_the_last():
    model = Sequential(Linear(2, 1))
    recorder = SnapshotRecorder(epochs=250)  # every 2 epochs

    recorder.record(0, model, float("nan"))
    for epoch in range(1, 251):
        recorder.record(epoch, model, 1.0 / epoch)

    assert recorder.every == 2
    assert recorder.epochs[:3] == [0, 2, 4]
    assert recorder.epochs[-1] == 250
    assert len(recorder) == 126
    assert np.isnan(recorder.losses[0])
    assert recorder.losses[1] == 0.5


def test_recorder_always_keeps_the_last_epoch_off_the_interval():
    recorder = SnapshotRecorder(epochs=205)  # every 2 epochs; 205 is odd
    model = Sequential(Linear(2, 1))

    for epoch in range(1, 206):
        recorder.record(epoch, model, 0.0)

    assert recorder.epochs[-1] == 205


def test_recorder_copies_parameters_rather_than_aliasing_them():
    model = Sequential(Linear(2, 1))
    recorder = SnapshotRecorder(epochs=2)

    recorder.record(1, model, 0.0)
    model._modules["0"].weight.data = model._modules["0"].weight.data + 5.0
    recorder.record(2, model, 0.0)

    first, second = recorder.params["0.weight"]
    assert np.allclose(second - first, 5.0)
