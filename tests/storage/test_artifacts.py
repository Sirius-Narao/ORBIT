import numpy as np
import pytest

from orbit.nn import Sequential
from orbit.nn.layers import Linear
from orbit.nn.activations import Tanh
from orbit.storage import save_checkpoint, load_checkpoint, checkpoint_exists


def make_model():
    return Sequential(Linear(2, 8), Tanh(), Linear(8, 1))


def test_checkpoint_exists_false_before_saving(tmp_path):
    assert checkpoint_exists("xor_test", root=tmp_path) is False


def test_save_checkpoint_creates_file_and_checkpoint_exists_true(tmp_path):
    model = make_model()

    path = save_checkpoint("xor_test", model, root=tmp_path)

    assert path.exists()
    assert path == tmp_path / "xor_test" / "checkpoints" / "model.npz"
    assert checkpoint_exists("xor_test", root=tmp_path) is True


def test_save_then_load_round_trips_weights(tmp_path):
    trained = make_model()
    save_checkpoint("xor_test", trained, root=tmp_path)

    original_weights = [param.data.copy() for param in trained.parameters()]

    # A freshly (randomly) initialized model with the same architecture -
    # loading the checkpoint should overwrite its random weights with the
    # exact values that were saved.
    fresh = make_model()
    load_checkpoint("xor_test", fresh, root=tmp_path)

    for loaded_param, original in zip(fresh.parameters(), original_weights):
        assert np.array_equal(loaded_param.data, original)


def test_load_checkpoint_raises_file_not_found_when_missing(tmp_path):
    model = make_model()

    with pytest.raises(FileNotFoundError):
        load_checkpoint("does_not_exist", model, root=tmp_path)


def test_snapshots_round_trip(tmp_path):
    from orbit.nn.training.snapshots import SnapshotRecorder
    from orbit.storage import load_snapshots, save_snapshots, snapshots_exist

    model = make_model()
    recorder = SnapshotRecorder(epochs=3)
    recorder.record(0, model, float("nan"))
    for epoch in range(1, 4):
        for param in model.parameters():
            param.data = param.data + 1.0
        recorder.record(epoch, model, 1.0 / epoch)

    assert not snapshots_exist("xor_test", root=tmp_path)
    save_snapshots("xor_test", recorder, root=tmp_path)
    loaded = load_snapshots("xor_test", root=tmp_path)

    assert snapshots_exist("xor_test", root=tmp_path)
    assert list(loaded["epochs"]) == [0, 1, 2, 3]
    assert np.isnan(loaded["loss"][0]) and np.isclose(loaded["loss"][3], 1 / 3)
    assert set(loaded["params"]) == {name for name, _ in model.named_parameters()}
    assert loaded["params"]["0.weight"].shape == (4, 2, 8)
    assert np.array_equal(loaded["params"]["0.weight"][-1], model._modules["0"].weight.data)


def test_load_snapshots_raises_when_missing(tmp_path):
    from orbit.storage import load_snapshots

    with pytest.raises(FileNotFoundError):
        load_snapshots("xor_test", root=tmp_path)
