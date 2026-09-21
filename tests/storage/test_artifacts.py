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
