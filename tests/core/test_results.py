import numpy as np
import json
from orbit.core import Results


def test_results_stores_fields():
    results = Results(
        name="run-1",
        final_loss=0.1234,
        loss_history=[1.0, 0.5, 0.1234],
        hyperparams={"epochs": 3, "lr": 0.1},
    )

    assert results.name == "run-1"
    assert results.final_loss == 0.1234
    assert results.loss_history == [1.0, 0.5, 0.1234]
    assert results.hyperparams == {"epochs": 3, "lr": 0.1}


def test_results_defaults_hyperparams_to_empty_dict():
    results = Results(name="run-2", final_loss=0.5, loss_history=[0.5])

    assert results.hyperparams == {}


def test_results_defaults_duration_seconds_to_none():
    results = Results(name="run-2", final_loss=0.5, loss_history=[0.5])

    assert results.duration_seconds is None


def test_results_defaults_gradient_norm_history_to_none():
    results = Results(name="run-2", final_loss=0.5, loss_history=[0.5])

    assert results.gradient_norm_history is None


def test_results_defaults_accuracy_history_to_none():
    results = Results(name="run-2", final_loss=0.5, loss_history=[0.5])

    assert results.accuracy_history is None


def test_results_defaults_test_loss_and_test_accuracy_to_none():
    results = Results(name="run-2", final_loss=0.5, loss_history=[0.5])

    assert results.test_loss is None
    assert results.test_accuracy is None


def test_results_repr_is_a_string_and_mentions_name():
    results = Results(name="run-3", final_loss=0.5, loss_history=[1.0, 0.5])

    text = repr(results)

    assert isinstance(text, str)
    assert "run-3" in text


def test_to_dict_casts_numpy_scalars_to_plain_floats():
    """
    A real Experiment.run() produces np.float64 in final_loss/loss_history
    (Trainer computes them via numpy arithmetic). json.dump can't serialize
    those directly, so to_dict() must cast them to plain float.
    """
    results = Results(
        name="numpy-run",
        final_loss=np.float64(0.1234),
        loss_history=[np.float64(1.0), np.float64(0.5)],
        hyperparams={"lr": 0.1},
    )

    data = results.to_dict()

    assert type(data["final_loss"]) is float
    assert all(type(loss) is float for loss in data["loss_history"])
    json.dumps(data)  # must not raise


def test_to_dict_includes_duration_seconds_cast_to_float():
    results = Results(
        name="numpy-run",
        final_loss=0.1234,
        loss_history=[1.0, 0.5],
        duration_seconds=np.float64(12.5),
    )

    data = results.to_dict()

    assert type(data["duration_seconds"]) is float
    assert data["duration_seconds"] == 12.5


def test_to_dict_duration_seconds_stays_none_when_unset():
    results = Results(name="run", final_loss=0.5, loss_history=[0.5])

    assert results.to_dict()["duration_seconds"] is None


def test_to_dict_includes_gradient_norm_history_cast_to_floats():
    results = Results(
        name="numpy-run",
        final_loss=0.1234,
        loss_history=[1.0, 0.5],
        gradient_norm_history=[np.float64(1.5), np.float64(0.9)],
    )

    data = results.to_dict()

    assert all(type(g) is float for g in data["gradient_norm_history"])
    assert data["gradient_norm_history"] == [1.5, 0.9]


def test_to_dict_gradient_norm_history_stays_none_when_unset():
    results = Results(name="run", final_loss=0.5, loss_history=[0.5])

    assert results.to_dict()["gradient_norm_history"] is None


def test_to_dict_includes_accuracy_history_cast_to_floats():
    results = Results(
        name="numpy-run",
        final_loss=0.1234,
        loss_history=[1.0, 0.5],
        accuracy_history=[np.float64(0.5), np.float64(0.75)],
    )

    data = results.to_dict()

    assert all(type(a) is float for a in data["accuracy_history"])
    assert data["accuracy_history"] == [0.5, 0.75]


def test_to_dict_accuracy_history_stays_none_when_unset():
    results = Results(name="run", final_loss=0.5, loss_history=[0.5])

    assert results.to_dict()["accuracy_history"] is None


def test_to_dict_includes_test_loss_and_test_accuracy_cast_to_floats():
    results = Results(
        name="numpy-run",
        final_loss=0.1234,
        loss_history=[1.0, 0.5],
        test_loss=np.float64(0.2),
        test_accuracy=np.float64(0.8),
    )

    data = results.to_dict()

    assert type(data["test_loss"]) is float
    assert type(data["test_accuracy"]) is float
    assert data["test_loss"] == 0.2
    assert data["test_accuracy"] == 0.8


def test_to_dict_test_loss_and_test_accuracy_stay_none_when_unset():
    results = Results(name="run", final_loss=0.5, loss_history=[0.5])

    data = results.to_dict()

    assert data["test_loss"] is None
    assert data["test_accuracy"] is None


def test_from_dict_round_trips_to_dict():
    original = Results(
        name="run-4",
        final_loss=0.42,
        loss_history=[1.0, 0.6, 0.42],
        hyperparams={"epochs": 3, "lr": 0.1},
        duration_seconds=3.75,
        gradient_norm_history=[2.0, 1.2, 0.4],
        accuracy_history=[0.5, 0.75, 0.9],
        test_loss=0.5,
        test_accuracy=0.85,
    )

    rebuilt = Results.from_dict(original.to_dict())

    assert rebuilt.name == original.name
    assert rebuilt.final_loss == original.final_loss
    assert rebuilt.loss_history == original.loss_history
    assert rebuilt.hyperparams == original.hyperparams
    assert rebuilt.duration_seconds == original.duration_seconds
    assert rebuilt.gradient_norm_history == original.gradient_norm_history
    assert rebuilt.accuracy_history == original.accuracy_history
    assert rebuilt.test_loss == original.test_loss
    assert rebuilt.test_accuracy == original.test_accuracy


def test_from_dict_defaults_duration_seconds_when_missing_from_old_data():
    old_data = {
        "name": "pre-duration-run",
        "final_loss": 0.5,
        "loss_history": [1.0, 0.5],
        "hyperparams": {},
    }

    rebuilt = Results.from_dict(old_data)

    assert rebuilt.duration_seconds is None


def test_from_dict_defaults_gradient_norm_history_when_missing_from_old_data():
    old_data = {
        "name": "pre-gradnorm-run",
        "final_loss": 0.5,
        "loss_history": [1.0, 0.5],
        "hyperparams": {},
        "duration_seconds": 1.0,
    }

    rebuilt = Results.from_dict(old_data)

    assert rebuilt.gradient_norm_history is None


def test_from_dict_defaults_accuracy_history_when_missing_from_old_data():
    old_data = {
        "name": "pre-accuracy-run",
        "final_loss": 0.5,
        "loss_history": [1.0, 0.5],
        "hyperparams": {},
        "duration_seconds": 1.0,
        "gradient_norm_history": [1.0, 0.5],
    }

    rebuilt = Results.from_dict(old_data)

    assert rebuilt.accuracy_history is None


def test_from_dict_defaults_test_loss_and_test_accuracy_when_missing_from_old_data():
    old_data = {
        "name": "pre-test-split-run",
        "final_loss": 0.5,
        "loss_history": [1.0, 0.5],
        "hyperparams": {},
        "duration_seconds": 1.0,
        "gradient_norm_history": [1.0, 0.5],
        "accuracy_history": None,
    }

    rebuilt = Results.from_dict(old_data)

    assert rebuilt.test_loss is None
    assert rebuilt.test_accuracy is None
