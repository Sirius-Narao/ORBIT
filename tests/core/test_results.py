import numpy as np
import json
from orbit.core.results import Results


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


def test_from_dict_round_trips_to_dict():
    original = Results(
        name="run-4",
        final_loss=0.42,
        loss_history=[1.0, 0.6, 0.42],
        hyperparams={"epochs": 3, "lr": 0.1},
    )

    rebuilt = Results.from_dict(original.to_dict())

    assert rebuilt.name == original.name
    assert rebuilt.final_loss == original.final_loss
    assert rebuilt.loss_history == original.loss_history
    assert rebuilt.hyperparams == original.hyperparams
