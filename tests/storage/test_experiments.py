import numpy as np
from orbit.core.results import Results
from orbit.storage.experiments import save_results, load_results
from orbit.core.dataset import TensorDataset
from orbit.core.dataloader import DataLoader
from orbit.nn.layers.linear import Linear
from orbit.nn.losses.mse import MSE
from orbit.nn.optimizers import SGD
from orbit.core.experiment import Experiment


def test_save_creates_parent_directories(tmp_path):
    results = Results(name="run", final_loss=0.5, loss_history=[1.0, 0.5])
    path = tmp_path / "runs" / "exp1" / "results.json"

    assert not path.parent.exists()

    save_results(results, path)

    assert path.exists()


def test_save_then_load_round_trips(tmp_path):
    results = Results(
        name="run-1",
        final_loss=0.1234,
        loss_history=[1.0, 0.5, 0.1234],
        hyperparams={"epochs": 3, "lr": 0.1, "batch_size": 2, "loss": "mse"},
    )
    path = tmp_path / "results.json"

    save_results(results, path)
    loaded = load_results(path)

    assert loaded.name == results.name
    assert loaded.final_loss == results.final_loss
    assert loaded.loss_history == results.loss_history
    assert loaded.hyperparams == results.hyperparams


def test_load_survives_a_real_experiment_run(tmp_path):
    """
    A real Experiment.run() produces np.float64 values in final_loss and
    loss_history (Trainer computes them via numpy arithmetic). This test
    would fail with a TypeError at save time if Results.to_dict() didn't
    cast those to plain floats before json.dump.
    """
    X = np.array([[1.0], [2.0], [3.0]])
    Y = np.array([[0.0], [0.0], [0.0]])
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.05)
    experiment = Experiment(model, MSE(), optimizer, dataloader, epochs=5, name="real-run")

    results = experiment.run()
    path = tmp_path / "real_run.json"

    save_results(results, path)
    loaded = load_results(path)

    assert np.isclose(loaded.final_loss, results.final_loss)
    assert loaded.hyperparams == results.hyperparams
