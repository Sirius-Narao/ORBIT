import json

import numpy as np

from orbit.cli.commands.run import run_experiment
from orbit.cli.commands.test import test_experiment as evaluate_experiment


def write_config(root, name, normalize):
    exp_dir = root / name
    exp_dir.mkdir(parents=True)
    config = {
        "name": name,
        "dataset": "xor",
        "model": [
            {"type": "Linear", "in_features": 2, "neurons": 8},
            {"type": "Tanh"},
            {"type": "Linear", "neurons": 1},
            {"type": "Sigmoid"},
        ],
        "loss": "MSE",
        "optimizer": "SGD",
        "learning_rate": 2.0,
        "batch_size": 4,
        "epochs": 50,
        "seed": 11,
        "test_split": 0.25,
        "normalize": normalize,
    }
    with open(exp_dir / "experiment.json", "w") as f:
        json.dump(config, f)


def test_orbit_test_reproduces_run_test_loss_with_normalization(tmp_path):
    """
    Normalization stats are never saved to disk - orbit test rebuilds them
    by replaying load_experiment's seeded split. If that replay produced
    different stats (or normalized with the wrong split), the reloaded
    checkpoint would see differently-scaled test inputs and this test loss
    would not match the one orbit run computed right after training.
    """
    write_config(tmp_path, "normalized", normalize="standard")

    run_results = run_experiment("normalized", root=tmp_path)
    test_results = evaluate_experiment("normalized", root=tmp_path)

    assert np.isclose(test_results.test_loss, run_results.test_loss)
