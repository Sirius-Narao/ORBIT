import json

from orbit.core import Results
from orbit.cli.commands.list import list_experiments


def write_results(exp_dir, final_loss=0.1234, loss_history=None, duration_seconds=None):
    results_dir = exp_dir / "results"
    results_dir.mkdir(parents=True)

    results = Results(
        name=exp_dir.name,
        final_loss=final_loss,
        loss_history=loss_history if loss_history is not None else [0.5, 0.3, final_loss],
        duration_seconds=duration_seconds,
    )

    with open(results_dir / "results.json", "w") as f:
        json.dump(results.to_dict(), f)


def test_list_experiments_shows_done_and_not_run(tmp_path, capsys):
    (tmp_path / "pending_exp").mkdir(parents=True)
    write_results(tmp_path / "done_exp", final_loss=0.1234, loss_history=[0.5, 0.3, 0.1234])

    list_experiments(root=tmp_path)

    out = capsys.readouterr().out
    assert "done_exp" in out
    assert "done" in out
    assert "0.1234" in out
    assert "3" in out  # epoch count
    assert "pending_exp" in out
    assert "not run" in out


def test_list_experiments_shows_duration_when_recorded(tmp_path, capsys):
    write_results(tmp_path / "timed_exp", final_loss=0.5, loss_history=[0.5], duration_seconds=7.89)

    list_experiments(root=tmp_path)

    out = capsys.readouterr().out
    assert "7.89s" in out


def test_list_experiments_shows_dash_for_missing_duration(tmp_path, capsys):
    write_results(tmp_path / "untimed_exp", final_loss=0.5, loss_history=[0.5])

    list_experiments(root=tmp_path)

    out = capsys.readouterr().out
    assert "untimed_exp" in out


def test_list_experiments_empty_dir(tmp_path, capsys):
    list_experiments(root=tmp_path)

    out = capsys.readouterr().out
    assert "No experiments found" in out


def test_list_experiments_missing_root(tmp_path, capsys):
    list_experiments(root=tmp_path / "does_not_exist")

    out = capsys.readouterr().out
    assert "No experiments found" in out


def test_list_experiments_ignores_non_directory_entries(tmp_path, capsys):
    (tmp_path / "real_exp").mkdir(parents=True)
    (tmp_path / "stray_file.txt").write_text("not an experiment")

    list_experiments(root=tmp_path)

    out = capsys.readouterr().out
    assert "real_exp" in out
    assert "stray_file.txt" not in out
