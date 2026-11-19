"""Check output routing without reading private assets or running training."""
from pathlib import Path

import pytest

import run_all
import run_hetero
import run_perturbations
from train_hetero import parse_args


@pytest.mark.parametrize("scenario", ["rotation", "color"])
def test_single_hetero_default(scenario):
    args = parse_args(["--scenario", scenario, "--method", "equal"])
    assert args.runs_dir == Path("runs") / scenario


@pytest.mark.parametrize("quick", [False, True])
@pytest.mark.parametrize("scenario", ["rotation", "color"])
def test_hetero_batch_routing(monkeypatch, capsys, scenario, quick):
    commands = []
    monkeypatch.setattr(run_hetero.subprocess, "run", lambda cmd, **kw: commands.append(cmd))
    # Dry-run skips assets and subprocesses; inspect its emitted commands instead.
    args = ["--scenario", scenario, "--methods", "equal", "--seeds", "0", "--dry-run"]
    run_hetero.main(args + (["--quick"] if quick else []))
    output = capsys.readouterr().out.replace("\\", "/")
    suffix = scenario + ("_quick" if quick else "")
    assert f"runs/{suffix}" in output
    assert f"results/{suffix}" in output
    assert not commands


@pytest.mark.parametrize("quick", [False, True])
def test_clean_batch_routing(capsys, quick):
    run_all.main(["--methods", "equal", "--seeds", "0", "--dry-run"]
                 + (["--quick"] if quick else []))
    output = capsys.readouterr().out.replace("\\", "/")
    suffix = "clean_quick" if quick else "clean"
    assert f"runs/{suffix}" in output
    assert f"results/{suffix}" in output


def test_perturbation_batch_routing(monkeypatch, capsys):
    monkeypatch.setattr(run_perturbations.subprocess, "run", lambda *a, **kw: None)
    run_perturbations.main(["--scenarios", "fine20", "--methods", "equal", "--dry-run"])
    output = capsys.readouterr().out.replace("\\", "/")
    assert "runs/perturbations/fine20" in output
    assert "results/perturbations/fine20" in output
