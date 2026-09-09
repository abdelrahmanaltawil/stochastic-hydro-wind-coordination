"""Configuration regressions for both command-line entry points."""
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from src.workflow import PROJECT_ROOT, load_config


def test_default_config_resolves_paths_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(solver="highs", output_dir=tmp_path / "runs")
    assert Path(config["water"]["network"]).is_file()
    assert Path(config["energy"]["network"]).is_file()
    assert config["paths"]["output_dir"] == str(tmp_path / "runs")
    assert config["solver"]["name"] == "highs"


@pytest.mark.parametrize("horizon", [0, -1, 2.5, True])
def test_invalid_horizon_rejected(tmp_path, horizon):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"T": horizon}))
    with pytest.raises(ValueError, match="positive integer"):
        load_config(path)


def test_direct_script_help_from_other_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src/workflow.py"), "--help"],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--solver" in result.stdout


def test_infeasible_run_saves_failure_and_metadata(tmp_path, monkeypatch):
    import json
    import pyomo.environ as pyo
    from src import workflow

    config = load_config(solver="highs", output_dir=tmp_path)
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(0, 1))
    model.impossible = pyo.Constraint(expr=model.x >= 2)
    model.objective = pyo.Objective(expr=model.x)
    monkeypatch.setattr(workflow.preprocessing, "load_networks", lambda _: {})
    monkeypatch.setattr(workflow.algorithm, "build_model", lambda *_: model)
    with pytest.raises(RuntimeError, match="no feasible incumbent"):
        workflow.run_pipeline(config)
    run_dir = next(tmp_path.iterdir())
    failure = json.loads((run_dir / "00_failure.json").read_text())
    assert failure["status"] == "failed"
    assert "infeasible" in failure["message"]
    assert not (run_dir / "00_summary.json").exists()
    assert (run_dir / "metadata/00_run_metadata.yaml").is_file()
