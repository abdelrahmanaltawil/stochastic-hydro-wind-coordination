"""Check native simulator outputs have physical content and exact horizons."""

from pathlib import Path

import pytest

from src.helpers.energy.energy_simulation import run_energy_simulation
from src.helpers.water.water_simulation import load_water_network, run_water_simulation
from src.preprocessing import _run_epanet_presim

ROOT = Path(__file__).resolve().parents[1]
DSS = ROOT / "data/inputs/system_config/energy/master.dss"
WATER = ROOT / "data/inputs/system_config/water/ANET.inp"


def test_energy_simulation_reports_actual_injections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_energy_simulation(str(DSS), num_steps=2)
    assert Path.cwd() == tmp_path
    assert result["time_hours"] == [0, 1]
    assert result["node"]["P_kw"]["loadbus"] == pytest.approx([-400.0, -350.0], abs=0.01)
    assert result["node"]["P_kw"]["sourcebus"][0] > 400
    assert result["link"]["current_amps"]["l1"][0] > 0
    assert result["losses_kw"][0] > 0


def test_energy_simulation_empty_lines_and_snapshot_mode(tmp_path):
    source = tmp_path / "source.dss"
    source.write_text("Clear\nNew Circuit.One bus1=source basekv=11 phases=3\nSet Voltagebases=[11]\nCalcvoltagebases\n")
    result = run_energy_simulation(str(source), mode="snap")
    assert result["link"]["current_amps"] == {}
    assert len(result["node"]["voltage_pu"]["source"]) == 1


def test_epanet_presimulation_exact_horizon_and_no_working_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sim = _run_epanet_presim(str(WATER), 2)
    assert sim is not None
    assert list(sim["flowrate"].index) == [0, 1]
    assert "head" in sim and "status" in sim
    assert list(tmp_path.iterdir()) == []


def test_water_simulator_rejects_misspelled_mode():
    wn = load_water_network(str(WATER))
    with pytest.raises(ValueError, match="Unknown simulator_type"):
        run_water_simulation(wn, "typo")


def test_legacy_comparison_rejects_truncated_and_nonfinite_series():
    from tests.helpers.validation_utils import compare_series
    with pytest.raises(ValueError, match="same nonzero shape"):
        compare_series([1, 2], [1])
    with pytest.raises(ValueError, match="finite"):
        compare_series([float("nan")], [1])


def test_legacy_comparison_requires_matching_times():
    import pandas as pd
    from tests.helpers.validation_utils import compute_comparison_metrics
    with pytest.raises(ValueError, match="Time indices"):
        compute_comparison_metrics({"head": pd.DataFrame({"tank": [1]}, index=[0])},
                                   {"head": pd.DataFrame({"tank": [1]}, index=[3600])})


def test_legacy_energy_runner_uses_independent_replay():
    import pyomo.environ as pyo
    from tests.helpers.validation_utils import run_energy_validation
    if not pyo.SolverFactory("highs").available(exception_flag=False):
        pytest.skip("HiGHS is unavailable")
    result = run_energy_validation(str(DSS), num_timesteps=2, solver="highs")
    assert result["replay_metrics"]["all_snapshots_converged"]
    assert result["replay_metrics"]["max_grid_import_error_kw"] > 0
    assert len(result["opt_data"]["voltage_pu"]) == 2
    assert result["metrics"]["voltage_pu"]["loadbus"]["n"] == 2
