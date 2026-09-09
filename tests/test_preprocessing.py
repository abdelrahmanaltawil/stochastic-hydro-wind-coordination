"""Regression tests for physical input data and reproducible path handling."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.preprocessing import build_network_data, create_run_directory, _build_energy_data, _profile

ROOT = Path(__file__).resolve().parents[1]
DSS = ROOT / "data/inputs/system_config/energy/master.dss"


def test_create_run_directory(tmp_path):
    first = create_run_directory(str(tmp_path / "output"))
    second = create_run_directory(str(tmp_path / "output"))
    assert first != second
    assert first.name.startswith("run_")
    assert (first / "metadata").is_dir()


def test_build_network_data_resolves_water_against_root(tmp_path, monkeypatch):
    (tmp_path / "network.inp").write_text("[TITLE]\nTest\n")
    config = {"run_water": True, "run_energy": False, "water": {"network": "network.inp"}, "T": 2}
    monkeypatch.chdir(ROOT)
    with patch("src.preprocessing._run_epanet_presim", return_value=None):
        data = build_network_data(config, project_root=tmp_path)
    assert data["water"]["inp_file"] == str(tmp_path / "network.inp")


def test_build_network_data_missing_water_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_network_data({"run_water": True, "water": {"network": "missing.inp"}, "T": 24}, tmp_path)


def test_energy_topology_units_and_hourly_loads():
    data = _build_energy_data({}, DSS, 25)
    assert data["buses"] == ["sourcebus", "loadbus"]
    assert data["loads"]["sourcebus"] == [0.0] * 25
    assert data["loads"]["loadbus"][:3] == [400.0, 350.0, 350.0]
    assert data["loads"]["loadbus"][24] == 400.0
    n, m, r, x, ampacity = data["lines"][0]
    assert (n, m) == ("sourcebus", "loadbus")
    assert r == pytest.approx(0.063 / (115 ** 2))
    assert x == pytest.approx(0.09 / (115 ** 2))
    assert ampacity * data["network"]["base_current_amps"] == pytest.approx(400)
    assert data["network"]["grid_buses"] == ["sourcebus"]
    assert data["reactive_loads"]["loadbus"][0] == pytest.approx(193.7288419)


def test_energy_load_paths_preserve_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = build_network_data({"run_water": False, "run_energy": True, "energy": {"network": str(DSS)}, "T": 2})
    assert Path.cwd() == tmp_path
    assert data["energy"]["dss_file"] == str(DSS)


def test_technology_capacity_is_not_duplicated_at_every_bus():
    data = _build_energy_data({"technologies": {
        "pv": {"capacity_kw": 100, "irradiance_profile": [0.5, 1]},
        "storage": {"capacity_kwh": 200, "max_charge_kw": 50, "max_discharge_kw": 50},
    }}, DSS, 2)
    assert data["pv_profile"] == {"sourcebus": [0.0, 0.0], "loadbus": [50.0, 100.0]}
    assert sum(spec["capacity_kwh"] for spec in data["storage_by_bus"].values()) == 200
    assert data["storage_by_bus"]["sourcebus"]["max_discharge_kw"] == 0


@pytest.mark.parametrize("values", [[0.1, 0.2, 0.3], [float("nan"), 0.1], [-0.1, 1.0]])
def test_invalid_profile_is_rejected(values):
    with pytest.raises(ValueError):
        _profile(values, 2, "test profile", minimum=0)


def test_profile_repeats_days_and_expands_scalar():
    assert _profile(list(range(24)), 25, "profile")[-1] == 0
    assert _profile(0.2, 2, "tariff") == [0.2, 0.2]


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError, match="T must"):
        build_network_data({"T": horizon})


def test_missing_dss_circuit_is_rejected(tmp_path):
    empty = tmp_path / "empty.dss"
    empty.write_text("Clear\n! Missing circuit\n")
    with pytest.raises(ValueError, match="circuit"):
        _build_energy_data({}, empty, 2)


def test_missing_storage_means_no_battery():
    data = _build_energy_data({}, DSS, 2)
    assert all(spec["capacity_kwh"] == 0 for spec in data["storage_by_bus"].values())
