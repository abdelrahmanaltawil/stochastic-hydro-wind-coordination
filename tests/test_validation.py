"""Checks for the independent simulator replay and residual acceptance gate."""
import math

import pandas as pd
import pyomo.environ as pyo
import pytest

from src.algorithm_tasks import build_model, solve_model
from src.preprocessing import load_networks
from src.validation import algebraic_residuals, replay_energy, replay_water
from src.workflow import load_config


def test_residuals_detect_constraint_bound_and_integrality_errors():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(domain=pyo.Binary)
    model.y = pyo.Var(bounds=(0, 2))
    model.limit = pyo.Constraint(expr=model.x + model.y <= 1)
    model.x.set_value(.25, skip_validation=True)
    model.y.set_value(3, skip_validation=True)
    assert algebraic_residuals(model) == {
        "max_constraint_violation": 2.25,
        "max_bound_violation": 1.,
        "max_integrality_violation": .25,
    }


def test_residuals_reject_nonfinite_values():
    model = pyo.ConcreteModel()
    model.x = pyo.Var()
    model.x.set_value(float('nan'), skip_validation=True)
    with pytest.raises(ValueError, match="Nonfinite"):
        algebraic_residuals(model)


def test_actual_two_hour_schedule_replays_both_networks(tmp_path):
    config = load_config(solver="highs")
    config["T"] = 2
    config["water"]["min_pressure_m"] = 20
    data = load_networks(config)
    model, results, _ = solve_model(build_model(data, config), solver="highs", timeout=15)
    assert str(results.solver.termination_condition) == "optimal"
    assert algebraic_residuals(model)["max_constraint_violation"] < 1e-5
    water = replay_water(model, data["water"]["inp_file"], config, tmp_path)
    energy = replay_energy(model, data["energy"], tmp_path)
    assert energy["all_snapshots_converged"]
    assert energy["voltage_violations"] == 0
    assert water["pressure_violations_at_report_times"] == 0
    assert math.isfinite(water["terminal_tank_change_m"]["T1"])
    assert pd.read_csv(tmp_path / "water_replay_heads.csv").time_seconds.tolist() == [0, 3600, 7200]
    assert len(pd.read_csv(tmp_path / "energy_replay_voltages.csv")) == 4
