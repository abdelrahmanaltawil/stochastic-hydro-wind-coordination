"""Conservation and formulation regressions on reproducible small networks.

These checks validate the optimization itself; nonlinear replay is a separate
experiment and never inferred merely from a successful mathematical solve.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest
import wntr

from src.algorithm_tasks import build_model, solve_model, _mean_pump_powers_from_sim
from src.helpers.water.hydraulic_utils import (
    add_pwl_constraint, create_piecewise_pipe_curve, create_piecewise_pump_curve,
)
from src.helpers.energy.power_utils import calc_line_admittance, create_pwl_current_segments


@pytest.fixture(scope="module", params=("highs", "glpk"))
def solver(request):
    if pyo.SolverFactory(request.param).available(exception_flag=False):
        return request.param
    pytest.skip(f"{request.param} is not installed")


def energy_data(T=1, buses=("source",), loads=None, lines=(), battery=None):
    return {
        "buses": list(buses), "lines": list(lines),
        "loads": loads or {b: [10.] * T for b in buses},
        "storage": {"capacity_kwh": 0., "max_charge_kw": 0., "max_discharge_kw": 0.,
                    "charge_efficiency": 1., "discharge_efficiency": 1.,
                    "initial_soc_frac": 0., **(battery or {})},
        "tariff": [0.1] * T,
        "network": {"base_power_kw": 1000., "reference_bus": buses[0]},
    }


def energy_model(data, T=1):
    return build_model({"energy": data}, {"run_water": False, "run_energy": True, "T": T})


def solve(model, solver):
    model, results, _ = solve_model(model, solver=solver, timeout=30)
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal
    return model


def test_last_interval_cannot_discharge_free_energy(solver):
    data = energy_data(battery={"capacity_kwh": 100., "max_discharge_kw": 50., "initial_soc_frac": 1.})
    m = solve(energy_model(data), solver)
    assert list(m.StateT) == [0, 1]
    assert pyo.value(m.Q_dis["source", 0]) == pytest.approx(0.)
    assert pyo.value(m.P_import["source", 0]) == pytest.approx(10.)
    assert pyo.value(m.E_soc["source", 1]) == pytest.approx(100.)


def test_storage_efficiencies_applied_once_and_all_hours_conserved(solver):
    data = energy_data(T=2, loads={"source": [0., 10.]}, battery={
        "capacity_kwh": 20., "max_charge_kw": 10., "max_discharge_kw": 10.,
        "charge_efficiency": .8, "discharge_efficiency": .5,
    })
    data["tariff"] = [.1, 1.]
    m = solve(energy_model(data, 2), solver)
    assert pyo.value(m.Q_ch["source", 0]) == pytest.approx(10.)
    assert pyo.value(m.Q_dis["source", 1]) == pytest.approx(4.)
    assert pyo.value(m.P_import["source", 1]) == pytest.approx(6.)
    assert pyo.value(m.E_soc["source", 2]) == pytest.approx(0.)
    assert pyo.value(m.objective) == pytest.approx(7.)


def test_feeder_transfers_active_and_reactive_power_with_correct_signs(solver):
    data = energy_data(buses=("source_bus", "load_bus"), loads={"source_bus": [0.], "load_bus": [100.]},
                       lines=[("source_bus", "load_bus", .01, .02, 1.)])
    data["reactive_loads"] = {"load_bus": [40.]}
    m = solve(energy_model(data), solver)
    line = next(iter(m.ELines))
    assert m.line_endpoints[line] == ("source_bus", "load_bus")
    assert pyo.value(m.P_import["source_bus", 0]) == pytest.approx(100.)
    assert pyo.value(m.P_import["load_bus", 0]) == 0
    assert pyo.value(m.P_line[line, 0]) == pytest.approx(100.)
    assert pyo.value(m.Q_line[line, 0]) == pytest.approx(40.)
    assert pyo.value(m.Q_grid["source_bus", 0]) == pytest.approx(40.)
    assert pyo.value(m.I_re[line, 0]) == pytest.approx(.1)
    assert pyo.value(m.I_im[line, 0]) == pytest.approx(-.04)
    assert pyo.value(m.U["load_bus", 0]) < 1.
    assert all(c.body.polynomial_degree() in (0, 1)
               for c in m.component_data_objects(pyo.Constraint, active=True))


def test_line_thermal_limit_cannot_be_bypassed_by_local_import(solver):
    data = energy_data(buses=("s", "d"), loads={"s": [0.], "d": [100.]},
                       lines=[("s", "d", .01, .01, .05)])
    m, results, _ = solve_model(energy_model(data), solver=solver, timeout=30)
    assert results.solver.termination_condition == pyo.TerminationCondition.infeasible
    assert m.P_import["d", 0].fixed


def test_export_is_priced(solver):
    data = energy_data(buses=("s",), loads={"s": [0.]})
    data["pv_profile"] = {"s": [5.]}
    data["export_tariff"] = [.05]
    m = solve(energy_model(data), solver)
    assert pyo.value(m.P_export["s", 0]) == pytest.approx(5.)
    assert pyo.value(m.objective) == pytest.approx(-.25)


def test_parallel_lines_and_underscored_buses_have_distinct_identifiers():
    data = energy_data(buses=("s_bus", "d_bus"),
                       lines=[("s_bus", "d_bus", .01, .02, 1.)] * 2)
    m = energy_model(data)
    assert len(m.ELines) == 2
    assert set(m.line_endpoints.values()) == {("s_bus", "d_bus")}


@pytest.mark.parametrize("T", [0, -1, 1.5, True])
def test_invalid_horizon_rejected(T):
    with pytest.raises(ValueError, match="positive integer"):
        energy_model(energy_data(), T)


def test_invalid_profile_and_efficiency_rejected():
    data = energy_data()
    data["tariff"] = [1., 2.]
    with pytest.raises(ValueError, match="hourly values"):
        energy_model(data)
    data["tariff"] = [1.]
    data["storage"]["charge_efficiency"] = 0.
    with pytest.raises(ValueError, match="battery"):
        energy_model(data)


def water_tank_file(tmp_path, initial=5., demand=.001):
    wn = wntr.network.WaterNetworkModel()
    wn.add_tank("tank", elevation=30., init_level=initial, min_level=0., max_level=5., diameter=10.)
    wn.add_junction("demand", base_demand=demand, elevation=0.)
    wn.add_pipe("pipe", "tank", "demand", length=100., diameter=.3, roughness=130.)
    path = tmp_path / "tank.inp"
    wntr.network.write_inpfile(wn, str(path))
    return path


def tank_model(path, T=1, closure=False):
    return build_model({"water": {"inp_file": str(path), "pipe_max_flows": {"pipe": .003}}},
                       {"T": T, "run_water": True, "water": {"terminal_tank_closure": closure}})


def test_final_hour_tank_water_is_counted_and_full_tank_can_discharge(tmp_path, solver):
    m = solve(tank_model(water_tank_file(tmp_path), T=2), solver)
    expected = 35. - 2 * 3600 * .001 / (np.pi * 25)
    assert pyo.value(m.H_terminal["tank"]) == pytest.approx(expected)
    assert pyo.value(m.Q["pipe", 0]) == pytest.approx(.001)
    assert pyo.value(m.Q["pipe", 1]) == pytest.approx(.001)
    assert all(v.fixed and pyo.value(v) == 0 for v in m.SlackPos.values())


def test_tank_terminal_bounds_prevent_last_hour_overdraw(tmp_path, solver):
    m = tank_model(water_tank_file(tmp_path, initial=.01))
    _, result, _ = solve_model(m, solver=solver)
    assert result.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_tank_closure_prevents_using_initial_inventory_as_free_source(tmp_path, solver):
    m = tank_model(water_tank_file(tmp_path), closure=True)
    _, result, _ = solve_model(m, solver=solver)
    assert result.solver.termination_condition == pyo.TerminationCondition.infeasible


def test_pipe_curve_contains_exact_origin_at_small_flow():
    pts = create_piecewise_pipe_curve(1e6, 1e-8, num_segments=3)
    assert (0., 0.) in pts
    assert len({x for x, _ in pts}) == len(pts)


def test_pwl_on_off_pump_gain_is_zero_when_off(solver):
    m = pyo.ConcreteModel()
    m.x = pyo.Var(); m.y = pyo.Var(); m.on = pyo.Var(domain=pyo.Binary)
    m.on.fix(0)
    add_pwl_constraint(m, "pump", m.x, m.y, [(0., 10.), (1., 0.)], activation=m.on)
    m.objective = pyo.Objective(expr=0.)
    solve(m, solver)
    assert pyo.value(m.x) == pytest.approx(0.)
    assert pyo.value(m.y) == pytest.approx(0.)


def test_single_point_pump_curve_matches_epanet():
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("r", base_head=0.); wn.add_junction("j")
    wn.add_curve("curve", "HEAD", [(0.05, 40.)])
    wn.add_pump("p", "r", "j", pump_type="HEAD", pump_parameter="curve")
    pts = create_piecewise_pump_curve(wn.get_link("p"), 6)
    assert pts[0] == pytest.approx((0., 160./3.))
    assert pts[-1] == pytest.approx((.1, 0.))


def test_pump_mean_power_calibration_and_efficiency():
    q = pd.DataFrame({"p": [.1, 0., .2]})
    h = pd.DataFrame({"p": [-20., 0., -10.]})
    assert _mean_pump_powers_from_sim(q, h, ["p"], .8)["p"] == pytest.approx(24.525)
    with pytest.raises(ValueError, match="efficiency"):
        _mean_pump_powers_from_sim(q, h, ["p"], 0.)


def test_nexus_adds_actual_load_and_expands_derived_import_limit(monkeypatch, tmp_path, solver):
    # A fixed-flow pump system permits an analytical coupled electricity check.
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("r", base_head=50.)
    wn.add_junction("j", base_demand=.01, elevation=0.)
    wn.add_curve("curve", "HEAD", [(.01, 10.)])
    wn.add_pump("pump", "r", "j", pump_type="HEAD", pump_parameter="curve")
    path = tmp_path / "pump.inp"; wntr.network.write_inpfile(wn, str(path))
    sim = {"flowrate": pd.DataFrame({"pump": [.01]}),
           "headloss": pd.DataFrame({"pump": [-10.]})}
    data = {"water": {"inp_file": str(path), "epanet_sim": sim}, "energy": energy_data()}
    m = build_model(data, {"T": 1, "run_nexus": True, "nexus": {"pump_power_kw": {"pump": 25.}}})
    solve(m, solver)
    assert pyo.value(m.PumpBusLoad["source", 0]) == pytest.approx(25.)
    assert pyo.value(m.P_import["source", 0]) == pytest.approx(35.)
    assert pyo.value(m.water_cost) == 0.
    assert pyo.value(m.objective) == pytest.approx(3.5)
    assert m.pump_bus == {"pump": "source"}
    with pytest.raises(ValueError, match="unknown electrical bus"):
        build_model(data, {"T": 1, "run_nexus": True, "nexus": {"pump_bus": {"pump": "typo"}}})


@pytest.mark.parametrize("points", [[], [(1., 1.)], [(0., 0.), (0., 1.)]])
def test_invalid_pwl_rejected(points):
    m = pyo.ConcreteModel(); m.x = pyo.Var(); m.y = pyo.Var()
    with pytest.raises(ValueError):
        add_pwl_constraint(m, "bad", m.x, m.y, points)


def test_electrical_helpers_validate_domains_and_keep_small_values():
    with pytest.raises(ValueError): calc_line_admittance(0., 0.)
    with pytest.raises(ValueError): create_pwl_current_segments(0.)
    pts = create_pwl_current_segments(1e-8, 3)
    assert pts[1][1] > 0.


def configurable_water_network():
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("r", base_head=30.)
    for name in ("a", "b", "c"):
        wn.add_junction(name, elevation=0.)
    wn.add_curve("curve", "HEAD", [(.01, 10.)])
    wn.add_pump("pump", "r", "a", pump_type="HEAD", pump_parameter="curve")
    wn.add_pipe("pipe", "a", "b", length=100., diameter=.3, roughness=130.)
    wn.add_pipe("last", "b", "c", length=100., diameter=.3, roughness=130.)
    return wn


@pytest.mark.parametrize("feature,match", [
    ("check_valve", "check valves"), ("closed_pipe", "closed pipes"),
    ("pipe_minor_loss", "minor-loss"), ("nonunit_speed", "nonunit speed"),
    ("speed_pattern", "speed patterns"), ("valve_type", "provisional PRV"),
    ("valve_minor_loss", "minor-loss"), ("closed_valve", "externally closed"),
    ("pipe_control", "pump-status controls"), ("speed_control", "pump-status controls"),
    ("emitter", "emitters"), ("pressure_demand", "Pressure-dependent"),
])
def test_unsupported_hydraulics_are_rejected(feature, match):
    from src.algorithm_tasks import _validate_water_network
    wn = configurable_water_network()
    if feature == "check_valve": wn.get_link("pipe").check_valve = True
    elif feature == "closed_pipe": wn.get_link("pipe").initial_status = wntr.network.LinkStatus.Closed
    elif feature == "pipe_minor_loss": wn.get_link("pipe").minor_loss = 2.
    elif feature == "nonunit_speed": wn.get_link("pump").base_speed = .8
    elif feature == "speed_pattern":
        wn.add_pattern("speed", [1., .8]); wn.get_link("pump").speed_pattern_name = "speed"
    elif feature.startswith("valve") or feature == "closed_valve":
        wn.remove_link("last")
        wn.add_valve("valve", "b", "c", diameter=.3,
                     valve_type="TCV" if feature == "valve_type" else "PRV", initial_setting=20.)
        if feature == "valve_minor_loss": wn.get_link("valve").minor_loss = 2.
        if feature == "closed_valve": wn.get_link("valve").initial_status = wntr.network.LinkStatus.Closed
    elif feature in ("pipe_control", "speed_control"):
        controls = wntr.network.controls
        target = wn.get_link("pipe" if feature == "pipe_control" else "pump")
        attribute = "status" if feature == "pipe_control" else "base_speed"
        action = controls.ControlAction(target, attribute, 0)
        wn.add_control("unsupported", controls.Control(controls.SimTimeCondition(wn, "=", 3600), action))
    elif feature == "emitter": wn.get_node("a").emitter_coefficient = .1
    elif feature == "pressure_demand": wn.options.hydraulic.demand_model = "PDD"
    with pytest.raises(ValueError, match=match):
        _validate_water_network(wn)


def test_supported_prv_and_pump_status_controls_remain_available():
    from src.algorithm_tasks import _validate_water_network
    wn = configurable_water_network()
    wn.remove_link("last")
    wn.add_valve("prv", "b", "c", diameter=.3, valve_type="PRV", initial_setting=20.)
    controls = wntr.network.controls
    action = controls.ControlAction(wn.get_link("pump"), "status", wntr.network.LinkStatus.Closed)
    wn.add_control("schedule", controls.Control(controls.SimTimeCondition(wn, "=", 3600), action))
    _validate_water_network(wn)
