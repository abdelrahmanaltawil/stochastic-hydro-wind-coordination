"""Two-operator coordination layer: contract accounting, subproblems, and the master."""
import copy

import pandas as pd
import pyomo.environ as pyo
import pytest
import wntr

from src import algorithm_tasks as algorithm
from src import coordination as co


@pytest.fixture(scope="module")
def solver():
    if pyo.SolverFactory("highs").available(exception_flag=False):
        return "highs"
    pytest.skip("highs is not installed")


def toy_site(tmp_path, T=3):
    """One pump feeding a tank; a one-bus energy operator with a small PV surplus at t=1."""
    wn = wntr.network.WaterNetworkModel()
    wn.add_reservoir("r", base_head=0.)
    wn.add_junction("j", base_demand=.02, elevation=0.)
    wn.add_tank("tank", elevation=25., init_level=2., min_level=0., max_level=5., diameter=10.)
    wn.add_curve("curve", "HEAD", [(.04, 30.)])
    wn.add_pump("pump", "r", "j", pump_type="HEAD", pump_parameter="curve")
    wn.add_pipe("pipe", "j", "tank", length=50., diameter=.3, roughness=130.)
    wn.options.time.duration = T * 3600
    path = tmp_path / "toy.inp"
    wntr.network.write_inpfile(wn, str(path))
    sim = {"flowrate": pd.DataFrame({"pump": [.045] * T, "pipe": [.03] * T}),
           "headloss": pd.DataFrame({"pump": [-27.] * T, "pipe": [.1] * T})}
    energy = {
        "buses": ["bus"], "lines": [],
        "loads": {"bus": [30., 30., 30.]}, "pv_profile": {"bus": [0., 60., 0.]},
        "storage": {"capacity_kwh": 0., "max_charge_kw": 0., "max_discharge_kw": 0.,
                    "charge_efficiency": 1., "discharge_efficiency": 1., "initial_soc_frac": 0.},
        "tariff": [0.1, 0.2, 0.3], "export_tariff": [0.01, 0.01, 0.01],
        "network": {"base_power_kw": 1000., "reference_bus": "bus", "grid_buses": ["bus"],
                    "max_grid_import_kw": 200., "max_grid_export_kw": 0.},
    }
    data = {"water": {"inp_file": str(path), "epanet_sim": sim, "config": {}}, "energy": energy}
    config = {"T": T, "run_nexus": True, "water": {"min_pressure_m": 0.0, "terminal_tank_closure": False},
              "nexus": {"pump_power_kw": {"pump": 20.}, "pump_bus": {"pump": "bus"}},
              "solver": {"name": "highs", "timeout": 60}}
    return data, config


def test_price_table_accepts_scalar_series_and_mapping():
    table = algorithm._pump_price_table(0.2, ["a", "b"], 2)
    assert table == {("a", 0): 0.2, ("a", 1): 0.2, ("b", 0): 0.2, ("b", 1): 0.2}
    table = algorithm._pump_price_table([0.1, 0.3], ["a"], 2)
    assert table == {("a", 0): 0.1, ("a", 1): 0.3}
    table = algorithm._pump_price_table({"a": [0.1, 0.3], "b": 0.5}, ["a", "b"], 2)
    assert table[("b", 1)] == 0.5
    with pytest.raises(ValueError, match="lacks an entry"):
        algorithm._pump_price_table({"a": 0.1}, ["a", "b"], 2)
    with pytest.raises(ValueError, match="finite hourly prices"):
        algorithm._pump_price_table([0.1], ["a"], 2)


def test_contract_accounting_identities():
    bill0, r0, r_star = 40., -110., -90.
    gain = r_star - r0
    table = co.payoff_table(bill0, r0, r_star, [0., .5, 1.])
    assert table["gain"] == pytest.approx(gain)
    base = table["rows"][0]
    assert base["water"] + base["energy"] == pytest.approx(r0)
    for row in table["rows"][1:]:
        assert row["water"] + row["energy"] == pytest.approx(r_star)
        assert row["water_gain"] == pytest.approx(row["share"] * gain)
        assert row["energy_gain"] == pytest.approx((1 - row["share"]) * gain)
    low, high = co.fee_window(bill0, r0, r_star, .5)
    assert high - low == pytest.approx(gain)
    assert low <= co.anchored_fee(bill0, r0, .5) <= high
    water, energy = co.contract_payoffs(r_star, .5, co.anchored_fee(bill0, r0, .5))
    assert water == pytest.approx(-bill0 + .5 * gain)
    assert energy == pytest.approx(r0 + bill0 + .5 * gain)


def test_head_ceiling_bounds_every_source_plus_shutoff(tmp_path):
    data, config = toy_site(tmp_path)
    cfg = copy.deepcopy(config)
    cfg.update(run_energy=False, run_nexus=False)
    cfg["water"]["pump_energy_price"] = [0.1, 0.1, 0.1]
    model = algorithm.build_model({"water": data["water"]}, cfg)
    assert model.head_ceiling_m == pytest.approx(1.05 * (30. + 40.) + 1.0)
    assert model.pump_flow_ceiling_m3s["pump"] == pytest.approx(.08)  # zero-head flow of the curve
    assert model.H["j", 0].ub == pytest.approx(model.head_ceiling_m)


def test_pump_flow_ceiling_respects_pressure_floor(tmp_path):
    data, config = toy_site(tmp_path)
    cfg = copy.deepcopy(config)
    cfg.update(run_energy=False, run_nexus=False)
    cfg["water"]["min_pressure_m"] = 20.0
    model = algorithm.build_model({"water": data["water"]}, cfg)
    wn = wntr.network.WaterNetworkModel(data["water"]["inp_file"])
    from src.helpers.water.hydraulic_utils import pump_flow_at_head
    assert model.pump_flow_ceiling_m3s["pump"] == pytest.approx(pump_flow_at_head(wn.get_link("pump"), 20.0))
    assert model.pump_flow_ceiling_m3s["pump"] < .08


def test_subproblems_and_bundle_recover_integrated_optimum(tmp_path, solver):
    data, config = toy_site(tmp_path)
    data, config, coupling = co.prepare(data, config)
    assert coupling["bus_capacity_kw"]["bus"] == pytest.approx(20.)
    integrated = algorithm.build_model(data, config)
    integrated, record = co.solve(integrated, solver, 60)
    c_star = float(pyo.value(integrated.energy_cost))
    # The water operator's response to the internal price of the restricted LP
    # meters a load the energy operator can serve at the integrated cost.
    lp = co.restricted_lp_prices(integrated, solver)
    rows = co.classify_internal_price(lp["price"], lp["state"], data["energy"]["tariff"],
                                      data["energy"]["export_tariff"], "bus")
    # Proposition 3: inside the band unless a connection limit binds; the zero
    # export rating binds at t=1 (curtailed PV), where the price drops to the
    # marginal value of surplus energy, below the sale price.
    for row in rows:
        if row["connection_binding"]:
            assert row["internal_price"] <= data["energy"]["export_tariff"][row["hour"]] + 1e-6
        else:
            assert row["within_band"]
    assert any(row["connection_binding"] for row in rows)
    result = co.price_coordination(
        data, config, coupling, solver=solver, timeout=60,
        price0=co.bus_price_from_series(data["energy"]["tariff"], coupling, config),
        master="bundle", max_iter=8, tol=1e-6, price_resolution=1e-4, trust_radius=1e-3)
    assert result["best_upper"] == pytest.approx(c_star, abs=1e-6)
    assert result["best_lower"] <= c_star + 1e-6
    assert result["messages"] == 2 * config["T"] * result["iterations"]
    history = result["history"]
    assert all(h["dual_certified"] <= h["dual_value"] + 1e-9 for h in history)
    response = co.energy_response(data, config, result["best"]["load"], coupling, solver=solver, timeout=60)
    assert response["energy_cost"] == pytest.approx(result["best"]["energy_cost"])


def test_flat_tariff_baseline_is_energy_minimal_then_level_holding(tmp_path, solver):
    data, config = toy_site(tmp_path)
    data, config, coupling = co.prepare(data, config)
    base = co.flat_tariff_baseline(data, config, coupling, 0.2, solver=solver, timeout=60)
    assert base["on_hours"] == base["min_on_hours"]
    assert base["bill"] == pytest.approx(0.2 * 20. * base["on_hours"])
    assert base["level_deviation_mh"] >= 0.
