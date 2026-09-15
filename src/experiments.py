"""Reproduce the manuscript's coordination study and nonlinear replay.

Run ``python -m src.experiments`` from the repository root. The study evaluates
the integrated benchmark, the flat-tariff baseline and the tariff benchmarks,
the revenue-sharing contract, the limited-information price coordination
scheme, and the connection-limit sensitivities on the case defined in
``data/inputs/paper_case.yaml``. Numerical artifacts are overwritten only
within the selected study directory and ``paper/results.tex`` /
``paper/results_macros.tex`` / ``paper/figures``.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyomo.environ as pyo

from . import algorithm_tasks as algorithm, coordination as co, postprocessing, preprocessing
from .validation import algebraic_residuals, replay_energy, replay_water
from .workflow import PROJECT_ROOT, load_config

DEFAULT_COORDINATION = {
    "flat_tariff": 0.16,
    "shares": [0.0, 0.5, 1.0],
    "master": "bundle",
    "max_iterations": 30,
    "subgradient_iterations": 15,
    "tolerance": 1e-4,
    "price_resolution": 1e-4,
    "proximal_step": 5e-4,
    "subproblem_rel_gap": 1e-3,
    "subproblem_timeout_s": 180,
    "import_limit_sweep_kw": [150, 125, 100, 90, 80, 70, 60],
    "export_limit_sweep_kw": [50, 25, 0],
}


# ---------------------------------------------------------------------------
# Solves with full artifacts
# ---------------------------------------------------------------------------

def solve_case(name, data, config, output, schedule=None, replay=True, options=None):
    """Solve the coupled model (optionally with fixed pump statuses) and save it."""
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    solver_log = directory / "solver.log"
    solver_log.unlink(missing_ok=True)
    start = time.perf_counter()
    model = algorithm.build_model(data, config)
    if schedule is not None:
        for (pump, t), status in schedule.items():
            model.Status[pump, t].fix(status)
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    solver_options = {**algorithm.solver_default_options(config["solver"]["name"]), **(options or {})}
    model, results, _ = algorithm.solve_model(
        model, config["solver"]["name"], config["solver"]["timeout"], str(solver_log), options=solver_options)
    solve_seconds = time.perf_counter() - start
    if not len(results.solution):
        return model, {"feasible": False, "termination_condition": str(results.solver.termination_condition),
                       "wall_solve_time_s": solve_seconds}
    solution = postprocessing.extract_solution(model)
    summary = postprocessing.create_summary(name, model, solution, results)
    summary.update(feasible=True, build_time_s=build_seconds, wall_solve_time_s=solve_seconds,
                   residuals=algebraic_residuals(model))
    for key, value in summary["residuals"].items():
        if value > 1e-4:
            raise RuntimeError(f"{name}: {key}={value} exceeds numerical feasibility tolerance")
    datasets = {"summary.json": summary, "config.yaml": config}
    for domain in ("water", "energy"):
        for key, rows in solution[domain].items():
            datasets[f"{domain}/{key}.csv"] = rows
    postprocessing.save_data(datasets, directory)
    if replay:
        validation = {
            "water": replay_water(model, data["water"]["inp_file"], config, directory / "validation"),
            "energy": replay_energy(model, data["energy"], directory / "validation"),
        }
        postprocessing.save_data({"validation.json": validation}, directory)
        summary["validation"] = validation
    return model, summary


def schedule_of(model):
    return {(p, t): int(round(pyo.value(model.Status[p, t]))) for p in model.Pumps for t in model.T}


def on_hours(schedule):
    return sorted(t for (_, t), status in schedule.items() if status)


def case_metrics(name, model, summary, load, import_price):
    hours = list(model.T)
    dt_h = float(pyo.value(model.dt)) / 3600.0
    blocks = {}
    for (b, t), kw in load.items():
        blocks[import_price[t]] = blocks.get(import_price[t], 0.0) + kw * dt_h
    return {
        "case": name,
        "energy_cost": float(pyo.value(model.energy_cost)),
        "settlement": -float(pyo.value(model.energy_cost)),
        "grid_import_kwh": float(sum(pyo.value(model.P_import[b, t]) for b in model.Buses for t in hours)) * dt_h,
        "grid_export_kwh": float(sum(pyo.value(model.P_export[b, t]) for b in model.Buses for t in hours)) * dt_h,
        "pv_kwh": float(sum(pyo.value(model.P_pv[b, t]) for b in model.Buses for t in hours)) * dt_h,
        "battery_charge_kwh": float(sum(pyo.value(model.Q_ch[b, t]) for b in model.Buses for t in hours)) * dt_h,
        "peak_grid_import_kw": float(max(sum(pyo.value(model.P_import[b, t]) for b in model.Buses) for t in hours)),
        "pump_on_hours": int(round(sum(pyo.value(model.Status[p, t]) for p in model.Pumps for t in hours))),
        "pump_energy_kwh": float(sum(load.values())) * dt_h,
        "pump_energy_by_price_kwh": {f"{price:.2f}": kwh for price, kwh in sorted(blocks.items())},
        "terminal_tank_head_m": float(pyo.value(model.H_terminal[next(iter(model.Tanks))])),
        "terminal_battery_kwh": float(sum(pyo.value(model.E_soc[b, len(hours)]) for b in model.Buses)),
        "solver_termination": summary["termination_condition"],
        "solve_time_s": summary["wall_solve_time_s"],
        "solver_relative_gap": summary.get("solver_relative_gap"),
        "variables": summary["num_variables"], "constraints": summary["num_constraints"],
        "on_hours": on_hours(schedule_of(model)),
    }


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def price_series(price_by_bus, bus, T):
    return [price_by_bus[(bus, t)] for t in range(T)]


def coordination_summary(name, result, c_star, tol=1e-6):
    history = result["history"]
    reached = next((h["iteration"] for h in history if h["primal_value"] <= c_star + tol), None)
    within_1pct = next((h["iteration"] for h in history
                        if h["gap"] is not None and h["gap"] <= 1e-2), None)
    final = history[-1]
    return {
        "name": name, "master": result["master"], "iterations": result["iterations"],
        "converged": result["converged"], "messages": result["messages"],
        "best_upper": result["best_upper"], "best_lower": result["best_lower"],
        "final_gap": final["gap"], "gap_within_1pct_iteration": within_1pct,
        "primal_reached_optimum_iteration": reached, "best_iteration": result["best"]["iteration"],
        "best_on_hours": on_hours(result["best"]["schedule"]), "elapsed_s": result["elapsed_s"],
        "water_time_total_s": sum(h["water_time_s"] for h in history),
        "energy_time_total_s": sum(h["energy_time_s"] + h["response_time_s"] for h in history),
        "initial_dual": history[0]["dual_certified"], "initial_primal": history[0]["primal_value"],
        "initial_gap": history[0]["gap"],
    }


def serializable_history(history, bus, T):
    rows = []
    for h in history:
        row = {k: v for k, v in h.items() if k not in ("price", "load", "accepted")}
        row["price"] = price_series(h["price"], bus, T)
        row["load"] = price_series(h["load"], bus, T)
        row["accepted"] = price_series(h["accepted"], bus, T)
        rows.append(row)
    return rows


def run_study(config, output, paper_dir, resume=False):
    """Run the study; with ``resume`` reuse saved coordination runs in ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config)
    coord = {**DEFAULT_COORDINATION, **(config.get("coordination") or {})}
    solver, timeout = config["solver"]["name"], config["solver"]["timeout"]
    sub_options = algorithm.solver_gap_options(solver, rel_gap=coord["subproblem_rel_gap"])
    raw = preprocessing.load_networks(config)
    data, config, coupling = co.prepare(raw, config)
    T = config["T"]
    import_price = list(data["energy"]["tariff"])
    export_price = list(data["energy"]["export_tariff"])
    bus = coupling["pump_buses"][0]
    study = {"case": {
        "pump_power_kw": coupling["powers"], "pump_bus": coupling["pump_bus"],
        "import_limit_kw": data["energy"]["network"]["max_grid_import_kw"],
        "export_limit_kw": data["energy"]["network"]["max_grid_export_kw"],
        "flat_tariff": coord["flat_tariff"], "import_price": import_price, "export_price": export_price,
        "water_demand_m3h": None, "feeder_load_kw": None, "pv_available_kw": None,
    }}
    # Profiles for the input figure.
    import wntr
    wn = wntr.network.WaterNetworkModel(data["water"]["inp_file"])
    study["case"]["water_demand_m3h"] = [3600 * sum(
        sum(ts.at(t * 3600) for ts in wn.get_node(j).demand_timeseries_list) * wn.options.hydraulic.demand_multiplier
        for j in wn.junction_name_list) for t in range(T)]
    study["case"]["feeder_load_kw"] = [sum(data["energy"]["loads"][b][t] for b in data["energy"]["buses"]) for t in range(T)]
    study["case"]["pv_available_kw"] = [sum(data["energy"]["pv_profile"][b][t] for b in data["energy"]["buses"]) for t in range(T)]
    study["case"]["tank"] = {k: {"elevation_m": wn.get_node(k).elevation, "diameter_m": wn.get_node(k).diameter,
                                 "init_level_m": wn.get_node(k).init_level, "min_level_m": wn.get_node(k).min_level,
                                 "max_level_m": wn.get_node(k).max_level} for k in wn.tank_name_list}

    # ---- E1: integrated benchmark -------------------------------------------------
    logging.info("E1: integrated benchmark")
    integrated, s_int = solve_case("integrated", data, config, output)
    schedule_star = schedule_of(integrated)
    load_star = co.load_from_schedule(schedule_star, coupling, config)
    c_star = float(pyo.value(integrated.energy_cost))
    lp = co.restricted_lp_prices(integrated, solver)
    internal = co.classify_internal_price(lp["price"], lp["state"], import_price, export_price, bus)
    study["internal_price"] = internal
    postprocessing.save_data({"internal_prices.csv": internal}, output)

    # ---- E1: flat-tariff baseline (level holding) ---------------------------------
    logging.info("E1: flat-tariff baseline")
    base = co.flat_tariff_baseline(data, config, coupling, coord["flat_tariff"], solver=solver, timeout=timeout)
    baseline, s_base = solve_case("baseline_flat", data, config, output, schedule=base["schedule"])
    c_base = float(pyo.value(baseline.energy_cost))
    bill0 = base["bill"]

    # ---- E1: tariff benchmarks ----------------------------------------------------
    logging.info("E1: pass-through and internal-price tariffs")
    passthrough = co.water_operator(data, config, co.bus_price_from_series(import_price, coupling, config),
                                    coupling, solver=solver, timeout=timeout)
    pt_model, s_pt = solve_case("tariff_passthrough", data, config, output, schedule=passthrough["schedule"], replay=False)
    internal_resp = co.water_operator(data, config, {k: v for k, v in lp["price"].items() if k[0] == bus},
                                      coupling, solver=solver, timeout=timeout)
    ip_model, s_ip = solve_case("tariff_internal", data, config, output, schedule=internal_resp["schedule"], replay=False)
    metrics = [
        case_metrics("baseline_flat", baseline, s_base, base["load"], import_price),
        case_metrics("tariff_passthrough", pt_model, s_pt, passthrough["load"], import_price),
        case_metrics("tariff_internal", ip_model, s_ip, internal_resp["load"], import_price),
        case_metrics("integrated", integrated, s_int, load_star, import_price),
    ]
    for row in metrics:
        row["gain"] = c_base - row["energy_cost"]
        row["captured_fraction"] = row["gain"] / (c_base - c_star) if c_base > c_star else None
    study["metrics"] = metrics
    study["baseline"] = {k: v for k, v in base.items() if k not in ("model", "schedule", "load")}
    study["baseline"]["on_hours"] = on_hours(base["schedule"])
    study["integrated"] = {"energy_cost": c_star, "on_hours": on_hours(schedule_star),
                           "solver_relative_gap": s_int.get("solver_relative_gap"),
                           "solver_lower_bound": s_int.get("solver_lower_bound"),
                           "solve_time_s": s_int["wall_solve_time_s"],
                           "variables": s_int["num_variables"], "constraints": s_int["num_constraints"],
                           "restricted_lp_objective": lp["objective"]}
    study["replay"] = {"baseline_flat": s_base["validation"], "integrated": s_int["validation"]}

    # ---- E2: the contract ---------------------------------------------------------
    logging.info("E2: contract accounting")
    r_base, r_star = -c_base, -c_star
    gain = c_base - c_star
    table = co.payoff_table(bill0, r_base, r_star, coord["shares"])
    window = {f"{share:.2f}": co.fee_window(bill0, r_base, r_star, share) for share in coord["shares"]}
    deviations = []
    for row in metrics:
        for share in coord["shares"]:
            fee = co.anchored_fee(bill0, r_base, share)
            water_dev, _ = co.contract_payoffs(row["settlement"], share, fee)
            water_star, _ = co.contract_payoffs(r_star, share, fee)
            deviations.append({"schedule": row["case"], "share": share,
                               "water_payoff": water_dev, "deviation_gain": water_dev - water_star})
    study["contract"] = {"gain": gain, "gain_fraction_of_site_cost": gain / c_base,
                         "gain_fraction_of_pumping_bill": gain / bill0, "bill0": bill0,
                         "settlement_baseline": r_base, "settlement_star": r_star,
                         "payoffs": table["rows"], "fee_window": window, "deviations": deviations,
                         "nash_share": 0.5}

    # ---- E4: distributed computation ----------------------------------------------
    logging.info("E4: price coordination")
    runs = {}
    marginal = co.marginal_cost_prices(data, config, base["load"], coupling, solver=solver, timeout=timeout)
    market = co.bus_price_from_series(import_price, coupling, config)
    study["coordination_starts"] = {"marginal_cost": price_series(marginal, bus, T),
                                    "market_price": price_series(market, bus, T)}
    histories = {}
    for name, start, master, iterations in (
            ("bundle_marginal", marginal, coord["master"], coord["max_iterations"]),
            ("bundle_market", market, coord["master"], coord["max_iterations"]),
            ("subgradient_marginal", marginal, "subgradient", coord["subgradient_iterations"])):
        saved = output / f"coordination_{name}.json"
        if resume and saved.is_file():
            logging.info("%s: reusing saved run %s", name, saved)
            payload = json.loads(saved.read_text(encoding="utf-8"))
            summary = payload["summary"]
            summary["primal_reached_optimum_iteration"] = next(
                (h["iteration"] for h in payload["history"] if h["primal_value"] <= c_star + 1e-6), None)
            study.setdefault("coordination", {})[name] = summary
            histories[name] = {"history": payload["history"]}
            continue
        result = co.price_coordination(
            data, config, coupling, solver=solver, timeout=timeout, price0=start, master=master,
            max_iter=iterations, tol=coord["tolerance"], price_resolution=coord["price_resolution"],
            trust_radius=coord["proximal_step"], water_options=sub_options,
            subproblem_timeout=coord["subproblem_timeout_s"], primal_target=None)
        runs[name] = result
        summary = coordination_summary(name, result, c_star)
        study.setdefault("coordination", {})[name] = summary
        histories[name] = {"history": serializable_history(result["history"], bus, T)}
        postprocessing.save_data({f"coordination_{name}.json": {
            "summary": summary, "history": histories[name]["history"],
            "center_price": price_series(result["center_price"], bus, T) if result.get("center_price") else None,
        }}, output)
        logging.info("%s: %s", name, json.dumps(summary))

    # ---- E6: dual function at the restricted-LP prices ----------------------------
    logging.info("E6: dual function at the restricted-LP prices")
    study["duality"] = dual_gap_at_internal_prices(data, config, coupling, lp, bus, c_star, solver, timeout)

    # ---- E5: connection-limit sensitivities ---------------------------------------
    logging.info("E5: connection-limit sweeps")
    sweeps = {}
    for key, values in (("max_grid_import_kw", coord["import_limit_sweep_kw"]),
                        ("max_grid_export_kw", coord["export_limit_sweep_kw"])):
        rows = []
        for value in values:
            sdata = copy.deepcopy(data)
            sdata["energy"]["network"][key] = float(value)
            sconfig = copy.deepcopy(config)
            sconfig["energy"][key] = float(value)
            name = f"sweep_{key.replace('max_grid_', '').replace('_kw', '')}_{value:g}"
            model, summary = solve_case(name, sdata, sconfig, output, replay=False)
            row = {"limit_kw": float(value), "integrated_feasible": summary.get("feasible", False),
                   "integrated_termination": summary.get("termination_condition")}
            if row["integrated_feasible"]:
                row["energy_cost"] = float(pyo.value(model.energy_cost))
                row["on_hours"] = on_hours(schedule_of(model))
                row["solver_relative_gap"] = summary.get("solver_relative_gap")
                prices = co.restricted_lp_prices(model, solver)
                rows_p = co.classify_internal_price(prices["price"], prices["state"], import_price, export_price, bus)
                row["hours_off_reference"] = [r["hour"] for r in rows_p if r["mode"] != "idle" and not r["equals_reference"]]
                row["hours_outside_band"] = [r["hour"] for r in rows_p if not r["within_band"]]
                row["internal_price"] = [r["internal_price"] for r in rows_p]
                row["binding_hours"] = [r["hour"] for r in rows_p if r["connection_binding"]]
            response = co.energy_response(sdata, sconfig, base["load"], coupling, solver=solver, timeout=timeout)
            row["baseline_feasible"] = response["feasible"]
            if response["feasible"]:
                row["baseline_energy_cost"] = response["energy_cost"]
                if row["integrated_feasible"]:
                    row["gain"] = response["energy_cost"] - row["energy_cost"]
            pt_response = co.energy_response(sdata, sconfig, passthrough["load"], coupling, solver=solver, timeout=timeout)
            row["passthrough_feasible"] = pt_response["feasible"]
            if pt_response["feasible"] and row["integrated_feasible"]:
                row["passthrough_energy_cost"] = pt_response["energy_cost"]
            rows.append(row)
        sweeps[key] = rows
    study["sweeps"] = sweeps

    # ---- provenance and outputs -----------------------------------------------------
    source_files = [*sorted((PROJECT_ROOT / "src").rglob("*.py")),
                    Path(data["water"]["inp_file"]), Path(data["energy"]["dss_file"])]
    provenance = {"python": sys.version, "platform": platform.platform(),
                  "solver": solver, "solver_version": str(pyo.SolverFactory(solver).version()),
                  "versions": {name: importlib.metadata.version(name) for name in
                               ("pyomo", "highspy", "wntr", "OpenDSSDirect.py", "numpy", "scipy", "pandas", "matplotlib")},
                  "sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in source_files if path.is_relative_to(PROJECT_ROOT)}}
    study["provenance"] = provenance
    study["coordination_settings"] = coord
    postprocessing.save_data({"study.json": study, "comparison.csv": metrics, "config.yaml": config}, output)
    schedules = {name: schedule_series(model) for name, model in (("baseline_flat", baseline), ("integrated", integrated))}
    make_figures(study, histories, schedules, config, paper_dir)
    write_results(study, config, paper_dir)
    return study


def dual_gap_at_internal_prices(data, config, coupling, lp, bus, c_star, solver, timeout):
    """Evaluate D at the restricted-LP prices and report the duality gap of the instance."""
    price = {k: v for k, v in lp["price"].items() if k[0] == bus}
    result = co.dual_at_prices(data, config, price, coupling, solver=solver, timeout=timeout,
                               water_options=algorithm.solver_gap_options(solver, rel_gap=1e-4))
    result["integrated_cost"] = c_star
    result["gap_certified"] = co.dual_bound_gap(result["dual_certified"], c_star)
    result["gap_value"] = co.dual_bound_gap(result["dual_value"], c_star)
    return result


def schedule_series(model):
    """Hourly series of a solved coupled model for the schedule figure."""
    hours = list(model.T)
    tank = next(iter(model.Tanks))
    return {
        "pump_kw": [sum(pyo.value(model.PumpBusLoad[b, t]) for b in model.Buses) for t in hours],
        "tank_head_m": [pyo.value(model.H[tank, t]) for t in hours] + [pyo.value(model.H_terminal[tank])],
        "net_import_kw": [sum(pyo.value(model.P_import[b, t] - model.P_export[b, t]) for b in model.Buses) for t in hours],
        "tank": tank,
    }


def load_saved_study(output):
    """Reload a study's saved artifacts for re-typesetting without re-solving."""
    import pandas as pd
    output = Path(output)
    study = json.loads((output / "study.json").read_text(encoding="utf-8"))
    histories = {}
    for path in sorted(output.glob("coordination_*.json")):
        name = path.stem.replace("coordination_", "")
        histories[name] = json.loads(path.read_text(encoding="utf-8"))
    schedules = {}
    for name in ("baseline_flat", "integrated"):
        case = output / name
        status = pd.read_csv(case / "water" / "pump_status.csv")
        heads = pd.read_csv(case / "water" / "heads.csv")
        dispatch = pd.read_csv(case / "energy" / "dispatch.csv")
        tank = next(iter(study["case"]["tank"]))
        T = int(dispatch["time"].max()) + 1
        pump = status.groupby("time")["power_kw"].sum().reindex(range(T), fill_value=0.0).tolist()
        head = heads[heads["node"] == tank].sort_values("time")["head"].tolist()
        net = dispatch.groupby("time").apply(lambda g: float((g["P_import"] - g["P_export"]).sum())).reindex(range(T)).tolist()
        schedules[name] = {"pump_kw": pump, "tank_head_m": head, "net_import_kw": net, "tank": tank}
    return study, histories, schedules


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.1,
    })


def make_figures(study, histories, schedules, config, paper_dir):
    """Figures from saved values: ``histories`` are the serialized coordination
    runs (price/load lists per iteration) and ``schedules`` the hourly series of
    the baseline and integrated cases."""
    _style()
    figures = paper_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    T = config["T"]
    edges = np.arange(T + 1)
    case = study["case"]
    grey, blue, orange = "#52514e", "#2a78d6", "#eb6834"

    # ---- inputs -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.6), sharex=True)
    axes[0, 0].stairs(case["import_price"], edges, color=grey, label="import $c^{\\mathrm{buy}}_t$")
    axes[0, 0].stairs(case["export_price"], edges, color=grey, linestyle="--", label="export $c^{\\mathrm{sell}}_t$")
    axes[0, 0].set_ylabel("price (unit/kWh)")
    axes[0, 0].legend(frameon=False, loc="upper left")
    axes[0, 1].stairs(case["water_demand_m3h"], edges, color=grey)
    axes[0, 1].set_ylabel("water demand (m$^3$/h)")
    axes[1, 0].stairs(case["feeder_load_kw"], edges, color=grey)
    axes[1, 0].set_ylabel("feeder load (kW)")
    axes[1, 1].stairs(case["pv_available_kw"], edges, color=grey)
    axes[1, 1].set_ylabel("PV available (kW)")
    for ax in axes.flat:
        ax.set_xlim(0, T)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.tick_params(length=2.5)
    for ax in axes[1]:
        ax.set_xlabel("hour")
        ax.set_xticks(np.arange(0, T + 1, 6))
    fig.tight_layout(pad=0.4)
    fig.savefig(figures / "inputs.pdf")
    plt.close(fig)

    # ---- schedules ----------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(3.5, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [0.8, 0.8, 1, 1]})
    internal = [row["internal_price"] for row in study["internal_price"]]
    axes[0].stairs(case["import_price"], edges, color=grey, label="$c^{\\mathrm{buy}}_t$")
    axes[0].stairs(case["export_price"], edges, color=grey, linestyle="--", label="$c^{\\mathrm{sell}}_t$")
    axes[0].stairs(internal, edges, color=blue, linewidth=1.5, label="$\\pi^\\star_t$")
    axes[0].set_ylabel("price\n(unit/kWh)")
    axes[0].legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.9, handlelength=1.6)
    styles = {"baseline_flat": ("Baseline (flat tariff)", orange, (0, (3.2, 1.6))),
              "integrated": ("Coordinated", blue, "solid")}
    tank_name = schedules["integrated"]["tank"]
    elevation = case["tank"][tank_name]["elevation_m"]
    initial = None
    for name, series in schedules.items():
        label, color, style = styles[name]
        pump = series["pump_kw"]
        levels = [h - elevation for h in series["tank_head_m"]]
        initial = levels[0]
        net = series["net_import_kw"]
        axes[1].stairs(pump, edges, color=color, linestyle=style, linewidth=1.3, label=label, baseline=None)
        axes[2].plot(edges, levels, color=color, linestyle=style, linewidth=1.3, marker="o", markersize=2.2,
                     markerfacecolor="white", markeredgewidth=0.8)
        axes[3].stairs(net, edges, color=color, linestyle=style, linewidth=1.3, baseline=None)
    axes[1].set_ylabel("pump load\n(kW)")
    axes[2].set_ylabel("tank level\n(m)")
    if initial is not None:
        axes[2].axhline(initial, color="0.55", linestyle=":", linewidth=0.8)
    axes[3].axhline(0, color="0.7", linewidth=0.6)
    axes[3].set_ylabel("net import\n(kW)")
    axes[3].set_xlabel("hour of day")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=2,
               frameon=False, handlelength=2.4, columnspacing=1.4)
    for ax in axes:
        ax.set_xlim(0, T)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.tick_params(length=2.5)
    axes[3].set_xticks(np.arange(0, T + 1, 4))
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.2, right=0.985, top=0.94, bottom=0.075, hspace=0.3)
    fig.savefig(figures / "coordination_results.pdf")
    fig.savefig(figures / "coordination_results.png", dpi=220)
    plt.close(fig)

    # ---- convergence --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    palette = {"bundle_marginal": (blue, "solid", "bundle, marginal-cost start"),
               "bundle_market": (blue, (0, (3.2, 1.6)), "bundle, market-price start"),
               "subgradient_marginal": (orange, "solid", "subgradient, marginal-cost start")}
    for name, result in histories.items():
        color, style, label = palette[name]
        k = [h["iteration"] + 1 for h in result["history"]]
        gaps = [max(100 * h["gap"], 1e-3) if h["gap"] is not None else float("nan") for h in result["history"]]
        axes[0].plot(k, gaps, color=color, linestyle=style, label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration $k$")
    axes[0].set_ylabel("certified gap (%)")
    axes[0].legend(frameon=False, loc="upper right")
    market = histories.get("bundle_market")
    if market is not None:
        show = sorted(set([h for h in (6, 12, 16) if h < T]))
        markers = ["solid", (0, (3.2, 1.6)), (0, (1, 1))]
        for hour, style in zip(show, markers):
            k = [h["iteration"] + 1 for h in market["history"]]
            axes[1].plot(k, [h["price"][hour] for h in market["history"]], color=blue, linestyle=style,
                         label=f"$\\pi^{{(k)}}_{{{hour}}}$")
            axes[1].axhline(internal[hour], color=grey, linestyle=style, linewidth=0.6)
        axes[1].set_xlabel("iteration $k$ (bundle, market-price start)")
        axes[1].set_ylabel("internal price (unit/kWh)")
        axes[1].legend(frameon=False, loc="center right", ncol=3, columnspacing=0.8)
    for ax in axes:
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.tick_params(length=2.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(figures / "convergence.pdf")
    plt.close(fig)

    # ---- sensitivities ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))
    for ax, (key, label) in zip(axes, (("max_grid_import_kw", "import limit $\\overline{p}^{\\mathrm{g}}$ (kW)"),
                                       ("max_grid_export_kw", "export limit $\\overline{p}^{\\mathrm{x}}$ (kW)"))):
        rows = study["sweeps"][key]
        xs = [r["limit_kw"] for r in rows if r.get("gain") is not None]
        ys = [r["gain"] for r in rows if r.get("gain") is not None]
        ax.plot(xs, ys, color=blue, marker="o", markersize=3)
        limits = sorted(r["limit_kw"] for r in rows)
        half = 0.5 * min((b - a for a, b in zip(limits, limits[1:])), default=5.0)
        only = [r["limit_kw"] for r in rows if r["integrated_feasible"] and not r["baseline_feasible"]]
        none = [r["limit_kw"] for r in rows if not r["integrated_feasible"]
                and str(r.get("integrated_termination", "")).startswith("infeasible")]
        if only:
            ax.axvspan(min(only) - half, max(only) + half, color=blue, alpha=0.12, lw=0, label="only coordination feasible")
        if none:
            ax.axvspan(min(none) - half, max(none) + half, color="0.75", alpha=0.5, lw=0, label="no schedule feasible")
        ax.set_xlim(max(0.0, min(limits) - half), max(limits) + half)
        ax.set_xlabel(label)
        ax.set_ylabel("gain $\\mathcal{G}$ (unit/day)")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.tick_params(length=2.5)
        if only or none:
            ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.4)
    fig.savefig(figures / "sensitivity.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Results section and macros
# ---------------------------------------------------------------------------

def _hours(values):
    values = [str(v) for v in values]
    if not values:
        return "none"
    if len(values) == 1:
        return f"hour {values[0]}"
    if len(values) == 2:
        return f"hours {values[0]} and {values[1]}"
    return "hours " + ", ".join(values[:-1]) + f", and {values[-1]}"


def _num(value, digits=2):
    return f"{value:,.{digits}f}".replace(",", "\\,")


def _pct(value, digits=1):
    return f"{100 * value:.{digits}f}"


def write_results(study, config, paper_dir):
    """Write paper/results.tex and paper/results_macros.tex from the saved study."""
    case, met = study["case"], {m["case"]: m for m in study["metrics"]}
    con, coord, rep = study["contract"], study["coordination"], study["replay"]
    integ = study["integrated"]
    T = config["T"]
    tau = case["flat_tariff"]
    pump_kw = sum(case["pump_power_kw"].values())
    prices = sorted(set(case["import_price"]))
    export_price = case["export_price"][0]
    c_base, c_star = met["baseline_flat"]["energy_cost"], met["integrated"]["energy_cost"]
    gain = con["gain"]
    internal = study["internal_price"]
    import_hours = [r["hour"] for r in internal if r["mode"] == "import"]
    export_hours = [r["hour"] for r in internal if r["mode"] == "export"]
    idle_hours = [r["hour"] for r in internal if r["mode"] == "idle"]
    binding = [r["hour"] for r in internal if r["connection_binding"]]
    warm, cold, subg = coord["bundle_marginal"], coord["bundle_market"], coord["subgradient_marginal"]
    sweep_i, sweep_x = study["sweeps"]["max_grid_import_kw"], study["sweeps"]["max_grid_export_kw"]
    nash = con["payoffs"][[r["share"] for r in con["payoffs"]].index(0.5)]
    window = con["fee_window"]["0.50"]
    solver = study["provenance"]["solver"]
    solver_name = {"highs": "HiGHS", "gurobi": "Gurobi", "cbc": "CBC", "glpk": "GLPK", "cplex": "CPLEX"}.get(solver, solver)
    solver_version = study["provenance"]["solver_version"]
    if isinstance(solver_version, str) and solver_version.startswith("("):
        solver_version = ".".join(part.strip() for part in solver_version.strip("()").split(","))

    def block_kwh(name):
        blocks = met[name]["pump_energy_by_price_kwh"]
        return ", ".join(f"{float(kwh) / pump_kw:.0f}~h at {price}" for price, kwh in blocks.items())

    macros = {
        "resSolver": solver_name, "resSolverVersion": solver_version,
        "resPumpPower": f"{pump_kw:.2f}", "resFlatTariff": f"{tau:.2f}",
        "resImportLimit": f"{case['import_limit_kw']:g}", "resExportLimit": f"{case['export_limit_kw']:g}",
        "resExportPrice": f"{export_price:.2f}",
        "resCostBase": _num(c_base), "resCostStar": _num(c_star), "resGain": _num(gain),
        "resGainPctSite": _pct(con["gain_fraction_of_site_cost"]),
        "resGainPctBill": _pct(con["gain_fraction_of_pumping_bill"], 0),
        "resBillZero": _num(con["bill0"]), "resOnHours": str(met["integrated"]["pump_on_hours"]),
        "resOnHoursBase": str(met["baseline_flat"]["pump_on_hours"]),
        "resCostPass": _num(met["tariff_passthrough"]["energy_cost"]),
        "resCapturedPass": _pct(met["tariff_passthrough"]["captured_fraction"]),
        "resCostInternal": _num(met["tariff_internal"]["energy_cost"]),
        "resCapturedInternal": _pct(met["tariff_internal"]["captured_fraction"]),
        "resNashWater": _num(nash["water_gain"]), "resNashFee": _num(nash["fee"]),
        "resFeeLow": _num(window[0]), "resFeeHigh": _num(window[1]),
        "resWarmIterations": str(warm["iterations"]), "resWarmMessages": str(warm["messages"]),
        "resWarmInitialGapPct": _pct(warm["initial_gap"], 2) if warm.get("initial_gap") is not None else "--",
        "resDualityGapPct": f"{100 * (study.get('duality', {}).get('gap_certified') or 0.0):.3f}",
        "resDualAtStar": _num(study["duality"]["dual_certified"]) if study.get("duality") else "--",
        "resWarmGapPct": _pct(warm["final_gap"], 2) if warm["final_gap"] is not None else "--",
        "resWarmReached": str(warm["primal_reached_optimum_iteration"] + 1) if warm["primal_reached_optimum_iteration"] is not None else "--",
        "resColdIterations": str(cold["iterations"]), "resColdMessages": str(cold["messages"]),
        "resColdGapPct": _pct(cold["final_gap"], 2) if cold["final_gap"] is not None else "--",
        "resColdReached": str(cold["primal_reached_optimum_iteration"] + 1) if cold["primal_reached_optimum_iteration"] is not None else "--",
        "resSubIterations": str(subg["iterations"]),
        "resSubGapPct": _pct(subg["final_gap"], 2) if subg["final_gap"] is not None else "--",
        "resSubReached": str(subg["primal_reached_optimum_iteration"] + 1) if subg["primal_reached_optimum_iteration"] is not None else "--",
        "resDualBoundWarm": _num(warm["best_lower"]), "resDualBoundCold": _num(cold["best_lower"]),
        "resIntegratedGap": f"{100 * (integ['solver_relative_gap'] or 0):.3f}",
        "resIntegratedTime": f"{integ['solve_time_s']:.0f}",
        "resVars": str(integ["variables"]), "resCons": str(integ["constraints"]),
        "resExportHours": _hours(export_hours), "resIdleHours": _hours(idle_hours),
        "resBindingHours": _hours(binding),
        "resTerminalTankBase": f"{next(iter(rep['baseline_flat']['water']['terminal_tank_change_m'].values())):+.3f}",
        "resTerminalTankStar": f"{next(iter(rep['integrated']['water']['terminal_tank_change_m'].values())):+.3f}",
    }
    only_coord = [r["limit_kw"] for r in sweep_i if r["integrated_feasible"] and not r["baseline_feasible"]]
    infeasible = [r["limit_kw"] for r in sweep_i if not r["integrated_feasible"]
                  and str(r.get("integrated_termination", "")).startswith("infeasible")]
    unsolved = [r["limit_kw"] for r in sweep_i if not r["integrated_feasible"]
                and not str(r.get("integrated_termination", "")).startswith("infeasible")]

    def trend(rows):
        both = [r for r in rows if r.get("gain") is not None]
        if len(both) < 2:
            return None
        first, last = both[0], both[-1]
        if abs(last["gain"] - first["gain"]) < 0.005:
            return f"leaves the gain unchanged at {first['gain']:.2f}/day"
        verb = "raises" if last["gain"] > first["gain"] else "lowers"
        return (f"{verb} the gain from {first['gain']:.2f}/day at {first['limit_kw']:g}~kW to "
                f"{last['gain']:.2f}/day at {last['limit_kw']:g}~kW")
    import_sentence = "Tightening the import rating " + (trend(sweep_i) or "changes the gain as listed")
    off_hours = [r for r in sweep_i if r["integrated_feasible"] and r.get("hours_off_reference")]
    if off_hours:
        import_sentence += (f"; at {off_hours[0]['limit_kw']:g}~kW and below the internal price departs from the market "
                            f"price in {_hours(off_hours[0]['hours_off_reference'])}, where the rating binds")
    import_sentence += "."
    if only_coord:
        import_sentence += (f" At {', '.join(f'{v:g}' for v in only_coord)}~kW the baseline schedule can no longer be served "
                            "while coordinated operation remains feasible: there, coordination is not merely profitable but "
                            "what keeps the site operating.")
    if infeasible:
        import_sentence += f" No schedule is feasible at {', '.join(f'{v:g}' for v in infeasible)}~kW."
    if unsolved:
        import_sentence += (f" At {', '.join(f'{v:g}' for v in unsolved)}~kW the solver found no incumbent within its "
                            "time limit, so the integrated value is not reported.")
    export_sentence = ("Tightening the export rating " + (trend(sweep_x) or "changes the gain as listed") +
                       ": with less room to export, both schedules absorb more of the midday surplus on site, and the internal "
                       "price in the curtailment hours falls to the marginal value of spilled energy, zero, for the coordinated "
                       "schedule and the baseline alike.")
    macros["resImportOnlyCoordination"] = ", ".join(f"{v:g}" for v in only_coord) if only_coord else "none"
    macros["resImportInfeasible"] = ", ".join(f"{v:g}" for v in infeasible) if infeasible else "none"
    macros["resExportZeroGain"] = _num(next((r["gain"] for r in sweep_x if r["limit_kw"] == 0 and r.get("gain") is not None), float("nan")))
    macro_text = "% Generated by python -m src.experiments; do not edit.\n" + "".join(
        f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items())
    (paper_dir / "results_macros.tex").write_text(macro_text, encoding="utf-8")

    # ---- tables -------------------------------------------------------------------
    names = {"baseline_flat": "Baseline: flat tariff, level holding", "tariff_passthrough": "Pass-through tariff $\\tau_t=c^{\\mathrm{buy}}_t$",
             "tariff_internal": "Internal-price tariff $\\tau_t=\\pi^\\star_t$", "integrated": "Integrated optimum (contract)"}
    arrangement_rows = "\n".join(
        f"{names[m['case']]} & {m['energy_cost']:.2f} & {m['gain']:.2f} & "
        f"{(100 * m['captured_fraction']) if m['captured_fraction'] is not None else 0:.1f} & {m['pump_on_hours']} & "
        f"{m['grid_import_kwh']:.0f} & {m['grid_export_kwh']:.0f} \\\\" for m in study["metrics"])
    payoff_rows = []
    for r in con["payoffs"]:
        if r["arrangement"] == "baseline":
            payoff_rows.append(f"Baseline (flat tariff) & -- & -- & {r['water']:.2f} & {r['energy']:.2f} & {r['total']:.2f} & -- & -- \\\\")
        else:
            payoff_rows.append(f"Contract & {r['share']:.2f} & {r['fee']:.2f} & {r['water']:.2f} & {r['energy']:.2f} & "
                               f"{r['total']:.2f} & {r['water_gain']:+.2f} & {r['energy_gain']:+.2f} \\\\")
    dev_rows = "\n".join(
        f"{names[d['schedule']]} & {d['water_payoff']:.2f} & {d['deviation_gain']:+.2f} \\\\"
        for d in con["deviations"] if d["share"] == 0.5)
    coord_rows = "\n".join(
        f"{label} & {c['iterations']} & {c['messages']} & {c['best_lower']:.2f} & {c['best_upper']:.2f} & "
        f"{100 * c['final_gap']:.2f} & {c['primal_reached_optimum_iteration'] + 1 if c['primal_reached_optimum_iteration'] is not None else '--'} & "
        f"{c['elapsed_s'] / 60:.1f} \\\\"
        for label, c in (("Bundle, marginal-cost start", warm),
                         ("Bundle, market-price start", cold),
                         ("Subgradient, marginal-cost start", subg)))
    sweep_rows = []
    for key, label in (("max_grid_import_kw", "Import"), ("max_grid_export_kw", "Export")):
        for r in study["sweeps"][key]:
            cost = (f"{r['energy_cost']:.2f}" if r["integrated_feasible"] else
                    ("infeasible" if str(r.get("integrated_termination", "")).startswith("infeasible") else "not solved"))
            bcost = f"{r['baseline_energy_cost']:.2f}" if r["baseline_feasible"] else "infeasible"
            gain_txt = f"{r['gain']:.2f}" if r.get("gain") is not None else "--"
            off = _hours(r.get("hours_off_reference", [])) if r["integrated_feasible"] else "--"
            sweep_rows.append(f"{label} & {r['limit_kw']:g} & {bcost} & {cost} & {gain_txt} & {off} \\\\")
    w = {n: rep[n]["water"] for n in ("baseline_flat", "integrated")}
    e = {n: rep[n]["energy"] for n in ("baseline_flat", "integrated")}
    term = {n: next(iter(w[n]["terminal_tank_change_m"].values())) for n in w}

    def pair(fmt, key, source):
        return " & ".join(fmt.format(source[n][key]) for n in ("baseline_flat", "integrated"))
    replay_rows = "\n".join([
        f"Minimum junction pressure (m) & {pair('{:.2f}', 'min_junction_pressure_m', w)} \\\\",
        f"Pressure violations at report times & {pair('{}', 'pressure_violations_at_report_times', w)} \\\\",
        f"Pump-status mismatches at report times & {pair('{}', 'pump_status_mismatches_at_report_times', w)} \\\\",
        f"Maximum tank-head discrepancy (m) & {pair('{:.3f}', 'max_tank_head_error_m', w)} \\\\",
        "Maximum flow discrepancy (L/s) & " + " & ".join(f"{1000 * w[n]['max_flow_error_m3s']:.3f}" for n in w) + " \\\\",
        "Terminal tank-head change vs.\\ initial (m) & " + " & ".join(f"${term[n]:+.3f}$" for n in w) + " \\\\",
        "All OpenDSS snapshots converged & " + " & ".join("yes" if e[n]["all_snapshots_converged"] else "no" for n in e) + " \\\\",
        f"Maximum voltage discrepancy (p.u.) & {pair('{:.5f}', 'max_voltage_error_pu', e)} \\\\",
        f"Voltage violations & {pair('{}', 'voltage_violations', e)} \\\\",
        f"Peak line loading (\\%) & {pair('{:.2f}', 'max_line_loading_percent', e)} \\\\",
        f"Maximum grid-exchange discrepancy (kW) & {pair('{:.3f}', 'max_grid_import_error_kw', e)} \\\\",
    ])
    price_blocks = "; ".join(f"{p:.2f} in " + _hours([t for t in range(T) if case["import_price"][t] == p])
                             for p in prices)
    tank = next(iter(case["tank"].values()))
    short = [n for n in term if term[n] < -1e-4]
    reserve_sentence = ("Both replays end with the tank at or above its initial level." if not short else
                        "The replay ends with the tank " + " and ".join(
                            f"{-term[n]:.3f}~m ({'baseline' if n == 'baseline_flat' else 'coordinated'})" for n in short)
                        + " below its initial level despite the replenishment constraint \\eqref{eq:w_tank_terminal}, "
                          "so the hourly hydraulic approximation does not preserve the terminal requirement exactly.")
    def hours_where(rows, predicate):
        return [r["hour"] for r in rows if predicate(r)]
    imp_ne = hours_where(internal, lambda r: r["mode"] == "import" and not r["equals_reference"])
    exp_ne = hours_where(internal, lambda r: r["mode"] == "export" and not r["equals_reference"])
    idle_out = hours_where(internal, lambda r: r["mode"] == "idle" and not r["within_band"])
    parts = []
    if import_hours:
        parts.append(f"in the {len(import_hours)} hours in which the site imports it equals the purchase price"
                     + (f" except in {_hours(imp_ne)}, where the import rating binds" if imp_ne else ""))
    if export_hours:
        parts.append(f"in the {len(export_hours)} export hours ({_hours(export_hours)}) it equals the sale price {export_price:.2f}"
                     + (f" except in {_hours(exp_ne)}" if exp_ne else ""))
    if idle_hours:
        parts.append(f"in the {len(idle_hours)} idle hours ({_hours(idle_hours)}) it lies "
                     + ("inside" if not idle_out else "inside, or below when PV is curtailed,")
                     + " the band $[c^{\\mathrm{sell}}_t, c^{\\mathrm{buy}}_t]$, set by the battery's intertemporal arbitrage")
    internal_sentence = "The internal price $\\pi^\\star_t$ of the restricted LP follows Proposition~\\ref{prop:price}: " + "; ".join(parts) + "."
    if binding and not (imp_ne or exp_ne):
        internal_sentence += (f" A rating binds in {_hours(binding)} without moving the price off the market price, because "
                              "the battery absorbs the marginal kilowatt at the same value.")

    peak_price = max(prices)
    surplus_hours = set(export_hours) | set(idle_hours)
    star_hours, base_hours = met["integrated"]["on_hours"], met["baseline_flat"]["on_hours"]
    star_mid = [t for t in star_hours if t in surplus_hours]
    star_peak = [t for t in star_hours if case["import_price"][t] == peak_price]
    base_peak = [t for t in base_hours if case["import_price"][t] == peak_price]
    base_mid = [t for t in base_hours if t in surplus_hours]
    how_sentence = ("Figure~\\ref{fig:computed} shows how the gain is earned: the coordinated schedule places "
                    f"{len(star_mid)} of its {len(star_hours)} pump-hours in the export and idle hours, where the site's own "
                    "energy is cheapest, "
                    + ("and none" if not star_peak else f"and {len(star_peak)}")
                    + f" in the peak block at {peak_price:.2f}; the baseline, holding its level, places {len(base_mid)} in the "
                    f"surplus hours and {len(base_peak)} in the peak block, which the grid serves at the peak price.")
    pass_peak = [t for t in met["tariff_passthrough"]["on_hours"] if case["import_price"][t] == peak_price]
    pass_mid = [t for t in met["tariff_passthrough"]["on_hours"] if t in surplus_hours]
    pass_frac = met["tariff_passthrough"]["captured_fraction"] or 0.0
    int_frac = met["tariff_internal"]["captured_fraction"] or 0.0
    tariff_sentence = (f"Passing the purchase price through to the water operator captures {100 * pass_frac:.1f}\\% of the gain: "
                       + ("the water operator then avoids the peak block " if not pass_peak else
                          f"the water operator still pumps {len(pass_peak)} hour(s) in the peak block ")
                       + f"but places only {len(pass_mid)} pump-hours in the surplus hours, having no reason to prefer them over "
                       "the other hours of the same purchase price. Charging the internal price itself captures "
                       f"{100 * int_frac:.1f}\\%"
                       + (": the water operator is indifferent among equally priced hours, and the tie it happened to break is "
                          "not the one the energy operator can serve most cheaply (Proposition~\\ref{prop:tariff})."
                          if int_frac < 0.9995 else
                          ", the whole gain: the tie among equally priced hours happened to be broken in the site's favour, "
                          "which Proposition~\\ref{prop:tariff} does not guarantee."))

    duality = study.get("duality")
    if duality:
        gap_pct = 100 * (duality["gap_certified"] or 0.0)
        if gap_pct <= 0.05:
            duality_sentence = (f"Evaluating the dual function at the restricted-LP prices $\\pi^\\star$ gives "
                                f"$D(\\pi^\\star)={duality['dual_certified']:.2f}$/day against $C^\\star={duality['integrated_cost']:.2f}$/day: "
                                "the Lagrangian relaxation of the interface has no duality gap on this instance "
                                "(Proposition~\\ref{prop:convergence}(iv)), so the residual certified gap of the runs is the price "
                                "of finite iterations, quantized prices, and subproblem tolerances, not a structural limit.")
        else:
            duality_sentence = (f"Evaluating the dual function at the restricted-LP prices $\\pi^\\star$ gives "
                                f"$D(\\pi^\\star)={duality['dual_certified']:.2f}$/day against $C^\\star={duality['integrated_cost']:.2f}$/day, "
                                f"a duality gap of {gap_pct:.2f}\\% that no price can close (Proposition~\\ref{{prop:convergence}}(iv)); "
                                "the certified gaps of the runs are bounded below by it.")
    else:
        duality_sentence = ""

    def reach_text(run):
        k = run["primal_reached_optimum_iteration"]
        return f"at iteration {k + 1}" if k is not None else "at no iteration within the budget"

    text = f"""\\subsection{{Test System and Data}}
The water operator runs the SNET network of the repository: one ground-level reservoir, one fixed-speed pump
(design point 50~L/s at 40~m, wire-to-water efficiency 0.75, calibrated on-state power $\\overline{{P}}_p={pump_kw:.2f}$~kW),
two junctions, two pipes, and one elevated cylindrical tank ({tank['diameter_m']:.0f}~m diameter, {tank['elevation_m']:.0f}~m
elevation, level {tank['min_level_m']:.0f}--{tank['max_level_m']:.0f}~m, initial {tank['init_level_m']:.0f}~m). Daily
demand is {sum(case['water_demand_m3h']):,.0f}~m$^3$ with the diurnal pattern of Fig.~\\ref{{fig:inputs}}; a 20~m pressure
floor and terminal replenishment are enforced. The tank holds about {(tank['max_level_m'] - tank['min_level_m']) * math.pi * tank['diameter_m'] ** 2 / 4 / (sum(case['water_demand_m3h']) / met['integrated']['pump_on_hours']):.1f} pump-hours of water, so
the shiftable volume is tank-limited in the sense of the two-period model. The energy operator owns the two-bus feeder of the
repository with its daily load shape scaled to a {max(case['feeder_load_kw']):.0f}~kW peak, a {max(case['pv_available_kw']):.0f}~kW$_\\mathrm{{p}}$
PV array, a 200~kWh/50~kW battery ($\\eta=0.95$, initial and minimum terminal energy 100~kWh), and a connection rated
{case['import_limit_kw']:g}~kW import and {case['export_limit_kw']:g}~kW export. Day-ahead purchase prices are {price_blocks}
(unit/kWh); the sale price is {export_price:.2f}. The flat retail tariff of the baseline is $\\bar\\tau={tau:.2f}$, the
time average of the purchase price. Profiles are synthetic and illustrative (Table~\\ref{{tab:case}}).

\\begin{{table}}[!t]
\\caption{{Case-Study Data (Synthetic Profiles, One-Hour Intervals)}}
\\label{{tab:case}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lp{{5.4cm}}@{{}}}}
\\toprule
Item & Value \\\\
\\midrule
Water network & SNET: 1 reservoir, 1 pump, 2 junctions, 2 pipes, 1 tank \\\\
Tank & {tank['diameter_m']:.0f}~m diameter, elevation {tank['elevation_m']:.0f}~m, level {tank['min_level_m']:.0f}--{tank['max_level_m']:.0f}~m, initial {tank['init_level_m']:.0f}~m \\\\
Demand, pressure floor & {sum(case['water_demand_m3h']):,.0f}~m$^3$/day with daily pattern; 20~m \\\\
Pump & 50~L/s at 40~m; $\\eta_p=0.75$; $\\overline{{P}}_p={pump_kw:.2f}$~kW; 6 PWL segments \\\\
Pipes, line currents & 12 PWL segments (signed domain), 5 segments \\\\
Feeder & Two-bus circuit, load shape peaking at {max(case['feeder_load_kw']):.0f}~kW \\\\
PV, battery & {max(case['pv_available_kw']):.0f}~kW$_\\mathrm{{p}}$; 200~kWh, 50~kW, $\\eta=0.95$, $e^0=100$~kWh \\\\
Connection & {case['import_limit_kw']:g}~kW import, {case['export_limit_kw']:g}~kW export; $0.9\\le v\\le1.1$~p.u. \\\\
Prices (unit/kWh) & purchase {'/'.join(f'{p:.2f}' for p in prices)} (off/mid/on-peak); sale {export_price:.2f}; flat tariff {tau:.2f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[!t]
\\centering
\\includegraphics[width=\\columnwidth]{{inputs.pdf}}
\\caption{{Input profiles: day-ahead purchase and sale prices, water demand, feeder load, and available PV power.}}
\\label{{fig:inputs}}
\\end{{figure}}

\\subsection{{Experiments}}
Six experiments are run: (E1) the flat-tariff baseline and the two tariff benchmarks against the integrated optimum;
(E2) the payoffs of the revenue-sharing contract, the fee window, and a numerical deviation test; (E3) the replay of the
baseline and coordinated schedules in EPANET and OpenDSS; (E4) the price coordination scheme with the bundle
master started from the energy operator's marginal costs and from the market price, and with the subgradient rule of the two-agent model; (E5) sweeps of the import and export
ratings of the connection; and (E6) the restricted-LP internal prices that verify Proposition~\\ref{{prop:price}}.
All models were solved with {solver_name} {solver_version} through Pyomo; the integrated MILP has {integ['variables']}
variables and {integ['constraints']} constraints and was solved to a {100 * (integ['solver_relative_gap'] or 0):.3f}\\% relative gap in
{integ['solve_time_s']:.0f}~s, the coordination subproblems to a {100 * study['coordination_settings']['subproblem_rel_gap']:.1f}\\% gap. Timing includes the
Pyomo interface overhead and is not a solver benchmark.

\\subsection{{The Value of Coordination (E1, E6)}}
Table~\\ref{{tab:arrangements}} compares the arrangements. Under the flat tariff the water operator needs
{met['baseline_flat']['pump_on_hours']} pump-hours and, holding its level, places {block_kwh('baseline_flat')} (unit/kWh);
the site's day-ahead cost is {c_base:.2f}/day. The integrated optimum pumps {block_kwh('integrated')} and costs {c_star:.2f}/day,
a coordination gain $\\mathcal{{G}}={gain:.2f}$/day, {100 * con['gain_fraction_of_site_cost']:.1f}\\% of the site's bill and
{100 * con['gain_fraction_of_pumping_bill']:.0f}\\% of the water operator's baseline pumping bill of {con['bill0']:.2f}/day.
{how_sentence}

{internal_sentence} {tariff_sentence}

\\begin{{table}}[!t]
\\caption{{Arrangements Compared (Unit/Day). Gain and Capture Are Relative to the Baseline}}
\\label{{tab:arrangements}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\\toprule
Arrangement & Cost & Gain & Capt.\\ (\\%) & On (h) & Import & Export \\\\
\\midrule
{arrangement_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[!t]
\\centering
\\includegraphics[width=\\columnwidth]{{coordination_results.pdf}}
\\caption{{Baseline versus coordinated operation. Top to bottom: purchase and sale prices with the internal price
$\\pi^\\star_t$; pump load; tank level (markers are inventory boundaries, the dotted line the initial level); net import
at the connection (negative is export).}}
\\label{{fig:computed}}
\\end{{figure}}

\\subsection{{Dividing the Gain (E2)}}
Table~\\ref{{tab:payoffs}} lists the payoffs. Because the site is a net importer, the settlement is negative
($R^\\star={con['settlement_star']:.2f}$/day), and the contract is read as bill sharing: with the fee anchored on the baseline,
$F(\\varphi)=\\mathrm{{bill}}^0+\\varphi R^0$, the water operator's payoff rises by $\\varphi\\mathcal{{G}}$ and the energy
operator's by $(1-\\varphi)\\mathcal{{G}}$ at every share (Corollary~\\ref{{cor:window}}). At the equal split $\\varphi=1/2$
the fee is {nash['fee']:.2f}/day and each operator gains {nash['water_gain']:.2f}/day; for that share any fee in
$[{window[0]:.2f},\\,{window[1]:.2f}]$ leaves both operators no worse off than the baseline, an interval of width
$\\mathcal{{G}}$. The deviation test in Table~\\ref{{tab:deviation}} evaluates the water operator's contract payoff at
$\\varphi=1/2$ if it ran one of the other schedules instead, with the energy operator responding optimally: every
deviation loses, as Theorem~\\ref{{thm:contract}} requires.

\\begin{{table}}[!t]
\\caption{{Daily Payoffs by Arrangement (Unit/Day); Fee Anchored on the Baseline}}
\\label{{tab:payoffs}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\\toprule
Arrangement & $\\varphi$ & $F$ & $\\Pi_{{\\mathrm{{W}}}}$ & $\\Pi_{{\\mathrm{{E}}}}$ & $\\Pi$ & $\\Delta\\Pi_{{\\mathrm{{W}}}}$ & $\\Delta\\Pi_{{\\mathrm{{E}}}}$ \\\\
\\midrule
{chr(10).join(payoff_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table}}[!t]
\\caption{{Water Operator's Contract Payoff at $\\varphi=1/2$ for Alternative Schedules (Unit/Day)}}
\\label{{tab:deviation}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lrr@{{}}}}
\\toprule
Schedule run by W & $\\Pi_{{\\mathrm{{W}}}}$ & Deviation gain \\\\
\\midrule
{dev_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Distributed Computation (E4)}}
Table~\\ref{{tab:coordination}} and Fig.~\\ref{{fig:convergence}} summarize the price coordination runs. Started from the
energy operator's marginal cost of serving the metered baseline schedule, the first message of the recommended protocol, the
scheme certifies a gap of {100 * warm['initial_gap']:.2f}\\% after the first exchange, the recovered schedule attains the
integrated optimum {reach_text(warm)}, and the bundle master stops after {warm['iterations']} iterations, when its model
predicts no further improvement at the exchanged price resolution, with a dual bound of {warm['best_lower']:.2f}/day and a
certified gap of {100 * warm['final_gap']:.2f}\\%. Started from the market price instead, the first exchange certifies only
{100 * cold['initial_gap']:.2f}\\%, because the purchase price leaves the energy operator indifferent in every import hour;
the scheme attains the optimum {reach_text(cold)}, reaches a 1\\% certificate {'at iteration ' + str(cold['gap_within_1pct_iteration'] + 1) if cold.get('gap_within_1pct_iteration') is not None else 'at no iteration'},
and ends after {cold['iterations']} iterations with a gap of {100 * cold['final_gap']:.2f}\\%. The subgradient rule from the
marginal-cost start attains the optimum {reach_text(subg)} and reaches a gap of {100 * subg['final_gap']:.2f}\\% after
{subg['iterations']} iterations. {duality_sentence} For comparison, the central solver's own bound on the integrated MILP is
{integ['solver_lower_bound']:.2f}, a gap of {100 * (integ['solver_relative_gap'] or 0):.3f}\\%, and it holds both operators' data. Each iteration exchanges {2 * T} scalars; no demand forecast, tank state, pump
characteristic, PV forecast, battery state, or market position crosses the boundary.

\\begin{{table}}[!t]
\\caption{{Price Coordination Runs: Iterations, Messages, Certified Dual and Primal Values, Final Gap, the Iteration at Which the Recovered Schedule First Attains the Integrated Optimum, and Wall Time}}
\\label{{tab:coordination}}
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2pt}}
\\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\\toprule
Run & Iter. & Msgs & Dual & Primal & Gap \\% & Opt.\\ at & Min \\\\
\\midrule
{coord_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure*}}[!t]
\\centering
\\includegraphics[width=\\textwidth]{{convergence.pdf}}
\\caption{{Left: certified gap between the recovered primal value and the certified dual bound per iteration for the bundle
master from the marginal-cost and market-price starts and for the subgradient rule. Right: internal-price iterates at three
hours for the bundle run from the market-price start; thin lines mark the restricted-LP prices $\\pi^\\star_t$.}}
\\label{{fig:convergence}}
\\end{{figure*}}

\\subsection{{Connection-Limit Sensitivities (E5)}}
Table~\\ref{{tab:sweep}} and Fig.~\\ref{{fig:sensitivity}} vary the ratings of the connection. {import_sentence} {export_sentence}

\\begin{{table}}[!t]
\\caption{{Connection-Limit Sweeps (Unit/Day). ``Off Reference'' Lists the Non-Idle Hours in Which the Internal Price Departs from the Market Price of the Connection's Direction}}
\\label{{tab:sweep}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}lrrrrp{{2.2cm}}@{{}}}}
\\toprule
Limit & kW & Baseline & Integrated & Gain & Off reference \\\\
\\midrule
{chr(10).join(sweep_rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure*}}[!t]
\\centering
\\includegraphics[width=\\textwidth]{{sensitivity.pdf}}
\\caption{{Coordination gain versus the import rating (left) and the export rating (right) of the connection. Shaded bands
mark ratings at which only coordinated operation is feasible or no schedule is feasible.}}
\\label{{fig:sensitivity}}
\\end{{figure*}}

\\subsection{{Independent Nonlinear Replay (E3)}}
The baseline and coordinated pump schedules are replayed in EPANET with a five-minute hydraulic time step and hourly
reports, the level controls of the input file being replaced by the hourly commands, and the optimized net bus injections
are solved as balanced constant-power snapshots in OpenDSS. Table~\\ref{{tab:replay}} summarizes the discrepancies.
Both schedules satisfy the pressure floor and the voltage band at every report time and the imposed pump statuses match the
MILP decisions. {reserve_sentence} The electrical discrepancies are small because the feeder is short and lightly loaded,
the regime in which the flat-voltage linearization is most accurate. Report-time checks do not certify feasibility between
reports; the results support the schedules as planning-stage outputs, not as a certification for field operation.

\\begin{{table}}[!t]
\\caption{{Replay of the Baseline and Coordinated Schedules in EPANET and OpenDSS}}
\\label{{tab:replay}}
\\centering
\\footnotesize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{@{{}}lrr@{{}}}}
\\toprule
Metric & Baseline & Coordinated \\\\
\\midrule
{replay_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    (paper_dir / "results.tex").write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "data/inputs/paper_case.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/results/paper_study")
    parser.add_argument("--solver", default=None)
    parser.add_argument("--timeout", type=float, default=None, help="Solver time limit per case, in seconds")
    parser.add_argument("--paper-dir", type=Path, default=PROJECT_ROOT / "paper")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override the coordination iteration budget")
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-typeset results.tex, results_macros.tex and the figures from the saved study without solving")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse coordination runs already saved in the output directory instead of re-running them")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config, solver=args.solver, timeout=args.timeout)
    if args.regenerate:
        study, histories, schedules = load_saved_study(args.output_dir.resolve())
        make_figures(study, histories, schedules, config, args.paper_dir.resolve())
        write_results(study, config, args.paper_dir.resolve())
        print(f"Regenerated manuscript results from {args.output_dir}")
        return
    if args.max_iterations is not None:
        config.setdefault("coordination", {})["max_iterations"] = args.max_iterations
        config["coordination"]["subgradient_iterations"] = min(
            args.max_iterations, config["coordination"].get("subgradient_iterations", DEFAULT_COORDINATION["subgradient_iterations"]))
    study = run_study(config, args.output_dir.resolve(), args.paper_dir.resolve(), resume=args.resume)
    print(json.dumps({"gain": study["contract"]["gain"], "integrated_cost": study["integrated"]["energy_cost"],
                      "coordination": study["coordination"]}, indent=2))


if __name__ == "__main__":
    main()
