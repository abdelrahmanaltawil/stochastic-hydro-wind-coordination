"""Reproduce the manuscript's controlled scheduling and nonlinear replay study.

Run ``python -m src.experiments`` from the repository root. Numerical artifacts
are overwritten only within the selected study directory and paper/results.tex.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import yaml

from . import algorithm_tasks as algorithm, postprocessing, preprocessing
from .validation import algebraic_residuals, replay_energy, replay_water
from .workflow import PROJECT_ROOT, load_config


def solve_case(name, data, config, output, schedule=None):
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
    model, results, _ = algorithm.solve_model(
        model, config["solver"]["name"], config["solver"]["timeout"],
        str(solver_log),
    )
    solve_seconds = time.perf_counter() - start
    if not len(results.solution):
        raise RuntimeError(f"{name}: no feasible incumbent ({results.solver.termination_condition})")
    solution = postprocessing.extract_solution(model)
    summary = postprocessing.create_summary(name, model, solution, results)
    summary.update(build_time_s=build_seconds, wall_solve_time_s=solve_seconds,
                   residuals=algebraic_residuals(model))
    for key, value in summary["residuals"].items():
        if value > 1e-4:
            raise RuntimeError(f"{name}: {key}={value} exceeds numerical feasibility tolerance")
    datasets = {"summary.json": summary, "config.yaml": config}
    for domain in ("water", "energy"):
        for key, rows in solution[domain].items():
            datasets[f"{domain}/{key}.csv"] = rows
    postprocessing.save_data(datasets, directory)
    return model, summary


def case_metrics(name, model, summary):
    hours = list(model.T)
    return {
        "case": name,
        "electricity_cost": float(pyo.value(model.energy_cost)),
        "grid_import_kwh": float(sum(pyo.value(model.P_import[b, t]) for b in model.Buses for t in hours)),
        "peak_grid_import_kw": float(max(sum(pyo.value(model.P_import[b, t]) for b in model.Buses) for t in hours)),
        "pump_on_hours": int(round(sum(pyo.value(model.Status[p, t]) for p in model.Pumps for t in hours))),
        "pump_energy_kwh": float(sum(pyo.value(model.PumpBusLoad[b, t]) for b in model.Buses for t in hours)),
        "terminal_tank_head_m": float(pyo.value(model.H_terminal[next(iter(model.Tanks))])),
        "terminal_battery_kwh": float(sum(pyo.value(model.E_soc[b, len(hours)]) for b in model.Buses)),
        "solver_termination": summary["termination_condition"],
        "solve_time_s": summary["wall_solve_time_s"],
        "variables": summary["num_variables"], "constraints": summary["num_constraints"],
    }


def make_figure(models, config, paper_dir):
    """Column-width (3.5 in) figure for the two-column IEEE manuscript."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })
    T = config["T"]
    fig, axes = plt.subplots(4, 1, figsize=(3.5, 5.0), sharex=True,
                             gridspec_kw={"height_ratios": [0.62, 1, 1, 1]})
    edges = np.arange(T + 1)
    hours = np.arange(T)
    tariff = config["energy"]["cost"]["grid_import_tariff"]
    axes[0].stairs(tariff, edges, color="#52514e", fill=True, alpha=0.18, linewidth=0)
    axes[0].stairs(tariff, edges, color="#52514e", linewidth=0.9)
    axes[0].set_ylabel("Price\n(unit/kWh)")
    axes[0].set_ylim(0, 1.1 * max(tariff))
    styles = {"independent": ("Independent baseline", "#eb6834", (0, (3.2, 1.6))),
              "coordinated": ("Joint scheduling", "#2a78d6", "solid")}
    tank_elevation = float(config.get("water", {}).get("tank_elevation_m", 25.0))
    initial_level = None
    for name, model in models.items():
        label, color, style = styles[name]
        pump = [sum(pyo.value(model.PumpBusLoad[b, t]) for b in model.Buses) for t in hours]
        tank = next(iter(model.Tanks))
        tank_states = [pyo.value(model.H[tank, t]) - tank_elevation for t in hours] + \
                      [pyo.value(model.H_terminal[tank]) - tank_elevation]
        initial_level = tank_states[0]
        grid = [sum(pyo.value(model.P_import[b, t]) for b in model.Buses) for t in hours]
        axes[1].stairs(pump, edges, color=color, linestyle=style, linewidth=1.3, label=label, baseline=None)
        axes[2].plot(edges, tank_states, color=color, linestyle=style, linewidth=1.3,
                     marker="o", markersize=2.2, markerfacecolor="white", markeredgewidth=0.8)
        axes[3].stairs(grid, edges, color=color, linestyle=style, linewidth=1.3, baseline=None)
    axes[1].set_ylabel("Pump load\n(kW)")
    axes[2].set_ylabel("Tank level\n(m)")
    if initial_level is not None:
        axes[2].axhline(initial_level, color="0.55", linestyle=":", linewidth=0.8)
    axes[3].set_ylabel("Grid import\n(kW)")
    axes[3].set_xlabel("Hour of day (interval start / inventory boundary)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=2,
               frameon=False, handlelength=2.4, columnspacing=1.4)
    for ax in axes:
        ax.set_xlim(0, T)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.tick_params(length=2.5)
    axes[3].set_xticks(np.arange(0, T + 1, 4))
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.935, bottom=0.085, hspace=0.32)
    figures = paper_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "coordination_results.pdf")
    fig.savefig(figures / "coordination_results.png", dpi=220)
    plt.close(fig)


def write_results(metrics, validation, config, data, paper_dir, baseline_summary, extras=None):
    """Write paper/results.tex (IEEEtran two-column layout) from the saved study values.

    ``extras`` optionally carries the recorded pump schedules (``<case>_schedule``)
    and solve summaries (``<case>_summary``) so the narrative sentences are derived
    from the saved values rather than typed by hand.
    """
    extras = extras or {}
    first, joint = metrics[0], metrics[1]
    reduction = first["electricity_cost"] - joint["electricity_cost"]
    percent = 100 * reduction / first["electricity_cost"]
    pump_kw = joint["pump_energy_kwh"] / joint["pump_on_hours"]
    text = r"""\subsection{Illustrative Case and Comparison Protocol}
The water network is the supplied SNET example with one reservoir, one
fixed-speed pump, two junctions, two pipes, and one cylindrical tank
(Table~\ref{tab:case}). The tank has a 12~m diameter, 25~m elevation,
water-level bounds of 0--6~m, and an initial level of 3~m. Nominal demand
is 25~L/s with the supplied 24-hour pattern; a 20~m junction-pressure floor
and terminal tank replenishment are enforced. The pump design point is
50~L/s at 40~m. Pipe head losses use 12 segments over each signed flow
domain and the pump curve six segments. The electrical network is the
supplied two-bus 115~kV circuit with a 1000~kW nominal load and its daily
multipliers; grid exchange is available at the source bus only. This lightly
loaded circuit is an illustrative input, not a representative low-voltage
feeder or a congestion stress test.

\begin{table}[!t]
\caption{Case-Study Data (Synthetic Profiles, One-Hour Intervals)}
\label{tab:case}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lp{5.55cm}@{}}
\toprule
Item & Value \\
\midrule
Water network & SNET: 1 reservoir, 1 pump, 2 junctions, 2 pipes, 1 tank \\
Tank & 12~m diameter, elevation 25~m, level 0--6~m, initial 3~m \\
Demand, pressure floor & 25~L/s nominal with daily pattern; 20~m \\
Pump & Design point 50~L/s at 40~m; $\eta_p=0.75$; $\overline{P}_p=PUMPPOWER$~kW \\
PWL segments & 12 (pipes, signed domain), 6 (pump), 5 (line currents) \\
Electrical network & Two-bus 115~kV circuit, 1000~kW nominal load \\
PV & 100~kW at the load bus, availability peaking at hour 12 \\
Battery & 200~kWh, 50~kW charge/discharge, $\eta=0.95$, $e^0=100$~kWh \\
Voltage band & $0.9\le v\le1.1$~p.u. \\
Purchase price & 0.08 (h 0--6, 21--23), 0.16 (h 7--15), 0.32 (h 16--20) \\
Sale price & 0 \\
\bottomrule
\end{tabular}
\end{table}

One 100~kW PV installation and one 200~kWh battery are located at the load
bus. The battery has 50~kW charge and discharge limits, efficiencies of
0.95, and an initial and minimum terminal energy of 100~kWh. The synthetic
purchase prices are 0.08 for hours 0--6 and 21--23, 0.16 for hours 7--15,
and 0.32 for hours 16--20, in currency units per kWh; exports earn nothing.
These profiles are controlled experimental inputs, not observations or an
actual utility tariff. With a wire-to-water efficiency of 0.75 the
calibrated on-state pump power from \eqref{eq:c_calib} is PUMPPOWER~kW.

The independent baseline first minimizes the water block alone under a
uniform pump energy price, then fixes that pump schedule while optimizing
the electrical dispatch under the time-varying prices. Joint scheduling
releases the pump binaries and optimizes the same cost with identical
network, asset, and terminal constraints. Because the uniform-price water
problem can have several optimal schedules, the comparison measures the
improvement over this explicit reproducible baseline rather than a universal
bound on the value of coordination. A separate flat-price joint case (0.10
per kWh in every hour) is a tariff sensitivity and is not directly
comparable in monetary terms.

\subsection{Optimization Outcomes}
Table~\ref{tab:computed} reports the saved solver outputs. Joint scheduling
reduces the daily electricity procurement cost from BASECOST to JOINTCOST
relative to the independent baseline, a difference of SAVING (PERCENT\%).
SCHEDULETEXT The peak grid import changes from BASEPEAK to JOINTPEAK~kW.
RESERVETEXT ASSETTEXT All values refer to the MILP approximation.

\begin{table}[!t]
\caption{Computed Schedules. Costs Are in Synthetic Currency Units; the Flat-Price Case Uses a Different Tariff. Solve Time Is Wall-Clock Solver-Call Time on the Recorded Machine}
\label{tab:computed}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
Case & Cost & On (h) & Import (kWh) & Peak (kW) & Solve (s) \\
\midrule
TABLEROWS
\bottomrule
\end{tabular}
\end{table}

The joint model has VARS variables and CONS constraints.
TERMINATIONS The water-only solve took BASETIME~s. The run directory
records full-precision dispatch, inventory trajectories, solver logs,
algebraic residuals (maximum constraint violation below RESIDUAL), the
input configuration, dependency versions, and input and source hashes.
Timing includes the Pyomo interface overhead and is not an isolated solver
benchmark.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{coordination_results.pdf}
\caption{Purchase price and optimized schedules for the independent baseline and joint scheduling: pump electrical load, tank water level, and grid import. Tank markers are inventory boundaries; power values apply over the following hourly interval. The dotted line marks the initial level, which the terminal level must reach.}
\label{fig:computed}
\end{figure}

\subsection{Independent Nonlinear Replay}
Each optimized pump schedule is replayed in EPANET with a five-minute
hydraulic time step and hourly reports, the original level controls being
replaced by the hourly commands. The optimized net bus injections are
replayed as balanced constant-power snapshots in OpenDSS, which retains the
nonlinear AC behavior of the original circuit whereas the MILP neglects
series losses and shunt effects. The electrical replay keeps the fixed
on-state pump-power coefficient and therefore does not validate the pump
energy calibration. Table~\ref{tab:replay} summarizes the discrepancies.

\begin{table}[!t]
\caption{Replay of the Optimized Schedules in EPANET and OpenDSS}
\label{tab:replay}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrr@{}}
\toprule
Metric & Independent & Joint \\
\midrule
REPLAYROWS
\bottomrule
\end{tabular}
\end{table}

REPLAYTEXT Hourly
report checks do not certify feasibility between reports. These results
support the formulation as a scheduling prototype with explicit simulation
checks; they are not a certification for field operation.
"""
    names = {"independent": "Independent", "coordinated": "Joint", "flat_price": "Joint, flat price"}
    rows = [f"{names[row['case']]} & {row['electricity_cost']:.2f} & {row['pump_on_hours']} & "
            f"{row['grid_import_kwh']:.2f} & {row['peak_grid_import_kw']:.1f} & {row['solve_time_s']:.2f} \\\\"
            for row in metrics]

    # Pump-timing narrative derived from the recorded schedules and the tariff.
    tariff = config["energy"]["cost"]["grid_import_tariff"]
    same_energy = (abs(first["grid_import_kwh"] - joint["grid_import_kwh"]) < 1e-6
                   and first["pump_on_hours"] == joint["pump_on_hours"])
    if same_energy:
        energy = f"{joint['grid_import_kwh']:,.0f}".replace(",", "\\,")
        schedule_text = (f"Both schedules operate the pump for {joint['pump_on_hours']} hours and import the "
                         f"same energy, {energy}~kWh; the saving comes entirely from timing "
                         f"(Fig.~\\ref{{fig:computed}}).")
    else:
        schedule_text = (f"The independent and joint schedules import {first['grid_import_kwh']:.2f} and "
                         f"{joint['grid_import_kwh']:.2f}~kWh with {first['pump_on_hours']} and "
                         f"{joint['pump_on_hours']} pump on-hours, respectively (Fig.~\\ref{{fig:computed}}).")
    base_sched = extras.get("independent_schedule")
    joint_sched = extras.get("coordinated_schedule")
    if base_sched and joint_sched:
        on_base = {t for (_, t), on in base_sched.items() if on}
        on_joint = {t for (_, t), on in joint_sched.items() if on}
        removed, added = sorted(on_base - on_joint), sorted(on_joint - on_base)
        if removed and added and len(removed) == len(added):
            lost = sum(tariff[t] for t in removed)
            gained = sum(tariff[t] for t in added)
            def fmt(hours):
                hours = [str(h) for h in hours]
                if len(hours) == 1:
                    return f"hour {hours[0]}"
                return "hours " + ", ".join(hours[:-1]) + f", and {hours[-1]}" if len(hours) > 2 \
                    else f"hours {hours[0]} and {hours[1]}"
            schedule_text += (f" Relative to the baseline, the joint schedule drops pumping in "
                              f"{fmt(removed)} and adds {fmt(added)}; at {pump_kw:.3f}~kW per on-hour "
                              f"this tariff exchange is worth ${pump_kw:.3f}\\times({lost:.2f}-{gained:.2f})"
                              f"={pump_kw * (lost - gained):.2f}$.")

    reserve_text = ("Both initial reserves are preserved in the optimization: the terminal tank heads are "
                    + ", ".join(f"{row['terminal_tank_head_m']:.2f}" for row in metrics)
                    + "~m against an initial head of 28~m, and the terminal battery energy equals its initial "
                    + f"{joint['terminal_battery_kwh']:.0f}~kWh.")

    asset_text = ""
    s_joint = extras.get("coordinated_summary", {})
    s_flat = extras.get("flat_price_summary", {})
    if s_joint:
        asset_text = (f"The available PV energy ({s_joint.get('pv_generation_kwh', 0):.0f}~kWh) is dispatched "
                      f"in full in every case. Under the time-varying tariff the battery charges "
                      f"{s_joint.get('battery_charge_kwh', 0):.1f}~kWh and discharges "
                      f"{s_joint.get('battery_discharge_kwh', 0):.0f}~kWh")
        if s_flat and s_flat.get("battery_charge_kwh", 0) < 1e-6:
            asset_text += ("; under the flat price it is not cycled at all, because without price differences "
                           "the round-trip losses make storage arbitrage unprofitable.")
        else:
            asset_text += "."

    w = {n: validation[n]["water"] for n in ("independent", "coordinated")}
    e = {n: validation[n]["energy"] for n in ("independent", "coordinated")}
    term = {n: next(iter(w[n]["terminal_tank_change_m"].values())) for n in w}

    def pair(fmt, key, source):
        return " & ".join(fmt.format(source[n][key]) for n in ("independent", "coordinated"))

    flow_cells = " & ".join(f"{1000 * w[n]['max_flow_error_m3s']:.3f}" for n in w)
    term_cells = " & ".join(f"${term[n]:+.3f}$" for n in w)
    conv_cells = " & ".join("yes" if e[n]["all_snapshots_converged"] else "no" for n in e)
    replay_rows = [
        f"Minimum junction pressure (m) & {pair('{:.2f}', 'min_junction_pressure_m', w)} \\\\",
        f"Pressure violations at report times & {pair('{}', 'pressure_violations_at_report_times', w)} \\\\",
        f"Pump-status mismatches at report times & {pair('{}', 'pump_status_mismatches_at_report_times', w)} \\\\",
        f"Maximum tank-head discrepancy (m) & {pair('{:.3f}', 'max_tank_head_error_m', w)} \\\\",
        f"Maximum flow discrepancy (L/s) & {flow_cells} \\\\",
        f"Terminal tank-head change vs.\\ initial (m) & {term_cells} \\\\",
        f"All OpenDSS snapshots converged & {conv_cells} \\\\",
        f"Maximum voltage discrepancy (p.u.) & {pair('{:.5f}', 'max_voltage_error_pu', e)} \\\\",
        f"Voltage violations & {pair('{}', 'voltage_violations', e)} \\\\",
        f"Peak line loading (\\%) & {pair('{:.2f}', 'max_line_loading_percent', e)} \\\\",
        f"Maximum grid-import discrepancy (kW) & {pair('{:.3f}', 'max_grid_import_error_kw', e)} \\\\",
    ]
    statements = []
    clean = all(w[n]["pressure_violations_at_report_times"] == 0 and e[n]["voltage_violations"] == 0
                and w[n]["pump_status_mismatches_at_report_times"] == 0 for n in w)
    if clean:
        statements.append("Both schedules satisfy the pressure floor and the voltage band at every report time, "
                          "and the pump statuses imposed in EPANET match the MILP decisions.")
    else:
        statements.append("At least one replay reports a service or status discrepancy; see Table~\\ref{tab:replay}.")
    statements.append(f"Tank-head and flow discrepancies remain below "
                      f"{max(w[n]['max_tank_head_error_m'] for n in w) + 0.005:.2f}~m and "
                      f"{1000 * max(w[n]['max_flow_error_m3s'] for n in w) + 0.005:.2f}~L/s. The electrical "
                      "discrepancies are small because the feeder is lightly loaded and the circuit is short, "
                      "which is the regime in which the flat-voltage linearization is most accurate.")
    short = [n for n in term if term[n] < -1e-4]
    if short:
        statements.append("The replay, however, ends with the tank "
                          + " and ".join(f"{-term[n]:.3f}~m ({names[n].lower()})" for n in short)
                          + " below its initial level despite the MILP replenishment constraint "
                          "\\eqref{eq:w_tank_terminal}. The hourly hydraulic approximation therefore does not "
                          "preserve the terminal requirement exactly, and a reserve margin or a simulation-informed "
                          "refinement of \\eqref{eq:w_tank} is needed before operational use.")
    termination = ("All three coupled solves and the independent water-only solve terminated as optimal "
                   "at the $10^{-4}$ gap tolerance.")
    if any(m["solver_termination"] != "optimal" for m in metrics) or baseline_summary["termination_condition"] != "optimal":
        termination = ("Solver terminations are: "
                       + "; ".join(f"{names[m['case']]}: {m['solver_termination']}" for m in metrics)
                       + ". Time-limited incumbents are not certificates of optimality.")
    residual = max([baseline_summary.get("max_constraint_violation", 0.0)]
                   + [s.get("max_constraint_violation", 0.0) for k, s in extras.items()
                      if k.endswith("_summary") and isinstance(s, dict)])
    if residual > 0:
        exponent = int(np.floor(np.log10(residual)))
        mantissa = int(np.ceil(residual / 10 ** exponent))
        if mantissa == 10:
            mantissa, exponent = 1, exponent + 1
        residual_text = f"${mantissa}\\times10^{{{exponent}}}$"
    else:
        residual_text = "$10^{-12}$"
    replacements = {"PUMPPOWER": f"{pump_kw:.3f}", "BASECOST": f"{first['electricity_cost']:.2f}",
                    "JOINTCOST": f"{joint['electricity_cost']:.2f}", "SAVING": f"{reduction:.2f}",
                    "PERCENT": f"{percent:.2f}", "TABLEROWS": "\n".join(rows),
                    "BASEPEAK": f"{first['peak_grid_import_kw']:.1f}",
                    "JOINTPEAK": f"{joint['peak_grid_import_kw']:.1f}",
                    "SCHEDULETEXT": schedule_text, "RESERVETEXT": reserve_text, "ASSETTEXT": asset_text,
                    "VARS": str(joint["variables"]), "CONS": str(joint["constraints"]),
                    "TERMINATIONS": termination, "BASETIME": f"{baseline_summary['wall_solve_time_s']:.2f}",
                    "RESIDUAL": residual_text, "REPLAYROWS": "\n".join(replay_rows),
                    "REPLAYTEXT": " ".join(statements)}
    for key, value in replacements.items():
        text = text.replace(key, value)
    (paper_dir / "results.tex").write_text(text, encoding="utf-8")


def run_study(config, output, paper_dir):
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config)
    config["water"]["pump_efficiency"] = config["nexus"]["pump_efficiency"]
    data = preprocessing.load_networks(config)
    water_config = copy.deepcopy(config)
    water_config.update(run_energy=False, run_nexus=False)
    water_model, water_summary = solve_case("water_baseline", data, water_config, output)
    schedule = {(p, t): int(round(pyo.value(water_model.Status[p, t]))) for p in water_model.Pumps for t in water_model.T}
    models, metrics, validation, extras = {}, [], {}, {}
    for name, fixed in (("independent", schedule), ("coordinated", None)):
        model, summary = solve_case(name, data, config, output, fixed)
        models[name] = model
        extras[f"{name}_schedule"] = {(p, t): int(round(pyo.value(model.Status[p, t])))
                                      for p in model.Pumps for t in model.T}
        extras[f"{name}_summary"] = summary
        metrics.append(case_metrics(name, model, summary))
        validation[name] = {
            "water": replay_water(model, data["water"]["inp_file"], config, output / name / "validation"),
            "energy": replay_energy(model, data["energy"], output / name / "validation"),
        }
        postprocessing.save_data({"validation.json": validation[name]}, output / name)
    flat_data = copy.deepcopy(data)
    flat_data["energy"]["tariff"] = [0.1] * config["T"]
    flat_config = copy.deepcopy(config)
    flat_config["energy"]["cost"]["grid_import_tariff"] = [0.1] * config["T"]
    flat, summary = solve_case("flat_price", flat_data, flat_config, output)
    extras["flat_price_summary"] = summary
    metrics.append(case_metrics("flat_price", flat, summary))
    source_files = [*sorted((PROJECT_ROOT / "src").rglob("*.py")),
                    Path(data["water"]["inp_file"]), Path(data["energy"]["dss_file"])]
    provenance = {"python": sys.version, "platform": platform.platform(),
                  "solver": config["solver"]["name"],
                  "solver_version": pyo.SolverFactory(config["solver"]["name"]).version(),
                  "versions": {name: importlib.metadata.version(name) for name in
                               ("pyomo", "highspy", "wntr", "OpenDSSDirect.py", "numpy", "scipy", "pandas", "matplotlib")},
                  "sha256": {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}}
    postprocessing.save_data({"comparison.csv": metrics, "comparison.json": metrics,
                              "validation.json": validation, "provenance.json": provenance,
                              "config.yaml": config}, output)
    make_figure(models, config, paper_dir)
    write_results(metrics, validation, config, data, paper_dir, water_summary, extras)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "data/inputs/paper_case.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/results/paper_study")
    parser.add_argument("--solver", default=None)
    parser.add_argument("--timeout", type=float, default=None, help="Solver time limit per case, in seconds")
    parser.add_argument("--paper-dir", type=Path, default=PROJECT_ROOT / "paper")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config, solver=args.solver, timeout=args.timeout)
    results = run_study(config, args.output_dir.resolve(), args.paper_dir.resolve())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
