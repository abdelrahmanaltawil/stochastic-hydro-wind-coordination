"""Extract complete model trajectories and save reproducible run artifacts."""

import datetime
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo
import yaml

from .helpers import utils


def _value(component):
    value = pyo.value(component, exception=False)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Cannot extract an uninitialized or non-finite value: {component.name}")
    return float(value)


def extract_solution(model: pyo.ConcreteModel) -> dict:
    """Extract operating intervals and all storage boundary states.

    A missing objective yields empty domains. A partially loaded solution raises
    an error instead of silently saving an incomplete trajectory. Raw numerical
    precision is retained so balance and small hydraulic flows can be audited.
    """
    objective = pyo.value(getattr(model, "objective", None), exception=False)
    if objective is None:
        return {"objective": None, "water": {}, "energy": {}}
    if not math.isfinite(float(objective)):
        raise ValueError("The model objective is non-finite")
    return {
        "objective": float(objective),
        "water": _extract_water(model) if hasattr(model, "Q") else {},
        "energy": _extract_energy(model) if hasattr(model, "P_import") else {},
    }


def _extract_water(model: pyo.ConcreteModel) -> dict:
    flows = [{"link": l, "time": t, "flow_rate": _value(model.Q[l, t])}
             for l in model.Links for t in model.T]
    heads = [{"node": n, "time": t, "head": _value(model.H[n, t])}
             for n in model.Nodes for t in model.T]
    if hasattr(model, "H_terminal"):
        heads.extend({"node": n, "time": len(model.T), "head": _value(model.H_terminal[n])}
                     for n in model.Tanks)
    pump_status = []
    for pump in model.Pumps:
        for t in model.T:
            status = int(round(_value(model.Status[pump, t])))
            row = {"pump": pump, "time": t, "status": status}
            powers = getattr(model, "pump_mean_power_kw", {})
            if pump in powers:
                row["power_kw"] = float(powers[pump]) * status
            pump_status.append(row)
    slack = []
    for n in model.Junctions:
        for t in model.T:
            positive, negative = _value(model.SlackPos[n, t]), _value(model.SlackNeg[n, t])
            if positive > 1e-8 or negative > 1e-8:
                slack.append({"node": n, "time": t, "slack_pos": positive, "slack_neg": negative})
    return {"flows": flows, "heads": heads, "pump_status": pump_status, "slack": slack}


def _extract_energy(model: pyo.ConcreteModel) -> dict:
    dispatch, soc, line_flows, voltages = [], [], [], []
    for bus in model.Buses:
        for t in model.T:
            row = {"bus": bus, "time": t}
            for name in ("P_import", "P_export", "P_pv", "Q_ch", "Q_dis", "Q_import", "Q_grid", "PumpBusLoad"):
                if hasattr(model, name):
                    row[name] = _value(getattr(model, name)[bus, t])
            dispatch.append(row)
            voltages.append({"bus": bus, "time": t, "U": _value(model.U[bus, t]),
                             "theta": _value(model.theta[bus, t])})
        for t in getattr(model, "StateT", model.T):
            soc.append({"bus": bus, "time": t, "E_soc": _value(model.E_soc[bus, t])})
    for line in model.ELines:
        for t in model.T:
            row = {"line": line, "time": t}
            for name in ("P_line", "Q_line", "I_re", "I_im", "phi", "chi"):
                if hasattr(model, name):
                    row[name] = _value(getattr(model, name)[line, t])
            line_flows.append(row)
    return {"dispatch": dispatch, "soc": soc, "line_flows": line_flows, "voltages": voltages}


def create_summary(run_id: str, model: pyo.ConcreteModel, results: dict, solver_results) -> dict:
    """Summarize solver quality, costs, demand slack, and energy accounting."""
    solver = solver_results.solver
    solver_time = getattr(solver, "time", None)
    try:
        solver_time = float(solver_time)
    except (ValueError, TypeError):
        solver_time = None
    if solver_time is not None and not math.isfinite(solver_time):
        solver_time = None
    summary = {
        "run_id": run_id, "solver_status": str(solver.status),
        "termination_condition": str(solver.termination_condition),
        "objective_value": results["objective"], "solver_time_s": solver_time,
        "num_variables": sum(1 for _ in model.component_data_objects(pyo.Var, active=True)),
        "num_constraints": sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True)),
    }
    for source, target in (("lower_bound", "solver_lower_bound"), ("upper_bound", "solver_upper_bound")):
        try:
            bound = float(getattr(solver_results.problem, source))
            summary[target] = bound if math.isfinite(bound) else None
        except (AttributeError, TypeError, ValueError):
            summary[target] = None
    lower, upper = summary["solver_lower_bound"], summary["solver_upper_bound"]
    summary["solver_relative_gap"] = (
        max(0.0, upper - lower) / max(abs(upper), 1e-10)
        if lower is not None and upper is not None else None
    )
    if results["objective"] is None:
        return summary
    dt_hours = float(pyo.value(getattr(model, "dt", 3600))) / 3600.0
    summary["horizon_hours"] = len(model.T) * dt_hours
    for name in ("water_cost", "energy_cost"):
        if hasattr(model, name):
            summary[name] = _value(getattr(model, name))
    slack = results.get("water", {}).get("slack", [])
    if results.get("water"):
        summary["water_slack_violations"] = len(slack)
        summary["water_slack_volume_m3"] = sum(r["slack_pos"] + r["slack_neg"] for r in slack) * dt_hours * 3600
        summary["pump_on_hours"] = sum(r["status"] for r in results["water"].get("pump_status", [])) * dt_hours
    dispatch = results.get("energy", {}).get("dispatch", [])
    if dispatch:
        for name, key in (("grid_import_kwh", "P_import"), ("grid_export_kwh", "P_export"),
                          ("pv_generation_kwh", "P_pv"), ("battery_charge_kwh", "Q_ch"),
                          ("battery_discharge_kwh", "Q_dis"), ("pump_energy_kwh", "PumpBusLoad")):
            summary[name] = sum(row.get(key, 0.0) for row in dispatch) * dt_hours
    max_violation = 0.0
    for constraint in model.component_data_objects(pyo.Constraint, active=True):
        body = _value(constraint.body)
        if constraint.has_lb():
            max_violation = max(max_violation, float(pyo.value(constraint.lower)) - body)
        if constraint.has_ub():
            max_violation = max(max_violation, body - float(pyo.value(constraint.upper)))
    summary["max_constraint_violation"] = max_violation
    return summary


def save_run_metadata(save_path: Path, metadata: dict, experiment_parameters: dict,
                      network_files: dict, logger: logging.Logger | None = None,
                      solver_log_path: str | None = None) -> dict:
    """Save source inputs and provenance; write failures propagate to the caller."""
    save_path = Path(save_path)
    (save_path / "inputs").mkdir(parents=True, exist_ok=True)
    start_time = datetime.datetime.fromisoformat(metadata["execution_start_time"])
    end_time = datetime.datetime.now(tz=start_time.tzinfo)
    saved = dict(metadata)
    saved.update({
        "execution_end_time": end_time.isoformat(),
        "execution_duration_min": (end_time - start_time).total_seconds() / 60.0,
        "git_commit": utils.get_git_revision_hash(), "python_version": sys.version,
        "platform": platform.platform(), "working_directory": os.getcwd(),
        "command": " ".join(sys.argv),
        "solver_log": str(solver_log_path) if solver_log_path else None,
    })
    try:
        saved["git_worktree_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        saved["git_worktree_dirty"] = None
    sources = {}
    for domain, network in network_files.items():
        source = (network.get("inp_file") or network.get("dss_file")) if isinstance(network, dict) else network
        if not source:
            continue
        source = Path(source).resolve()
        destination = save_path / "inputs" / domain / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copy2(source, destination)
        sources[domain] = {"original_path": str(source), "saved_path": str(destination.relative_to(save_path)),
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    saved["network_sources"] = sources
    saved["package_versions"] = {}
    for package in ("pyomo", "wntr", "opendssdirect.py", "numpy", "pandas", "scipy"):
        try:
            saved["package_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            saved["package_versions"][package] = "unavailable"
    save_data({"00_run_metadata.yaml": saved,
               "inputs/00_experiment_parameters.yaml": experiment_parameters}, save_path)
    return saved


def _serializable(value):
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_data(datasets: dict, save_path: Path) -> None:
    """Write CSV, JSON, and YAML datasets using their declared formats."""
    save_path = Path(save_path)
    for filename, data in datasets.items():
        if data is None:
            continue
        full_path = save_path / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        if full_path.suffix == ".json":
            full_path.write_text(json.dumps(_serializable(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        elif full_path.suffix in (".yaml", ".yml"):
            full_path.write_text(yaml.safe_dump(_serializable(data), sort_keys=False), encoding="utf-8")
        elif full_path.suffix == ".csv":
            frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            frame.to_csv(full_path, index=False)
        else:
            raise ValueError(f"Unsupported result file type: {filename}")
        logging.info("Saved %s", full_path)


def save_results(results: dict, summary: dict, run_dir: Path) -> None:
    """Save every extracted domain table and a top-level summary."""
    run_dir = Path(run_dir)
    datasets = {f"{domain}/{name}.csv": rows for domain in ("water", "energy")
                for name, rows in results.get(domain, {}).items()}
    datasets["summary.json"] = {**summary, "run_id": run_dir.name}
    save_data(datasets, run_dir)
