"""Postprocessing — solution extraction and result saving.

Handles both water and energy results, writing to run_dir/water/ and
run_dir/energy/ subdirectories respectively.
"""

# imports
import datetime
import yaml
import logging
import subprocess
import sys
import os
import platform
import socket
import getpass
from pathlib import Path
import json
import shutil
import pandas as pd
import pyomo.environ as pyo

from helpers import utils



def extract_solution(model: pyo.ConcreteModel) -> dict:
    """Extract all solved variable values from the model.

    Returns:
        Dict with 'objective' and domain sub-dicts 'water' and/or 'energy'.
    """
    try:
        obj_val = pyo.value(model.objective)
    except (ValueError, TypeError):
        logging.warning("Model objective has no value — returning empty results.")
        return {"objective": None, "water": {}, "energy": {}}

    results = {"objective": obj_val, "water": {}, "energy": {}}

    if hasattr(model, "Q"):
        results["water"] = _extract_water(model)

    if hasattr(model, "P_import"):
        results["energy"] = _extract_energy(model)

    logging.info("Extracting solution...")

    return results


def _extract_water(model: pyo.ConcreteModel) -> dict:
    flows, heads, pump_status, slack = [], [], [], []

    for l in model.Links:
        for t in model.T:
            flows.append({"link": l, "time": t, "flow_rate": round(pyo.value(model.Q[l, t]), 4)})

    for n in model.Nodes:
        for t in model.T:
            heads.append({"node": n, "time": t, "head": round(pyo.value(model.H[n, t]), 4)})

    for p in model.Pumps:
        for t in model.T:
            try:
                pump_status.append({"pump": p, "time": t,
                                    "status": int(round(pyo.value(model.Status[p, t])))})
            except Exception:
                pass

    for n in model.Junctions:
        for t in model.T:
            vp = pyo.value(model.SlackPos[n, t])
            vn = pyo.value(model.SlackNeg[n, t])
            if vp > 1e-6 or vn > 1e-6:
                slack.append({"node": n, "time": t, "slack_pos": vp, "slack_neg": vn})

    return {"flows": flows, "heads": heads, "pump_status": pump_status, "slack": slack}


def _extract_energy(model: pyo.ConcreteModel) -> dict:
    dispatch, soc, line_flows, voltages = [], [], [], []

    for b in model.Buses:
        for t in model.T:
            dispatch.append({
                "bus": b, "time": t,
                "P_import": round(pyo.value(model.P_import[b, t]), 4),
                "P_export": round(pyo.value(model.P_export[b, t]), 4),
                "P_pv":     round(pyo.value(model.P_pv[b, t]), 4),
                "Q_ch":     round(pyo.value(model.Q_ch[b, t]), 4),
                "Q_dis":    round(pyo.value(model.Q_dis[b, t]), 4),
            })
            soc.append({
                "bus": b, "time": t,
                "E_soc": round(pyo.value(model.E_soc[b, t]), 4),
            })
            voltages.append({
                "bus": b, "time": t,
                "U": round(pyo.value(model.U[b, t]), 6),
                "theta": round(pyo.value(model.theta[b, t]), 6),
            })

    for l in model.ELines:
        for t in model.T:
            line_flows.append({
                "line": l, "time": t,
                "P_line": round(pyo.value(model.P_line[l, t]), 4),
                "Q_line": round(pyo.value(model.Q_line[l, t]), 4),
                "phi":    round(pyo.value(model.phi[l, t]), 6),
                "chi":    round(pyo.value(model.chi[l, t]), 6),
            })

    return {"dispatch": dispatch, "soc": soc,
            "line_flows": line_flows, "voltages": voltages}


def create_summary(run_id: str, model: pyo.ConcreteModel,
                   results: dict, solver_results) -> dict:
    """Compute summary metrics.

    Args:
        run_id:         Run directory name.
        model:          Solved ConcreteModel.
        results:        From extract_solution() -> water and energy.
        solver_results: Pyomo solver result object.

    Returns:
        Summary dict saved to summary.json.
    """

    logging.info("Writing solution summary...")

    n_vars = sum(1 for _ in model.component_data_objects(pyo.Var, active=True))
    n_cons = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True))

    summary = {
        "run_id": run_id,
        "solver_status": str(solver_results.solver.status),
        "termination_condition": str(solver_results.solver.termination_condition),
        "objective_value": results["objective"],
        "solver_time_s": getattr(solver_results.solver, "time", "N/A"),
        "num_variables": n_vars,
        "num_constraints": n_cons,
    }

    water = results.get("water", {})
    if water.get("slack"):
        summary["water_slack_violations"] = len(water["slack"])

    return summary


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_run_metadata(
    save_path: Path,
    metadata: dict,
    experiment_parameters: dict,
    network_files: dict[str, str],
    logger: logging.Logger,
    solver_log_path: str = None
    ) -> None:
    """Saves run environment and versioning, details."""
    try:
        start_time = datetime.datetime.fromisoformat(metadata["execution_start_time"])
        end_time = datetime.datetime.now()
        duration_min = (end_time - start_time).total_seconds() / 60.0

        metadata = {
            "experiment_id": save_path.parts[-1].split(" -- ")[-1],
            "execution_start_time": metadata["execution_start_time"],
            "execution_end_time": end_time.isoformat(),
            "execution_duration_min": round(duration_min, 2),
            "timestamp": end_time.isoformat(),
            "git_commit": utils.get_git_revision_hash(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "user": getpass.getuser(),
            "hostname": socket.gethostname(),
            "working_directory": os.getcwd(),
            "command": " ".join(sys.argv),
            "solver_log": str(solver_log_path) if solver_log_path else None,
        }

        # Save run metadata        
        meta_path = save_path / "00_run_metadata.yaml"
        with open(meta_path, "w") as f:
            yaml.dump(metadata, f)
        logging.info(f"Saved {meta_path.name} to {save_path.relative_to(save_path.parent)}")

        # Save experiment parameters
        param_path = save_path / "inputs" / "00_experiment_parameters.yaml"
        with open(param_path, "w") as f:
            yaml.dump(experiment_parameters, f)
        logging.info(f"Saved {param_path.name} to {save_path.relative_to(save_path.parent)}")

        # Save network files
        for network_name, network_data in network_files.items():
            network_file = network_data["inp_file"] if isinstance(network_data, dict) else network_data
            if not network_file:
                continue
            network_path = save_path / "inputs" / network_name / Path(network_file).name
            network_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(network_file, network_path)
            logging.info(f"Saved {network_name} network file - {Path(network_file).name} - to "
                         f"{network_path.relative_to(save_path.parent)}")

        # Save run and solver logs
        logging.info(f"Saved {Path(logger.handlers[-1].baseFilename).name} and {Path(solver_log_path).name} to "
                     f"{save_path.relative_to(save_path.parent)}")

        return metadata

    except Exception as e:
        logging.error(f"Failed to save run metadata, parameters & logs: {e}", exc_info=True)



def save_data(datasets: dict, save_path: Path) -> None:
    """Save datasets to CSV (.csv) or JSON (.json) under save_path."""
    
    for filename, data in datasets.items():
        if data is None:
            continue

        # Create full path
        full_path = save_path / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Save data
        if filename.endswith(".json") or filename.endswith(".yaml"):
            with open(full_path, "w") as f:
                json.dump(data, f, indent=2)


        elif filename.endswith(".csv"):
            data_df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            data_df.to_csv(full_path, index=False)


        else:
            logging.warning(f"Unsupported file type: {filename.split('/')[-1]}")
            continue

        logging.info(f"Saved {full_path.name} to {full_path.parent.relative_to(save_path)}")

