"""Notebook-compatible validation helpers backed by independent schedule replay.

New work should import ``src.validation``. The legacy runner names remain, but
now solve the actual model and replay its decisions; they do not fit hydraulic
constraints to the same simulation used as the comparison target.
"""
import contextlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import wntr

from src.algorithm_tasks import build_model, solve_model
from src.preprocessing import load_networks
from src.validation import replay_water, replay_energy

logger = logging.getLogger("econex.tests.validation")


@contextlib.contextmanager
def validation_logging(report_file: Path):
    report_file = Path(report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(report_file, mode="w")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)


def compare_series(opt_series, sim_series) -> Dict:
    """Compare complete, finite trajectories; never truncate a mismatch."""
    first = np.atleast_1d(opt_series).astype(float)
    second = np.atleast_1d(sim_series).astype(float)
    if first.shape != second.shape or not first.size:
        raise ValueError("Comparison series must have the same nonzero shape")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Comparison series must contain only finite values")
    error = np.abs(first - second)
    nonzero = np.abs(second) >= 1e-6
    relative = error[nonzero] / np.abs(second[nonzero])
    correlation = (float(np.corrcoef(first, second)[0, 1])
                   if np.std(first) > 1e-9 and np.std(second) > 1e-9
                   else float(np.allclose(first, second)))
    return {"n": int(first.size), "mae": float(error.mean()), "max_diff": float(error.max()),
            "max_rel_err_pct": float(relative.max() * 100) if relative.size else 0.0,
            "mean_rel_err_pct": float(relative.mean() * 100) if relative.size else 0.0,
            "correlation": correlation}


def extract_water_opt_timeseries(model, wn) -> Dict:
    times = list(model.T)
    index = [t * pyo.value(model.dt) for t in times]
    def frame(names, var):
        return pd.DataFrame({name: [pyo.value(var[name, t]) for t in times] for name in names}, index=index)
    return {"tank_heads": frame(wn.tank_name_list, model.H),
            "junction_heads": frame(wn.junction_name_list, model.H),
            "pump_flows": frame(wn.pump_name_list, model.Q),
            "pipe_flows": frame(wn.pipe_name_list, model.Q)}


def extract_water_sim_timeseries(wn, sim_results, num_steps) -> Dict:
    index = [t * wn.options.time.report_timestep for t in range(num_steps)]
    return {"tank_heads": sim_results.node["head"].loc[index, wn.tank_name_list],
            "junction_heads": sim_results.node["head"].loc[index, wn.junction_name_list],
            "pump_flows": sim_results.link["flowrate"].loc[index, wn.pump_name_list],
            "pipe_flows": sim_results.link["flowrate"].loc[index, wn.pipe_name_list]}


def extract_energy_opt_timeseries(model, buses, lines) -> Dict:
    times = list(model.T)
    index = [t * pyo.value(model.dt) for t in times]
    return {"voltage_pu": pd.DataFrame({b: [pyo.value(model.U[b, t]) for t in times]
                                       for b in buses}, index=index),
            "P_kw": pd.DataFrame({b: [pyo.value(model.P_import[b, t] - model.P_export[b, t])
                                       for t in times] for b in buses}, index=index)}


def compute_comparison_metrics(opt_data: Dict, sim_data: Dict) -> Dict:
    metrics = {}
    for name in sorted(opt_data.keys() & sim_data.keys()):
        opt, sim = opt_data[name], sim_data[name]
        if opt.empty or sim.empty:
            continue
        if not opt.index.equals(sim.index):
            raise ValueError(f"Time indices do not match for {name}")
        common = sorted(set(opt.columns) & set(sim.columns))
        metrics[name] = {column: compare_series(opt[column].values, sim[column].values)
                         for column in common}
    return metrics


def plot_validation_results(opt_data: Dict, sim_data: Dict, save_dir: Path) -> None:
    """Generate and save overlay plots for Pyomo vs. Simulation time series."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    dtypes = set(opt_data.keys()) | set(sim_data.keys())
    for dtype in dtypes:
        opt_df = opt_data.get(dtype, pd.DataFrame())
        sim_df = sim_data.get(dtype, pd.DataFrame())
        
        if opt_df.empty or sim_df.empty:
            continue
            
        common = sorted(list(set(opt_df.columns) & set(sim_df.columns)))
        if not common:
            continue
            
        dtype_dir = save_dir / dtype
        dtype_dir.mkdir(exist_ok=True)
        
        logger.info(f"Plotting {len(common)} items for {dtype}...")
        
        plot_items = common
        chunk_size = 12
        
        for i in range(0, len(plot_items), chunk_size):
            chunk = plot_items[i:i + chunk_size]
            n_items = len(chunk)
            
            cols = min(4, n_items)
            rows = (n_items + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            if n_items == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for idx, col in enumerate(chunk):
                ax = axes[idx]
                opt_series = opt_df[col].values
                sim_series = sim_df.loc[opt_df.index, col].values
                t_hours = np.asarray(opt_df.index, dtype=float) / 3600
                
                ax.plot(t_hours, sim_series, label='Nonlinear simulation', linestyle='--', marker='o', alpha=0.7)
                ax.plot(t_hours, opt_series, label='Pyomo Optimization', linestyle='-', alpha=0.7)
                
                ax.set_title(f"{dtype.replace('_', ' ').title()} - {col}")
                ax.set_xlabel("Time Step (Hour)")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            for idx in range(n_items, len(axes)):
                axes[idx].set_visible(False)
                
            plt.tight_layout()
            part_suffix = f"_part_{i // chunk_size + 1}" if len(plot_items) > chunk_size else ""
            plt.savefig(dtype_dir / f"plot{part_suffix}.png", dpi=150)
            plt.close()


def run_energy_validation(dss_file: str, num_timesteps: int = 24,
                          solver: str = "highs", save_dir: Path = None) -> dict:
    """Solve the energy-only model and independently replay AC bus injections."""
    config = {"run_water": False, "run_energy": True, "run_nexus": False,
              "T": num_timesteps, "energy": {"network": str(Path(dss_file).resolve())}}
    data = load_networks(config)
    model, result, _ = solve_model(build_model(data, config), solver=solver, timeout=120)
    with tempfile.TemporaryDirectory(prefix="econex-validation-") as scratch:
        output = Path(save_dir or scratch)
        replay = replay_energy(model, data["energy"], output)
        frame = pd.read_csv(output / "energy_replay_voltages.csv")
        frame["time"] *= 3600
        sim_data = {"voltage_pu": frame.pivot(index="time", columns="bus", values="voltage_pu")}
    opt_data = extract_energy_opt_timeseries(model, data["energy"]["buses"], data["energy"]["lines"])
    metrics = compute_comparison_metrics(opt_data, sim_data)
    if save_dir:
        Path(save_dir, "comparison_metrics.json").write_text(json.dumps(metrics, indent=2))
        plot_validation_results(opt_data, sim_data, Path(save_dir))
    return {"status": "completed", "termination": str(result.solver.termination_condition),
            "metrics": metrics, "replay_metrics": replay, "opt_data": opt_data, "sim_data": sim_data}


def run_water_validation(inp_file: str, num_timesteps: int = 24,
                         solver: str = "highs", save_dir: Path = None) -> dict:
    """Solve the water-only model and independently replay optimized pump status.

    Legacy timeseries compare interval starts; ``replay_metrics`` additionally
    includes terminal tank states and pressure/status feasibility observations.
    """
    config = {"run_water": True, "run_energy": False, "run_nexus": False,
              "T": num_timesteps, "water": {"network": str(Path(inp_file).resolve())}}
    data = load_networks(config)
    model, result, _ = solve_model(build_model(data, config), solver=solver, timeout=120)
    wn = wntr.network.WaterNetworkModel(data["water"]["inp_file"])
    opt_data = extract_water_opt_timeseries(model, wn)
    with tempfile.TemporaryDirectory(prefix="econex-validation-") as scratch:
        output = Path(save_dir or scratch)
        replay = replay_water(model, data["water"]["inp_file"], config, output)
        heads = pd.read_csv(output / "water_replay_heads.csv", index_col="time_seconds")
        flows = pd.read_csv(output / "water_replay_flowrate.csv", index_col="time_seconds")
        index = [t * 3600 for t in model.T]
        sim_data = {"tank_heads": heads.loc[index, wn.tank_name_list],
                    "junction_heads": heads.loc[index, wn.junction_name_list],
                    "pump_flows": flows.loc[index, wn.pump_name_list],
                    "pipe_flows": flows.loc[index, wn.pipe_name_list]}
    metrics = compute_comparison_metrics(opt_data, sim_data)
    if save_dir:
        Path(save_dir, "comparison_metrics.json").write_text(json.dumps(metrics, indent=2))
        plot_validation_results(opt_data, sim_data, Path(save_dir))
    return {"status": "completed", "termination": str(result.solver.termination_condition),
            "metrics": metrics, "replay_metrics": replay, "opt_data": opt_data, "sim_data": sim_data,
            "timestep": 3600}
