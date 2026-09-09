"""Energy system simulation helper (OpenDSS-based).

Used as a pre-optimization network exploration tool and as a post-optimization
validation tool to compare optimal dispatch against power-flow simulation.
Not a standalone workflow — call these functions from tests or workflow.py.
"""

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import opendssdirect as dss
import pandas as pd

logger = logging.getLogger(__name__)


def load_energy_network(dss_file: str) -> None:
    """Compile an OpenDSS circuit from a master .dss file.

    Uses opendssdirect (global engine state) to load the circuit.
    Must be called before any Solution or element queries.

    Args:
        dss_file: Absolute path to the OpenDSS master file.
    """
    if not dss_file:
        raise ValueError("No dss_file specified.")

    path = Path(dss_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OpenDSS network not found: {path}")
    logger.info("Compiling OpenDSS circuit: %s", path)
    cwd = os.getcwd()
    previous_allow = dss.Basic.AllowChangeDir()
    try:
        dss.Basic.AllowChangeDir(False)
        dss.Command(f"Compile [{path}]")
        if not dss.Basic.NumCircuits():
            raise ValueError(f"OpenDSS file did not define a circuit: {path}")
    finally:
        dss.Basic.AllowChangeDir(previous_allow)
        os.chdir(cwd)
    logger.debug(f"Circuit loaded: {dss.Circuit.Name()}")


def run_energy_simulation(dss_file: str, mode: str = "daily", num_steps: int = 24) -> dict:
    """Run a power-flow simulation for the compiled OpenDSS circuit.

    Args:
        dss_file:  Path to the OpenDSS master file.
        mode:      'snap' for a snapshot, or 'daily' for hourly solves.
        num_steps: Number of daily intervals (default 24). A daily solve samples
                   DSS hours 1 through num_steps: profile point 1 is interval 0.

    Returns:
        Dict with keys:
            'node': {'voltage_pu': {bus: [24 values]},
                     'P_kw':       {bus: [24 values]},
                     'Q_kvar':     {bus: [24 values]}}
            'link': {'current_amps': {branch: [24 values]},
                     'loading_pct':  {branch: [24 values]}}
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if mode not in ("snap", "daily"):
        raise ValueError(f"Unknown simulation mode: {mode!r}. Use 'snap' or 'daily'.")
    load_energy_network(dss_file)
    if mode == "snap":
        dss.Solution.Mode(0)
        steps = [None]
    elif mode == "daily":
        dss.Solution.Mode(2)        # Daily mode
        dss.Solution.Number(1)      # one step at a time
        dss.Solution.StepSize(3600) # 1-hour steps
        dss.Solution.Hour(0)
        dss.Solution.Seconds(0)
        steps = list(range(num_steps))

    node_voltage: dict[str, list] = {}
    node_P: dict[str, list] = {}
    node_Q: dict[str, list] = {}
    link_current: dict[str, list] = {}
    link_loading: dict[str, list] = {}
    losses_kw = []

    logger.info(f"Running energy simulation (mode={mode}, steps={len(steps)})")

    for step in steps:
        dss.Solution.Solve()
        if not dss.Solution.Converged():
            raise RuntimeError(f"OpenDSS power flow did not converge at interval {step}")
        losses_kw.append(float(dss.Circuit.Losses()[0] / 1000))

        # Bus voltages (per-unit, average of all phases)
        for bus in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus)
            pu_mags = dss.Bus.puVmagAngle()[0::2]   # magnitudes from interleaved [mag, ang, ...]
            avg_pu = float(sum(pu_mags) / len(pu_mags)) if pu_mags else 0.0
            node_voltage.setdefault(bus, []).append(avg_pu)

        # Device injections: positive means supply to the network; loads are
        # negative. Summing every branch terminal instead would return zero by
        # Kirchhoff's law and conceal the load being validated.
        buses = dss.Circuit.AllBusNames()
        injections = {bus: [0.0, 0.0] for bus in buses}
        for element in dss.Circuit.AllElementNames():
            if element.split(".", 1)[0].lower() not in {"load", "vsource", "generator", "storage", "pvsystem", "capacitor"}:
                continue
            dss.Circuit.SetActiveElement(element)
            if not dss.CktElement.Enabled():
                continue
            bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
            count = dss.CktElement.NumConductors()
            power = dss.CktElement.Powers()[:2 * count]
            injections[bus][0] -= sum(power[0::2])
            injections[bus][1] -= sum(power[1::2])
        for bus, (P, Q) in injections.items():
            node_P.setdefault(bus, []).append(float(P))
            node_Q.setdefault(bus, []).append(float(Q))

        # Branch currents and loading
        active = dss.Lines.First()
        while active:
            name = dss.Lines.Name()
            currents = dss.CktElement.CurrentsMagAng()
            I_mag = max(currents[0::2], default=0.0)
            norm_amps = dss.Lines.NormAmps()
            if norm_amps <= 0:
                raise ValueError(f"Line {name} has no positive normal ampacity")
            loading = float(I_mag / norm_amps * 100.0)
            link_current.setdefault(name, []).append(float(I_mag))
            link_loading.setdefault(name, []).append(loading)
            active = dss.Lines.Next()

    logger.info("Energy simulation completed")

    return {
        "time_hours": list(range(len(steps))),
        "losses_kw": losses_kw,
        "simulation_type": "native_dss_baseline",
        "node": {
            "voltage_pu": node_voltage,
            "P_kw": node_P,
            "Q_kvar": node_Q,
        },
        "link": {
            "current_amps": link_current,
            "loading_pct": link_loading,
        },
    }


def create_energy_summary(
    run_id: str,
    results: dict,
    voltage_tolerance: float = 0.10,
) -> dict:
    """Compute electrical performance metrics from simulation results.

    Args:
        run_id:            Run identifier.
        results:           From run_energy_simulation().
        voltage_tolerance: Acceptable deviation from 1.0 pu (default ±10%).

    Returns:
        Summary dict with voltage, current, and loss metrics.
    """
    voltages = results["node"]["voltage_pu"]
    all_voltages = [v for series in voltages.values() for v in series]
    if not all_voltages or any(not math.isfinite(v) for v in all_voltages):
        raise ValueError("Simulation contains no finite bus voltage series")

    V_min, V_max = min(all_voltages), max(all_voltages)
    V_mean = sum(all_voltages) / len(all_voltages)

    lo, hi = 1.0 - voltage_tolerance, 1.0 + voltage_tolerance
    violations = [v for v in all_voltages if v < lo or v > hi]

    loading = results["link"]["loading_pct"]
    all_loading = [v for series in loading.values() for v in series]
    max_loading = max(all_loading) if all_loading else 0.0
    overloaded = [v for v in all_loading if v > 100.0]

    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "voltage": {
                "min_pu": V_min,
                "max_pu": V_max,
                "mean_pu": V_mean,
                "num_violations": len(violations),
            },
            "line_loading": {
                "max_pct": max_loading,
                "num_overloaded": len(overloaded),
            },
            "losses": {"total_kwh": sum(results.get("losses_kw", []))},
        },
    }


def save_energy_results(results: dict, summary: dict, run_dir: Path) -> None:
    """Save energy simulation results to run_dir/energy/.

    Args:
        results:  From run_energy_simulation().
        summary:  From create_energy_summary().
        run_dir:  Top-level run directory.
    """
    energy_dir = run_dir / "energy"
    energy_dir.mkdir(parents=True, exist_ok=True)

    for group in ("node", "link"):
        for name, series in results[group].items():
            frame = pd.DataFrame(series, index=results.get("time_hours"))
            frame.index.name = "time_hours"
            frame.to_csv(energy_dir / f"{name}.csv")

    with open(energy_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"Energy simulation results saved to {energy_dir}")
