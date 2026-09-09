"""Independent nonlinear replay of optimized hourly schedules.

Replay measures approximation error. A feasible MILP alone is not a certificate
that the nonlinear hydraulic and AC networks satisfy the same constraints.
"""
from pathlib import Path
import os
import tempfile

import numpy as np
import opendssdirect as dss_module
import pyomo.environ as pyo
import wntr


def algebraic_residuals(model):
    """Maximum constraint, variable-bound, and integer violations (native units)."""
    constraints = 0.0
    bounds = 0.0
    integers = 0.0
    for con in model.component_data_objects(pyo.Constraint, active=True):
        value = pyo.value(con.body)
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite constraint value: {con.name}")
        if con.has_lb():
            constraints = max(constraints, pyo.value(con.lower) - value)
        if con.has_ub():
            constraints = max(constraints, value - pyo.value(con.upper))
    for var in model.component_data_objects(pyo.Var, active=True):
        if var.value is None:
            continue  # Some inactive auxiliaries can be omitted by the solver.
        value = pyo.value(var)
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite variable value: {var.name}")
        if var.lb is not None:
            bounds = max(bounds, var.lb - value)
        if var.ub is not None:
            bounds = max(bounds, value - var.ub)
        if var.is_integer():
            integers = max(integers, abs(value - round(value)))
    return {"max_constraint_violation": float(constraints),
            "max_bound_violation": float(bounds),
            "max_integrality_violation": float(integers)}


def replay_water(model, inp_file, config, output_dir):
    """Apply hourly pump decisions in EPANET and compare flows and tank states."""
    horizon = len(model.T)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    wn = wntr.network.WaterNetworkModel(str(inp_file))
    for name in list(wn.control_name_list):
        wn.remove_control(name)
    wn.options.time.duration = horizon * 3600
    wn.options.time.hydraulic_timestep = 300
    wn.options.time.report_timestep = 3600
    controls = wntr.network.controls
    for pump in model.Pumps:
        for t in model.T:
            status = wntr.network.LinkStatus.Open if pyo.value(model.Status[pump, t]) > 0.5 else wntr.network.LinkStatus.Closed
            condition = controls.SimTimeCondition(wn, "=", int(t) * 3600)
            action = controls.ControlAction(wn.get_link(pump), "status", status)
            wn.add_control(f"schedule_{pump}_{t}", controls.Control(condition, action))
    with tempfile.TemporaryDirectory(prefix="econex-replay-") as scratch:
        result = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(scratch) / "replay"))
    expected_times = [int(t) * 3600 for t in model.T]
    if not set(expected_times + [horizon * 3600]).issubset(result.node["head"].index):
        raise RuntimeError("EPANET replay did not return the complete horizon.")
    heads = result.node["head"]
    flow = result.link["flowrate"]
    pressure = result.node["pressure"][wn.junction_name_list]
    for name, frame in (("head", heads), ("flow", flow), ("pressure", pressure)):
        if not np.isfinite(frame.to_numpy()).all():
            raise RuntimeError(f"EPANET replay returned nonfinite {name} values.")
    flow_errors = [float(flow.loc[int(t) * 3600, link]) - pyo.value(model.Q[link, t])
                   for link in model.Links for t in model.T]
    tank_errors = [float(heads.loc[int(t) * 3600, tank]) - pyo.value(model.H[tank, t])
                   for tank in model.Tanks for t in model.T]
    tank_errors += [float(heads.loc[horizon * 3600, tank]) - pyo.value(model.H_terminal[tank])
                    for tank in model.Tanks]
    status_mismatches = sum(
        int((result.link["status"].loc[int(t) * 3600, pump] > 0.5) !=
            (pyo.value(model.Status[pump, t]) > 0.5))
        for pump in model.Pumps for t in model.T
    )
    threshold = config.get("water", {}).get("min_pressure_m", 0.0)
    tank_violations = 0
    final_changes = {}
    for tank in model.Tanks:
        node = wn.get_node(tank)
        levels = heads[tank] - node.elevation
        tank_violations += int(((levels < node.min_level - 1e-4) |
                                (levels > node.max_level + 1e-4)).sum())
        final_changes[tank] = float(heads.loc[horizon * 3600, tank] - heads.loc[0, tank])
    for name, frame in (("heads", heads), ("flowrate", flow), ("pressure", pressure),
                        ("pump_status", result.link["status"][wn.pump_name_list])):
        frame.to_csv(output / f"water_replay_{name}.csv", index_label="time_seconds")
    return {
        "hydraulic_timestep_seconds": 300,
        "report_timestep_seconds": 3600,
        "min_junction_pressure_m": float(pressure.min().min()),
        "pressure_violations_at_report_times": int((pressure < threshold - 1e-4).sum().sum()),
        "max_flow_error_m3s": float(np.max(np.abs(flow_errors))) if flow_errors else 0.0,
        "flow_rmse_m3s": float(np.sqrt(np.mean(np.square(flow_errors)))) if flow_errors else 0.0,
        "max_tank_head_error_m": float(np.max(np.abs(tank_errors))) if tank_errors else 0.0,
        "tank_bound_violations_at_report_times": tank_violations,
        "terminal_tank_change_m": final_changes,
        "pump_status_mismatches_at_report_times": status_mismatches,
    }


def replay_energy(model, energy_data, output_dir):
    """Run balanced AC snapshots for the scheduled net bus injections in OpenDSS."""
    import pandas as pd

    dss = dss_module.NewContext()
    cwd = os.getcwd()
    try:
        dss.Text.Command(f'Compile [{energy_data["dss_file"]}]')
    finally:
        os.chdir(cwd)
    for load in dss.Loads.AllNames():
        if load.lower() != "none":
            dss.Text.Command(f"Edit Load.{load} enabled=no")
    for index, bus in enumerate(model.Buses):
        dss.Circuit.SetActiveBus(bus)
        kv = dss.Bus.kVBase() * np.sqrt(3)
        dss.Text.Command(f"New Load.schedule{index} bus1={bus} phases=3 conn=wye kV={kv} kW=0 kvar=0 model=1 Vminpu=0 Vmaxpu=2")
    dss.Solution.Mode(0)
    dss.Solution.ControlMode(-1)
    rows = []
    converged = []
    import_errors = []
    max_loading = 0.0
    for t in model.T:
        for index, bus in enumerate(model.Buses):
            pump = pyo.value(model.PumpBusLoad[bus, t])
            net_kw = (energy_data["loads"][bus][t] + pump - pyo.value(model.P_pv[bus, t])
                      + pyo.value(model.Q_ch[bus, t]) - pyo.value(model.Q_dis[bus, t]))
            net_kvar = energy_data["reactive_loads"][bus][t]
            dss.Loads.Name(f"schedule{index}")
            dss.Loads.kW(net_kw)
            dss.Loads.kvar(net_kvar)
        dss.Solution.Solve()
        converged.append(bool(dss.Solution.Converged()))
        if not converged[-1]:
            raise RuntimeError(f"OpenDSS replay failed to converge at hour {t}.")
        for bus in model.Buses:
            dss.Circuit.SetActiveBus(bus)
            voltage = float(np.mean(dss.Bus.puVmagAngle()[::2]))
            if not np.isfinite(voltage):
                raise RuntimeError(f"OpenDSS returned a nonfinite voltage at {bus}, hour {t}.")
            rows.append({"bus": bus, "time": int(t), "voltage_pu": voltage,
                         "optimized_voltage_pu": pyo.value(model.U[bus, t])})
        grid_kw = -float(dss.Circuit.TotalPower()[0])
        if not np.isfinite(grid_kw):
            raise RuntimeError(f"OpenDSS returned nonfinite grid power at hour {t}.")
        scheduled_kw = sum(pyo.value(model.P_import[b, t] - model.P_export[b, t]) for b in model.Buses)
        import_errors.append(grid_kw - scheduled_kw)
        active = dss.Lines.First()
        while active:
            currents = dss.CktElement.CurrentsMagAng()[::2]
            if not np.isfinite(currents).all():
                raise RuntimeError(f"OpenDSS returned a nonfinite line current at hour {t}.")
            amps = max(currents)
            max_loading = max(max_loading, 100 * amps / dss.Lines.NormAmps())
            active = dss.Lines.Next()
    frame = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "energy_replay_voltages.csv", index=False)
    tolerance = energy_data["network"]["voltage_tolerance"]
    return {
        "all_snapshots_converged": all(converged),
        "min_voltage_pu": float(frame.voltage_pu.min()),
        "max_voltage_pu": float(frame.voltage_pu.max()),
        "max_voltage_error_pu": float((frame.voltage_pu - frame.optimized_voltage_pu).abs().max()),
        "voltage_violations": int(((frame.voltage_pu < 1 - tolerance) |
                                  (frame.voltage_pu > 1 + tolerance)).sum()),
        "max_line_loading_percent": float(max_loading),
        "max_grid_import_error_kw": float(max(abs(v) for v in import_errors)),
    }
