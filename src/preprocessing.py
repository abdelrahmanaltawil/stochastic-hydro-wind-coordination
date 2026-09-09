"""Load reproducible hydraulic and balanced electrical model inputs."""

import logging
import math
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import opendssdirect as dss
import wntr

from .helpers.energy.energy_simulation import load_energy_network


def _profile(values, T, name, default=0.0, minimum=None, maximum=None):
    """Expand a scalar, an exact horizon, or a repeating 24-hour profile."""
    if values is None or (isinstance(values, (list, tuple)) and not values):
        values = [default] * T
    elif isinstance(values, (int, float)) and not isinstance(values, bool):
        values = [values] * T
    elif len(values) == 24:
        values = [values[t % 24] for t in range(T)]
    elif len(values) != T:
        raise ValueError(f"{name} must contain {T} values or a repeating 24-hour profile")
    try:
        result = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if any(not math.isfinite(v) or (minimum is not None and v < minimum)
           or (maximum is not None and v > maximum) for v in result):
        raise ValueError(f"{name} contains non-finite or out-of-range values")
    return result


def load_networks(config: dict, project_root: Path | None = None) -> dict:
    """Resolve files and load data; nexus mode implies both networks.

    Relative paths are resolved against ``project_root`` when supplied and
    otherwise against the current working directory. All returned paths are
    absolute. ``T`` counts one-hour operating intervals, starting at index zero.
    """
    T = config.get("T", 24)
    if isinstance(T, bool) or not isinstance(T, int) or T < 1:
        raise ValueError("T must be a positive integer number of hourly intervals")
    root = Path(project_root or Path.cwd()).resolve()
    data = {}
    for domain in ("water", "energy"):
        if not (config.get(f"run_{domain}", domain == "water") or config.get("run_nexus", False)):
            continue
        section = config.get(domain, {})
        if not section.get("network"):
            raise ValueError(f"{domain}.network is required")
        path = Path(section["network"]).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{domain.title()} network not found: {path}")
        if domain == "water":
            data[domain] = {"inp_file": str(path), "config": dict(section)}
            sim = _run_epanet_presim(str(path), T)
            if sim is not None:
                data[domain]["epanet_sim"] = sim
        else:
            data[domain] = _build_energy_data(section, path, T)
        logging.info("Loaded %s network: %s", domain, path)
    if not data:
        raise ValueError("Enable at least one of run_water, run_energy, or run_nexus")
    return data


def _run_epanet_presim(inp_file: str, T: int) -> dict | None:
    """Simulate once, sharing operating points without leaving EPANET files.

    Link tables contain exactly T rows at 0, 3600, ..., (T-1)*3600 seconds.
    Failure is reported and returns None, allowing the model's explicit fallback.
    """
    cwd = os.getcwd()
    try:
        wn = wntr.network.WaterNetworkModel(inp_file)
        wn.options.time.duration = (T - 1) * 3600
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.report_timestep = 3600
        wn.options.time.report_start = 0
        with tempfile.TemporaryDirectory(prefix="econex-epanet-") as directory:
            results = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(directory) / "network"))
        times = [t * 3600 for t in range(T)]
        sim = {name: results.link[name].loc[times].reset_index(drop=True)
               for name in ("flowrate", "headloss", "status") if name in results.link}
        sim["head"] = results.node["head"].loc[times].reset_index(drop=True)
        logging.info("EPANET pre-simulation: %d links over %d h", sim["flowrate"].shape[1], T)
        return sim
    except Exception as exc:
        logging.warning("EPANET pre-simulation failed on %s (%s: %s)",
                        Path(inp_file).name, type(exc).__name__, exc)
        return None
    finally:
        os.chdir(cwd)


def _build_energy_data(energy_cfg: dict, dss_path: Path, T: int) -> dict:
    """Read a balanced three-phase line feeder and nominal hourly DSS loads.

    Power is in kW/kvar. Positive-sequence line impedances and ampacities are
    converted to per-unit using the circuit voltage and configured kVA base.
    The configured PV and battery each describe one installation, at their
    optional ``bus`` or the last bus with a load. Unsupported equipment is
    rejected rather than silently omitted from the electrical model.
    """
    dss_path = Path(dss_path).resolve()
    load_energy_network(str(dss_path))
    buses = list(dss.Circuit.AllBusNames())
    if not buses:
        raise ValueError(f"OpenDSS circuit has no buses: {dss_path}")
    for equipment, count in (("transformers", dss.Transformers.Count()),
                             ("generators", dss.Generators.Count()),
                             ("PV systems", dss.PVsystems.Count()),
                             ("storage", dss.Storages.Count()),
                             ("capacitors", dss.Capacitors.Count())):
        if count:
            raise ValueError(f"DSS {equipment} are not supported by the balanced line-feeder model")
    if dss.Vsources.Count() != 1:
        raise ValueError("The balanced feeder requires exactly one voltage source")
    dss.Vsources.First()
    reference = dss.CktElement.BusNames()[0].split(".")[0].lower()
    dss.Circuit.SetActiveBus(reference)
    voltage_kv = dss.Bus.kVBase() * math.sqrt(3)
    base_power = float(energy_cfg.get("base_power", 1000.0))
    if not math.isfinite(base_power) or base_power <= 0 or voltage_kv <= 0:
        raise ValueError("Electrical voltage and power bases must be positive")
    z_base = voltage_kv ** 2 * 1000.0 / base_power
    i_base = base_power / (math.sqrt(3) * voltage_kv)
    lines, line_names = [], []
    active = dss.Lines.First()
    while active:
        if dss.CktElement.NumPhases() != 3:
            raise ValueError("Only balanced three-phase lines are supported")
        n, m = (name.split(".")[0].lower() for name in (dss.Lines.Bus1(), dss.Lines.Bus2()))
        for bus in (n, m):
            dss.Circuit.SetActiveBus(bus)
            if not math.isclose(dss.Bus.kVBase() * math.sqrt(3), voltage_kv, rel_tol=1e-5):
                raise ValueError("The feeder must use one nominal voltage level")
        length = dss.Lines.Length()
        r, x = dss.Lines.R1() * length / z_base, dss.Lines.X1() * length / z_base
        limit = dss.Lines.NormAmps() / i_base
        if not all(math.isfinite(v) for v in (r, x, limit)) or r < 0 or abs(r) + abs(x) == 0 or limit <= 0:
            raise ValueError(f"Invalid impedance or ampacity for line {dss.Lines.Name()}")
        lines.append((n, m, r, x, limit))
        line_names.append(dss.Lines.Name())
        active = dss.Lines.Next()

    loads = {b: [0.0] * T for b in buses}
    reactive_loads = {b: [0.0] * T for b in buses}
    load_buses = []
    scale = float(energy_cfg.get("load_scale", 1.0))
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("energy.load_scale must be finite and nonnegative")
    active = dss.Loads.First()
    while active:
        if dss.CktElement.NumPhases() != 3:
            raise ValueError("Only balanced three-phase loads are supported")
        bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
        load_buses.append(bus)
        kw, kvar = dss.Loads.kW(), dss.Loads.kvar()
        pmult, qmult, actual = [1.0] * T, [1.0] * T, False
        shape = dss.Loads.Daily()
        if shape and shape.lower() != "none":
            dss.LoadShape.Name(shape)
            if not math.isclose(dss.LoadShape.HrInterval(), 1.0):
                raise ValueError(f"Loadshape {shape} must have a one-hour interval")
            p = dss.LoadShape.PMult()
            q = dss.LoadShape.QMult()
            if not p:
                raise ValueError(f"Loadshape {shape} is empty")
            pmult = [float(p[t % len(p)]) for t in range(T)]
            # DSS returns [0.0] when no Q multiplier array was specified.
            qmult = [float(q[t % len(q)]) for t in range(T)] if len(q) == len(p) else pmult
            actual = dss.LoadShape.UseActual()
        if energy_cfg.get("load_profile") is not None:
            pmult = _profile(energy_cfg["load_profile"], T, "energy.load_profile", default=1.0, minimum=0)
            qmult, actual = pmult, False
        for t in range(T):
            loads[bus][t] += scale * (pmult[t] if actual else kw * pmult[t])
            reactive_loads[bus][t] += scale * (qmult[t] if actual else kvar * qmult[t])
        active = dss.Loads.Next()
    if any(not math.isfinite(v) or v < 0 for series in loads.values() for v in series):
        raise ValueError("DSS active demand must be finite and nonnegative")

    tech = energy_cfg.get("technologies", {}) or {}
    pv = tech.get("pv", {}) or {}
    storage_cfg = tech.get("storage", {}) or {}
    default_bus = load_buses[-1] if load_buses else reference
    pv_bus = str(pv.get("bus", default_bus)).lower()
    storage_bus = str(storage_cfg.get("bus", default_bus)).lower()
    if pv_bus not in buses or storage_bus not in buses:
        raise ValueError("PV and storage bus names must exist in the DSS circuit")
    pv_capacity = float(pv.get("capacity_kw", 0.0))
    if not math.isfinite(pv_capacity) or pv_capacity < 0:
        raise ValueError("PV capacity must be finite and nonnegative")
    irradiance = _profile(pv.get("irradiance_profile"), T, "PV irradiance", minimum=0, maximum=1)
    pv_profile = {b: [pv_capacity * value if b == pv_bus else 0.0 for value in irradiance] for b in buses}
    storage = {
        "capacity_kwh": float(storage_cfg.get("capacity_kwh", 0.0)),
        "charge_efficiency": float(storage_cfg.get("charge_efficiency", 0.95)),
        "discharge_efficiency": float(storage_cfg.get("discharge_efficiency", 0.95)),
        "max_charge_kw": float(storage_cfg.get("max_charge_kw", 0.0)),
        "max_discharge_kw": float(storage_cfg.get("max_discharge_kw", 0.0)),
        "initial_soc_frac": float(storage_cfg.get("initial_soc_fraction", 0.5)),
    }
    for key, value in storage.items():
        upper = 1.0 if "efficiency" in key or key == "initial_soc_frac" else math.inf
        if not math.isfinite(value) or not 0 <= value <= upper or ("efficiency" in key and value == 0):
            raise ValueError(f"Invalid storage parameter {key}: {value}")
    storage_by_bus = {b: dict(storage) for b in buses}
    for b in buses:
        if b != storage_bus:
            storage_by_bus[b].update(capacity_kwh=0.0, max_charge_kw=0.0, max_discharge_kw=0.0)
    cost = energy_cfg.get("cost", {}) or {}
    tariff = _profile(cost.get("grid_import_tariff"), T, "Import tariff", default=0.1)
    export_tariff = _profile(cost.get("grid_export_tariff"), T, "Export tariff")
    network = {
        "nominal_voltage_pu": 1.0, "nominal_voltage_kv": voltage_kv,
        "base_power_kw": base_power, "base_current_amps": i_base,
        "reference_bus": reference, "grid_buses": [reference],
        "voltage_tolerance": energy_cfg.get("voltage_tolerance", 0.10),
        "n_current_segments": energy_cfg.get("n_current_segments", 5),
    }
    return {
        "dss_file": str(dss_path), "buses": buses, "lines": lines,
        "line_names": line_names, "loads": loads, "reactive_loads": reactive_loads,
        "pv_profile": pv_profile, "storage": storage, "storage_by_bus": storage_by_bus,
        "tariff": tariff, "export_tariff": export_tariff, "network": network,
        "config": energy_cfg,
    }


def build_network_data(config: dict, project_root: Path | None = None) -> dict:
    """Compatibility alias for :func:`load_networks`."""
    return load_networks(config, project_root=project_root)


def create_run_directory(base_path: str) -> Path:
    """Create a unique standalone run directory (legacy notebook API)."""
    directory = Path(base_path).resolve() / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    (directory / "metadata").mkdir(parents=True)
    return directory
