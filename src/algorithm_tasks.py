"""Algorithm tasks — model construction and solving.

build_model() assembles a shared Pyomo ConcreteModel by delegating to
private sub-model builders based on config flags:
  - config['run_water']  → _add_water_submodel()   (hydraulic scheduling)
  - config['run_energy'] → _add_energy_submodel()  (E1–E13)
  - config['run_nexus']  → _add_nexus_constraints() (fixed-power pump coupling)

solve_model() wraps Pyomo's SolverFactory with timeout and logging.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import wntr
from pyomo.opt import SolverFactory

from .helpers.water.hydraulic_utils import (
    calc_K,
    create_piecewise_pipe_curve,
    create_piecewise_pump_curve,
    add_pwl_constraint,
    pump_head_and_slope,
    pump_flow_at_head,
)
from .helpers.energy.power_utils import (
    calc_line_admittance,
    linearized_ac_coefficients,
    create_pwl_current_segments,
)


_MAX_FLOW = 5.0    # m³/s — loose upper bound for water big-M and variable bounds
_MAX_HEAD = 500.0  # metres — fallback ceiling when no network-derived bound exists


def _network_head_ceiling(wn: "wntr.network.WaterNetworkModel", T: int) -> float:
    """Valid upper bound on every nodal head over the horizon.

    Head can only be added by pumps and only decreases along the direction of
    flow, so no node can sit above the highest source (reservoir head or tank
    ceiling) plus the sum of the pump shutoff heads. The bound is used for the
    head variables and, following MILPNet, as the big-M of the pump and valve
    disjunctions; a tight M strengthens the LP relaxation of every solver.
    """
    sources = [float(wn.get_node(r).head_timeseries.at(t * 3600))
               for r in wn.reservoir_name_list for t in range(T)]
    sources += [float(wn.get_node(k).elevation + wn.get_node(k).max_level) for k in wn.tank_name_list]
    shutoff = 0.0
    for name, pump in wn.pumps():
        try:
            shutoff += max(0.0, float(pump_head_and_slope(pump, 0.0)[0]))
        except ValueError:
            shutoff += float(max(h for _, h in pump.get_pump_curve().points))
    if not sources:
        return _MAX_HEAD
    ceiling = 1.05 * (max(sources) + shutoff) + 1.0
    return float(min(max(ceiling, 1.0), _MAX_HEAD))


def _pump_price_table(spec, pumps, T: int, default: float = 0.1) -> dict:
    """Return {(pump, t): price} from a scalar, a horizon list, or a per-pump mapping.

    ``spec`` may be None (use ``default`` everywhere), a number, a list of T
    prices applied to every pump, or a mapping pump -> number or list of T
    prices. Prices are the electricity prices the water operator faces for
    its metered pump load in each interval (currency per kWh).
    """
    def horizon(values, label):
        if np.isscalar(values):
            return [float(values)] * T
        values = [float(v) for v in values]
        if len(values) != T or not all(np.isfinite(v) for v in values):
            raise ValueError(f"{label} must contain {T} finite hourly prices.")
        return values
    if spec is None:
        spec = default
    if isinstance(spec, dict):
        table = {}
        for p in pumps:
            if p not in spec:
                raise ValueError(f"pump_energy_price lacks an entry for pump {p!r}.")
            series = horizon(spec[p], f"pump_energy_price[{p}]")
            table.update({(p, t): series[t] for t in range(T)})
        return table
    series = horizon(spec, "pump_energy_price")
    return {(p, t): series[t] for p in pumps for t in range(T)}


def _linearize_hw(K: float, Q0: float, range_Q: float = None):
    """First-order Taylor expansion (or secant) of dH = sign(Q)·K·|Q|^1.852 around Q0.

    Returns (a, b) such that dH ≈ a + b·Q.
    If range_Q is provided, uses MILPNet's two_point_linear approach (secant line).
    At Q0=0 the true tangent is flat; it is only a local approximation.
    """
    if abs(Q0) < 1e-8:
        return 0.0, 0.0
    
    if range_Q is not None and range_Q > 0:
        # MILPNet two-point linear (secant)
        e1 = 1.852
        Q_1 = Q0 * (1 - range_Q)
        Q_2 = Q0 * (1 + range_Q)
        dH_1 = np.sign(Q_1) * K * abs(Q_1) ** e1
        dH_2 = np.sign(Q_2) * K * abs(Q_2) ** e1
        
        if abs(Q_2 - Q_1) < 1e-8:
            b = 1.852 * K * abs(Q0) ** 0.852
        else:
            b = (dH_2 - dH_1) / (Q_2 - Q_1)
        a = dH_2 - Q_2 * b
    else:
        # MILPNet one-point linear (tangent)
        h0 = np.sign(Q0) * K * abs(Q0) ** 1.852
        b = 1.852 * K * abs(Q0) ** 0.852  # Derivative is always positive
        a = h0 - b * Q0
        
    return float(a), float(b)


def _linearize_pump(pump, Q0: float):
    """First-order expansion of the EPANET fixed-speed pump curve around Q0.

    Returns (c, d) such that H ≈ c + d·Q.
    """
    head, slope = pump_head_and_slope(pump, Q0)
    return float(head - slope * Q0), float(slope)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(data: dict, config: dict) -> pyo.ConcreteModel:
    """Construct the shared Pyomo ConcreteModel.

    Args:
        data:   From preprocessing.load_networks() — keys 'water' and/or 'energy'.
        config: Unified configuration dict.

    Returns:
        ConcreteModel ready to pass to solve_model().
    """
    model = pyo.ConcreteModel(name="EcoNex_Optimization")

    T = config.get("T", 24)
    if isinstance(T, bool) or not isinstance(T, int) or T < 1:
        raise ValueError("T must be a positive integer number of hourly intervals.")
    model.T = pyo.RangeSet(0, T - 1, doc="Hourly time steps")
    model.StateT = pyo.RangeSet(0, T, doc="Interval boundary states, including terminal state")
    model.dt = pyo.Param(initialize=3600, doc="Step size [s]")

    run_water = config.get("run_water", True) or config.get("run_nexus", False)
    run_energy = config.get("run_energy", False) or config.get("run_nexus", False)

    if run_water:
        if "water" not in data:
            raise ValueError("run_water=true but no water data was loaded in preprocessing.")
        water_data = dict(data["water"])
        water_data["config"] = {**water_data.get("config", {}), **config.get("water", {})}
        _add_water_submodel(model, water_data)

    if run_energy:
        if "energy" not in data:
            raise ValueError("run_energy=true but no energy data was loaded in preprocessing.")
        _add_energy_submodel(model, data["energy"])

    if config.get("run_nexus", False):
        _add_nexus_constraints(model, data, config)

    _build_objective(model)

    logging.info("Model assembled successfully")
    return model


def solver_gap_options(solver: str, rel_gap: float = None, abs_gap: float = None) -> dict:
    """Solver-specific option names for the MIP optimality-gap tolerances."""
    options = {}
    if solver.startswith("gurobi"):
        if rel_gap is not None:
            options["MIPGap"] = rel_gap
        if abs_gap is not None:
            options["MIPGapAbs"] = abs_gap
    elif solver in ("highs", "appsi_highs"):
        if rel_gap is not None:
            options["mip_rel_gap"] = rel_gap
        if abs_gap is not None:
            options["mip_abs_gap"] = abs_gap
    elif solver == "cplex":
        if rel_gap is not None:
            options["mip_tolerances_mipgap"] = rel_gap
        if abs_gap is not None:
            options["mip_tolerances_absmipgap"] = abs_gap
    elif solver == "cbc":
        if rel_gap is not None:
            options["ratioGap"] = rel_gap
        if abs_gap is not None:
            options["allowableGap"] = abs_gap
    elif solver == "glpk":
        if rel_gap is not None:
            options["mipgap"] = rel_gap
    return options


def solver_default_options(solver: str) -> dict:
    """Solver options used for every study solve.

    HiGHS's default heuristic effort left several connection-rating instances
    of the integrated MILP without any incumbent for 600 s; raising it makes
    the solver find feasible schedules within seconds and prove optimality
    in about a minute. Other solvers need nothing.
    """
    if solver in ("highs", "appsi_highs"):
        return {"mip_heuristic_effort": 0.3}
    return {}


def solve_model(
    model: pyo.ConcreteModel,
    solver: str = "glpk",
    timeout: int = 300,
    logfile: str = None,
    tee: bool = False,
    options: dict = None,
) -> tuple:
    """Solve a linear mixed-integer model and return (model, results, logfile).

    Solutions are loaded only when a solver returns an incumbent. An infeasible
    or unbounded model is returned without manufacturing variable values.
    ``options`` are passed to the solver verbatim (see solver_gap_options()).
    """
    opt = SolverFactory(solver)
    if not opt.available(exception_flag=False):
        raise RuntimeError(f"Solver '{solver}' is not available.")
    if timeout <= 0:
        raise ValueError("Solver timeout must be positive.")
    if solver == "glpk":
        opt.options["tmlim"] = int(timeout)
        if logfile:
            opt.options["log"] = str(logfile)
    elif solver.startswith("gurobi"):
        opt.options["TimeLimit"] = timeout
        if logfile:
            opt.options["LogFile"] = str(logfile)
    elif solver == "cplex":
        opt.options["timelimit"] = timeout
    elif solver == "cbc":
        opt.options["seconds"] = timeout
    elif solver in ("highs", "appsi_highs"):
        opt.options["time_limit"] = timeout
        if logfile and solver == "highs":
            opt.options["log_file"] = str(logfile)
    for key, value in (options or {}).items():
        opt.options[key] = value
    logging.info("Solving with %s (timeout=%ss)", solver, timeout)
    results = opt.solve(model, tee=tee, load_solutions=False)
    if len(results.solution):
        model.solutions.load_from(results)
    logging.info("Solver finished: %s / %s", results.solver.status,
                 results.solver.termination_condition)
    return model, results, logfile


# ---------------------------------------------------------------------------
# Water sub-model  (hydraulic scheduling)
# ---------------------------------------------------------------------------

def _pipe_max_flows_from_sim(flowrate, pipe_names) -> dict:
    """Per-pipe PWL ceiling (2× peak |flow|) from a cached EPANET flowrate table.

    MILPNet approach: set each pipe's PWL domain to twice its maximum absolute
    flow so the segments cover the real operating range rather than a global
    worst-case ceiling. Pipes absent from the table fall back to _MAX_FLOW.
    """
    result = {}
    for pipe in pipe_names:
        if pipe in flowrate.columns:
            peak = float(flowrate[pipe].abs().max())
            result[pipe] = max(peak * 2.0, 0.01)
        else:
            result[pipe] = _MAX_FLOW
    return result


def _mean_pump_powers_from_sim(flowrate, headloss, pump_names, pump_efficiency: float = 1.0) -> dict:
    """Per-pump mean electrical power [kW] from cached EPANET link tables.

    Thomas & Sela (MILPNet) convention — take each pump's hydraulic power over
    the timesteps it actually runs and average it to one scalar:

        power      = flowrate · (−headloss)              # ΔH > 0 across a pump
        P_hyd [kW] = 9.81 · mean(power | power > 0)      # ρg·Q·ΔH, ρ = 1000
        P_elec     = P_hyd / η_pump

    Pumps that never run (or are missing from the tables) map to 0.0.
    """
    if not 0 < pump_efficiency <= 1:
        raise ValueError("Pump efficiency must lie in (0, 1].")
    powers = {}
    for p in pump_names:
        if p not in flowrate.columns or p not in headloss.columns:
            powers[p] = 0.0
            continue
        hyd = flowrate[p] * (-headloss[p])           # Q · ΔH  [m⁴/s]
        running = hyd[hyd > 1e-9]
        mean_qh = float(running.mean()) if len(running) else 0.0
        powers[p] = 9.81 * mean_qh / pump_efficiency
    return powers


def _epanet_pipe_max_flows(wn: "wntr.network.WaterNetworkModel", T: int) -> dict:
    """Fallback: simulate the network on demand and return 2× peak flow per pipe.

    Used only when no shared EPANET pre-simulation is cached on the data dict
    (see preprocessing._run_epanet_presim). Delegates the reduction to
    _pipe_max_flows_from_sim so cached and on-demand paths stay identical.

    Returns:
        Dict[pipe_name, float] — per-pipe PWL upper bound in m³/s, or {} on failure.
    """
    import copy
    import os
    wn_sim = copy.deepcopy(wn)
    wn_sim.options.time.duration = T * 3600
    wn_sim.options.time.hydraulic_timestep = 3600
    wn_sim.options.time.report_timestep = 3600
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="econex-pipe-bounds-") as sim_dir:
            res = wntr.sim.EpanetSimulator(wn_sim).run_sim(file_prefix=str(Path(sim_dir) / "network"))
        result = _pipe_max_flows_from_sim(res.link["flowrate"], wn.pipe_name_list)
        logging.info(f"EPANET on-demand sim: per-pipe max flows computed for {len(result)} pipes")
        return result
    except Exception as exc:
        logging.warning(f"EPANET on-demand sim failed ({exc}); using global max_flow fallback")
        return {}
    finally:
        os.chdir(cwd)


def _validate_water_network(wn) -> None:
    """Reject EPANET physics that the hourly hydraulic formulation omits.

    Pump status controls are intentionally replaced by optimization decisions.
    Zero-loss PRVs retain the provisional three-state approximation below;
    other controlled equipment requires an explicit extension of the model.
    """
    if str(wn.options.hydraulic.headloss).upper() not in ("H-W", "HW"):
        raise ValueError("The hydraulic optimization supports Hazen-Williams head loss only.")
    if str(wn.options.hydraulic.demand_model).upper() not in ("DD", "DDA"):
        raise ValueError("Pressure-dependent demand is unsupported; use demand-driven hydraulics.")
    for name, node in wn.junctions():
        if node.emitter_coefficient not in (None, 0):
            raise ValueError(f"Junction {name!r}: emitters are unsupported.")
    for name, node in wn.tanks():
        if node.vol_curve_name:
            raise ValueError(f"Tank {name!r}: volume curves are unsupported; use cylindrical tanks.")
        if node.overflow:
            raise ValueError(f"Tank {name!r}: overflow is unsupported.")
    for name, pipe in wn.pipes():
        if pipe.check_valve:
            raise ValueError(f"Pipe {name!r}: check valves are unsupported.")
        if pipe.initial_status == wntr.network.LinkStatus.Closed:
            raise ValueError(f"Pipe {name!r}: closed pipes are unsupported.")
        if pipe.minor_loss != 0:
            raise ValueError(f"Pipe {name!r}: minor-loss coefficients are unsupported.")
    for name, pump in wn.pumps():
        if pump.pump_type != "HEAD":
            raise ValueError(f"Pump {name!r}: only fixed-speed HEAD pump curves are supported.")
        if pump.base_speed != 1.0 or pump.speed_pattern_name:
            raise ValueError(f"Pump {name!r}: nonunit speed and speed patterns are unsupported.")
    for name, valve in wn.valves():
        if valve.valve_type != "PRV":
            raise ValueError(f"Valve {name!r}: only the provisional PRV approximation is supported.")
        if valve.minor_loss != 0:
            raise ValueError(f"Valve {name!r}: minor-loss coefficients are unsupported.")
        if valve.initial_status == wntr.network.LinkStatus.Closed:
            raise ValueError(f"Valve {name!r}: externally closed valves are unsupported.")
    for name, control in wn.controls():
        for action in control.actions():
            target, attribute = action.target()
            if target.name not in wn.pump_name_list or attribute != "status":
                raise ValueError(f"Control {name!r}: only pump-status controls may be replaced by scheduling.")


def _add_water_submodel(model: pyo.ConcreteModel, data: dict) -> None:
    """Add hydraulic variables, parameters, and constraints to model.

    Enforces junction continuity, head-flow approximations, and tank inventory.
    Registers model.water_cost (Expression) for the shared objective.

    Args:
        model: Shared ConcreteModel with model.T and model.dt already set.
        data:  {'inp_file': str, 'config': dict}
    """
    inp_file = data["inp_file"]
    logging.info(f"Building water sub-model from {inp_file}")
    wn = wntr.network.WaterNetworkModel(inp_file)
    water_cfg = data.get("config", {})
    _validate_water_network(wn)

    # Sets
    junctions  = wn.junction_name_list
    tanks      = wn.tank_name_list
    reservoirs = wn.reservoir_name_list
    nodes      = wn.node_name_list
    pipes      = wn.pipe_name_list
    pumps      = wn.pump_name_list
    valves     = wn.valve_name_list
    links      = wn.link_name_list

    model.Junctions  = pyo.Set(initialize=junctions)
    model.Tanks      = pyo.Set(initialize=tanks)
    model.Reservoirs = pyo.Set(initialize=reservoirs)
    model.Nodes      = pyo.Set(initialize=nodes)
    model.Pipes      = pyo.Set(initialize=pipes)
    model.Pumps      = pyo.Set(initialize=pumps)
    model.Valves     = pyo.Set(initialize=valves)
    model.Links      = pyo.Set(initialize=links)

    link_map = {name: (lnk.start_node_name, lnk.end_node_name)
                for name, lnk in wn.links()}

    # Parameters
    tank_areas     = {t: np.pi * (wn.get_node(t).diameter / 2) ** 2 for t in tanks}
    initial_levels = {t: wn.get_node(t).level + wn.get_node(t).elevation for t in tanks}
    min_levels     = {t: wn.get_node(t).min_level + wn.get_node(t).elevation for t in tanks}
    max_levels     = {t: wn.get_node(t).max_level + wn.get_node(t).elevation for t in tanks}

    T_val = len(list(model.T))
    # Sum every demand category and apply the EPANET global demand multiplier.
    base_demands = {
        j: [sum(ts.at(t * 3600) for ts in wn.get_node(j).demand_timeseries_list)
            * wn.options.hydraulic.demand_multiplier for t in range(T_val)]
        for j in junctions
    }
    pipe_Ks = {
        p: calc_K(wn.get_link(p).length, wn.get_link(p).diameter, wn.get_link(p).roughness)
        for p in pipes
    }

    # MILPNet approach: run EPANET first to get realistic per-pipe flow ceilings.
    # Using a global ceiling wastes all PWL segments on an irrelevant range when
    # actual flows are orders of magnitude smaller (e.g. 0.1 vs 5.0 m³/s).
    # EPANET operating-point data (per pipe/pump, per timestep) for linearization.
    # When present (validation mode), replaces binary SOS2 PWL with fast linear constraints.
    pipe_flows_sim = data.get("pipe_flows_sim")  # dict[pipe, list[float]]
    pump_flows_sim = data.get("pump_flows_sim")  # dict[pump, list[float]]

    # Per-pipe max flows for binary SOS2 PWL (non-validation mode). Prefer the
    # shared EPANET pre-simulation cached in preprocessing; only simulate here if
    # that cache is absent. (When pipe_flows_sim is set we use linearization and
    # these ceilings are not needed.)
    epanet_sim = data.get("epanet_sim")
    if pipe_flows_sim is None:
        if data.get("pipe_max_flows"):
            pipe_max_flows = data["pipe_max_flows"]
        elif epanet_sim is not None:
            pipe_max_flows = _pipe_max_flows_from_sim(epanet_sim["flowrate"], wn.pipe_name_list)
            logging.info("Water sub-model: pipe PWL bounds from shared EPANET pre-simulation")
        else:
            pipe_max_flows = _epanet_pipe_max_flows(wn, T_val)
    else:
        pipe_max_flows = {}  # not needed when using linearization

    model.TankArea = pyo.Param(model.Tanks, initialize=tank_areas)

    # Network-derived head ceiling: bounds the head variables and sets the
    # big-M of the pump/valve disjunctions (MILPNet: "just above the largest
    # bound of the associated continuous variable").
    head_ub = _network_head_ceiling(wn, T_val)
    min_pressure = float(water_cfg.get("min_pressure_m", 0.0))
    model.head_ceiling_m = head_ub

    # Variables
    if pipe_flows_sim:
        # Bound each pipe's flow to the direction and ~2× magnitude of the EPANET simulation.
        # Restrict tangent validation to the simulated flow direction; this is
        # a local approximation and is not used for general scheduling.
        def q_bounds(m, l, t):
            if l in m.Pipes:
                Q0 = pipe_flows_sim.get(l, [0.0] * T_val)[t]
                margin = max(abs(Q0) * 2.0, 0.005)
                if abs(Q0) < 1e-6:
                    return (-margin, margin)
                return (0.0, margin) if Q0 > 0 else (-margin, 0.0)
            return (0.0, _MAX_FLOW)  # pumps: non-negative
    else:
        def q_bounds(m, l, t):
            return (-_MAX_FLOW, _MAX_FLOW) if l in m.Pipes else (0, _MAX_FLOW)

    model.Q       = pyo.Var(model.Links, model.T, bounds=q_bounds, domain=pyo.Reals)
    model.H       = pyo.Var(model.Nodes, model.T, bounds=(0, head_ub), domain=pyo.NonNegativeReals)
    model.Status  = pyo.Var(model.Pumps, model.T, domain=pyo.Binary)
    model.SlackPos = pyo.Var(model.Junctions, model.T, domain=pyo.NonNegativeReals)
    model.SlackNeg = pyo.Var(model.Junctions, model.T, domain=pyo.NonNegativeReals)
    if not water_cfg.get("allow_demand_slack", False):
        model.SlackPos.fix(0.0)
        model.SlackNeg.fix(0.0)
    model.MinimumPressure = pyo.Constraint(
        model.Junctions, model.T,
        rule=lambda m, n, t: m.H[n, t] >= wn.get_node(n).elevation + min_pressure,
    )
    model.TankHeadBounds = pyo.Constraint(
        model.Tanks, model.T,
        rule=lambda m, n, t: (min_levels[n], m.H[n, t], max_levels[n]),
    )

    # W1 — Mass balance
    def mass_balance_rule(m, n, t):
        inflow  = sum(m.Q[l, t] for l in m.Links if link_map[l][1] == n)
        outflow = sum(m.Q[l, t] for l in m.Links if link_map[l][0] == n)
        return inflow - outflow + m.SlackPos[n, t] - m.SlackNeg[n, t] == base_demands[n][t]

    model.MassBalance = pyo.Constraint(model.Junctions, model.T, rule=mass_balance_rule)

    # W4 — Tank dynamics
    def tank_dynamics_rule(m, n, t):
        if t == 0:
            return m.H[n, t] == initial_levels[n]
        inflow  = sum(m.Q[l, t - 1] for l in m.Links if link_map[l][1] == n)
        outflow = sum(m.Q[l, t - 1] for l in m.Links if link_map[l][0] == n)
        return m.H[n, t] == m.H[n, t - 1] + (m.dt / m.TankArea[n]) * (inflow - outflow)

    model.TankDynamics = pyo.Constraint(model.Tanks, model.T, rule=tank_dynamics_rule)
    
    # Account for the last interval explicitly. Bounds alone allow filling at
    # minimum level and discharging at maximum level; no spurious pipe closure.
    model.H_terminal = pyo.Var(
        model.Tanks, bounds=lambda m, n: (min_levels[n], max_levels[n]),
    )
    def terminal_tank_rule(m, n):
        t = m.T.last()
        net = sum(m.Q[l, t] for l in m.Links if link_map[l][1] == n) - sum(
            m.Q[l, t] for l in m.Links if link_map[l][0] == n)
        return m.H_terminal[n] == m.H[n, t] + m.dt / m.TankArea[n] * net
    model.TankTerminalDynamics = pyo.Constraint(model.Tanks, rule=terminal_tank_rule)
    if water_cfg.get("terminal_tank_closure", not bool(pipe_flows_sim)):
        model.TankClosure = pyo.Constraint(
            model.Tanks, rule=lambda m, n: m.H_terminal[n] >= initial_levels[n],
        )

    # Reservoir heads (fixed)
    model.ResHead = pyo.Constraint(
        model.Reservoirs, model.T,
        rule=lambda m, n, t: m.H[n, t] == wn.get_node(n).head_timeseries.at(t * 3600)
    )

    # W1–W3 — Pipe head loss (Hazen-Williams)
    model.dH = pyo.Var(model.Pipes, model.T, domain=pyo.Reals)
    model.dH_def = pyo.Constraint(
        model.Pipes, model.T,
        rule=lambda m, p, t: m.dH[p, t] == m.H[link_map[p][0], t] - m.H[link_map[p][1], t]
    )
    t_list = list(model.T)
    if pipe_flows_sim:
        # MILPNet approach adapted: linearize H-W around EPANET operating point.
        # Replaces pipe-PWL binaries; pump/valve status variables remain binary.
        for p in pipes:
            K_p = pipe_Ks[p]
            flows_p = pipe_flows_sim.get(p, [0.0] * T_val)
            for t_idx, t in enumerate(t_list):
                a, b = _linearize_hw(K_p, flows_p[t_idx])
                model.add_component(
                    f"lin_hw_{p}_{t}",
                    pyo.Constraint(expr=model.dH[p, t] == a + b * model.Q[p, t])
                )
    else:
        # Binary SOS2 PWL fallback (general optimization, no prior simulation)
        for p in pipes:
            pwl_max_q = pipe_max_flows.get(p, _MAX_FLOW)
            pts = create_piecewise_pipe_curve(pipe_Ks[p], max_flow=pwl_max_q, num_segments=water_cfg.get("n_pipe_segments", 12))
            for t in model.T:
                add_pwl_constraint(model, f"pwl_pipe_{p}_{t}", model.Q[p, t], model.dH[p, t], pts)

    # W5–W7 — Pump head-flow curve + ON/OFF coupling
    model.PumpHeadGain = pyo.Var(model.Pumps, model.T, domain=pyo.NonNegativeReals)
    pump_flow_ceilings = {}
    if pump_flows_sim:
        # Linearize pump curve around the EPANET operating point, retaining ON/OFF status.
        for p in pumps:
            pump_link = wn.get_link(p)
            flows_p = pump_flows_sim.get(p, [0.0] * T_val)
            for t_idx, t in enumerate(t_list):
                c, d = _linearize_pump(pump_link, flows_p[t_idx])
                model.add_component(
                    f"lin_pump_{p}_{t}",
                    pyo.Constraint(expr=model.PumpHeadGain[p, t] == c * model.Status[p, t] + d * model.Q[p, t])
                )
    else:
        # Binary SOS2 PWL fallback. The curve is interpolated over the
        # hydraulically admissible flow range: a running pump must lift water
        # at least from the highest head its suction node can take to the
        # lowest head its discharge node may have (pressure floor or tank
        # bottom), so flows beyond the curve point at that minimum gain are
        # never feasible and would only weaken the LP relaxation.
        def node_head_floor(n):
            if n in junctions:
                return wn.get_node(n).elevation + min_pressure
            if n in tanks:
                return min_levels[n]
            return min(float(wn.get_node(n).head_timeseries.at(t * 3600)) for t in range(T_val))

        def node_head_ceiling(n):
            if n in tanks:
                return max_levels[n]
            if n in reservoirs:
                return max(float(wn.get_node(n).head_timeseries.at(t * 3600)) for t in range(T_val))
            return head_ub

        for p in pumps:
            up, down = link_map[p]
            min_gain = max(0.0, node_head_floor(down) - node_head_ceiling(up))
            pump_link = wn.get_link(p)
            ceiling = pump_flow_at_head(pump_link, min_gain) if min_gain > 0 else None
            pts = create_piecewise_pump_curve(pump_link, num_segments=6, max_flow=ceiling)
            pump_flow_ceilings[p] = pts[-1][0]
            for t in model.T:
                add_pwl_constraint(model, f"pwl_pump_{p}_{t}", model.Q[p, t], model.PumpHeadGain[p, t], pts, activation=model.Status[p, t])

    model.pump_flow_ceiling_m3s = dict(pump_flow_ceilings)
    model.PumpStatusFlow = pyo.Constraint(
        model.Pumps, model.T,
        rule=lambda m, p, t: m.Q[p, t] <= pump_flow_ceilings.get(p, _MAX_FLOW) * m.Status[p, t]
    )

    # With zero head gain while OFF, head differences are bounded by the head ceiling.
    _M = head_ub
    model.PumpHeadCoup1 = pyo.Constraint(
        model.Pumps, model.T,
        rule=lambda m, p, t: -_M * (1 - m.Status[p, t]) <= m.H[link_map[p][1], t] - m.H[link_map[p][0], t] - m.PumpHeadGain[p, t]
    )
    model.PumpHeadCoup2 = pyo.Constraint(
        model.Pumps, model.T,
        rule=lambda m, p, t: m.H[link_map[p][1], t] - m.H[link_map[p][0], t] - m.PumpHeadGain[p, t] <= _M * (1 - m.Status[p, t])
    )

    # MILPNet Valve Logic (PRV 3-state and generic valves)
    # ----------------------------------------------------
    model.ValveV1 = pyo.Var(model.Valves, model.T, domain=pyo.Binary) # Active (PRV only)
    model.ValveV2 = pyo.Var(model.Valves, model.T, domain=pyo.Binary) # Open
    model.ValveV3 = pyo.Var(model.Valves, model.T, domain=pyo.Binary) # Closed

    model.ValveState = pyo.Constraint(
        model.Valves, model.T,
        rule=lambda m, v, t: m.ValveV1[v, t] + m.ValveV2[v, t] + m.ValveV3[v, t] == 1
    )

    _eps_flow = 0.0
    _eps_head = 0.0

    valve_settings = {}
    is_prv = {}
    for v in valves:
        valve_obj = wn.get_link(v)
        is_prv[v] = valve_obj.valve_type == 'PRV'
        if is_prv[v]:
            end_node = wn.get_node(valve_obj.end_node_name)
            valve_settings[v] = end_node.elevation + valve_obj.setting
        else:
            valve_settings[v] = 0.0
    # Heads lie in [0, head_ub], so head differences and setting offsets are
    # covered by head_ub plus the largest setting.
    _M_valve = head_ub + max(valve_settings.values(), default=0.0)

    model.ValveFlowLower = pyo.Constraint(
        model.Valves, model.T,
        rule=lambda m, v, t: m.Q[v, t] >= _eps_flow * (1 - m.ValveV3[v, t])
    )
    model.ValveFlowUpper = pyo.Constraint(
        model.Valves, model.T,
        rule=lambda m, v, t: m.Q[v, t] <= _MAX_FLOW * (1 - m.ValveV3[v, t])
    )

    model.ValveConstraints = pyo.ConstraintList()
    for v in valves:
        for t in model.T:
            start_n = link_map[v][0]
            end_n = link_map[v][1]
            
            if is_prv[v]:
                H_set = valve_settings[v]
                # V1=1 (Active): H_end == H_set, H_start >= H_set
                model.ValveConstraints.add(model.H[end_n, t] - H_set <= _M_valve * (1 - model.ValveV1[v, t]))
                model.ValveConstraints.add(H_set - model.H[end_n, t] <= _M_valve * (1 - model.ValveV1[v, t]))
                model.ValveConstraints.add(model.H[start_n, t] >= H_set - _M_valve * (1 - model.ValveV1[v, t]))
                
                # V2=1 (Open): H_start == H_end, H_start <= H_set
                model.ValveConstraints.add(model.H[start_n, t] - model.H[end_n, t] <= _M_valve * (1 - model.ValveV2[v, t]))
                model.ValveConstraints.add(model.H[end_n, t] - model.H[start_n, t] <= _M_valve * (1 - model.ValveV2[v, t]))
                model.ValveConstraints.add(model.H[start_n, t] <= H_set + _M_valve * (1 - model.ValveV2[v, t]))
                
                # V3=1 (Closed): H_start <= H_end (check valve logic)
                model.ValveConstraints.add(model.H[start_n, t] - model.H[end_n, t] + _eps_head * model.ValveV3[v, t] <= _M_valve * (1 - model.ValveV3[v, t]))
            else:
                # Generic Valve: V1 is not used.
                model.ValveConstraints.add(model.ValveV1[v, t] == 0)
                # V2=1 (Open): H_start == H_end
                model.ValveConstraints.add(model.H[start_n, t] - model.H[end_n, t] <= _M_valve * (1 - model.ValveV2[v, t]))
                model.ValveConstraints.add(model.H[end_n, t] - model.H[start_n, t] <= _M_valve * (1 - model.ValveV2[v, t]))

    # Fixed ON-power approximation, calibrated once from the shared simulation.
    powers = dict(data.get("pump_power_kw", {}))
    if not powers and pumps:
        powers = _pump_mean_powers(inp_file, T_val, pumps,
                                  float(water_cfg.get("pump_efficiency", 1.0)), epanet_sim)
    model.pump_mean_power_kw = powers
    # Electricity price the water operator faces for its metered pump load. A
    # scalar is the flat retail tariff of the stand-alone problem; a horizon
    # list or per-pump mapping is the time-varying (internal) price used by
    # the decentralized coordination scheme (see src/coordination.py).
    dt_h = float(pyo.value(model.dt)) / 3600.0
    price = _pump_price_table(
        water_cfg.get("pump_energy_price"), pumps, T_val,
        default=float(water_cfg.get("pump_energy_tariff", 0.1)),
    )
    model.pump_energy_price = price
    model.pump_energy_cost = pyo.Expression(rule=lambda m: dt_h * sum(
        powers[p] * price[p, t] * m.Status[p, t] for p in m.Pumps for t in m.T))
    # Optional level-holding tie-break (weight in currency per metre-hour):
    # among schedules of equal energy cost, prefer the one that keeps each
    # tank closest to its initial level. This is the discrete analogue of
    # "float" operation and makes the flat-tariff baseline reproducible; the
    # weight must be too small to trade against one pump-hour of energy.
    tracking_weight = float(water_cfg.get("level_tracking_weight", 0.0))
    if tracking_weight < 0:
        raise ValueError("level_tracking_weight must be nonnegative.")
    if tracking_weight > 0:
        model.TankDevPos = pyo.Var(model.Tanks, model.T, domain=pyo.NonNegativeReals)
        model.TankDevNeg = pyo.Var(model.Tanks, model.T, domain=pyo.NonNegativeReals)
        model.TankDeviation = pyo.Constraint(
            model.Tanks, model.T,
            rule=lambda m, n, t: m.H[n, t] - initial_levels[n] == m.TankDevPos[n, t] - m.TankDevNeg[n, t])
        model.level_tracking_cost = pyo.Expression(rule=lambda m: tracking_weight * sum(
            m.TankDevPos[n, t] + m.TankDevNeg[n, t] for n in m.Tanks for t in m.T))
    else:
        model.level_tracking_cost = pyo.Expression(expr=0.0)
    model.water_cost = pyo.Expression(rule=lambda m: (
        m.pump_energy_cost + m.level_tracking_cost
        + sum(1e9 * (m.SlackPos[n, t] + m.SlackNeg[n, t]) for n in m.Junctions for t in m.T)
    ))

    logging.info(f"Water sub-model: {len(nodes)} nodes, {len(links)} links, {len(pumps)} pumps "
                 f"(head ceiling {head_ub:.1f} m)")


# ---------------------------------------------------------------------------
# Energy sub-model  (E1–E13)
# ---------------------------------------------------------------------------

def _add_energy_submodel(model: pyo.ConcreteModel, data: dict) -> None:
    """Balanced, lossless linear AC feeder with batteries and PWL ampacity.

    Inputs: loads/PV/dispatch in kW, energy in kWh, reactive loads in kvar,
    impedances and currents in per unit on network.base_power_kw. Storage
    charge/discharge denote AC-terminal powers; efficiencies occur only in SoC.
    """
    buses = list(data["buses"])
    if not buses or len(set(buses)) != len(buses):
        raise ValueError("Energy data must contain distinct buses.")
    network = data.get("network", {})
    ref = network.get("reference_bus", buses[0])
    grid_buses = list(data.get("grid_buses", network.get("grid_buses", [ref])))
    if ref not in buses or not grid_buses or any(b not in buses for b in grid_buses):
        raise ValueError("Reference bus and grid buses must belong to the electrical network.")
    base_kw = float(network.get("base_power_kw", 1.0))
    U0 = float(network.get("nominal_voltage_pu", 1.0))
    v_tol = float(network.get("voltage_tolerance", 0.10))
    n_seg = int(network.get("n_current_segments", 5))
    if base_kw <= 0 or U0 <= 0 or not 0 < v_tol < 1 or n_seg < 1:
        raise ValueError("Invalid electrical base, voltage tolerance, or segment count.")
    times = list(model.T)
    dt_h = pyo.value(model.dt) / 3600.0
    def profile(values, label, default=0.0):
        if values is None:
            return [default] * len(times)
        if np.isscalar(values):
            values = [float(values)] * len(times)
        values = list(values)
        if len(values) != len(times) or not all(np.isfinite(v) for v in values):
            raise ValueError(f"{label} must have {len(times)} finite hourly values.")
        return values
    loads = {b: profile(data.get("loads", {}).get(b), f"Load at {b}") for b in buses}
    qloads = {b: profile(data.get("reactive_loads", {}).get(b), f"Reactive load at {b}") for b in buses}
    pv = {b: profile(data.get("pv_profile", {}).get(b), f"PV at {b}") for b in buses}
    if any(v < 0 for b in buses for v in loads[b] + pv[b]):
        raise ValueError("Active loads and PV availability must be nonnegative.")
    tariff = profile(data.get("tariff"), "Import tariff", 0.1)
    export_tariff = profile(data.get("export_tariff"), "Export tariff")
    zero_storage = {"capacity_kwh": 0., "max_charge_kw": 0., "max_discharge_kw": 0.,
                    "charge_efficiency": 1., "discharge_efficiency": 1., "initial_soc_frac": 0.}
    stor = {b: {**zero_storage, **data.get("storage", {}),
                **data.get("storage_by_bus", {}).get(b, {})} for b in buses}
    for b, spec in stor.items():
        if (any(not np.isfinite(float(v)) for v in spec.values())
            or any(spec[k] < 0 for k in ("capacity_kwh", "max_charge_kw", "max_discharge_kw"))
            or not 0 <= spec["initial_soc_frac"] <= 1
            or any(not 0 < spec[k] <= 1 for k in ("charge_efficiency", "discharge_efficiency"))):
            raise ValueError(f"Invalid battery parameters at {b}.")

    # Keep endpoints explicitly: bus identifiers may themselves contain '_'.
    line_params, endpoints = {}, {}
    for i, (n, m, R, X, I_max) in enumerate(data.get("lines", [])):
        if n not in buses or m not in buses or n == m:
            raise ValueError(f"Invalid electrical line endpoints {n!r}, {m!r}.")
        name = f"{n}_{m}"
        if name in line_params:
            name = f"{name}__{i}"
        line_params[name] = (float(R), float(X), float(I_max))
        endpoints[name] = (n, m)
    model.line_endpoints = endpoints
    model.reference_bus = ref
    model.base_power_kw = base_kw
    model.Buses = pyo.Set(initialize=buses)
    model.GridBuses = pyo.Set(initialize=grid_buses)
    model.ELines = pyo.Set(initialize=list(line_params))
    model.P_pv = pyo.Var(model.Buses, model.T, domain=pyo.NonNegativeReals)
    model.P_import = pyo.Var(model.Buses, model.T, domain=pyo.NonNegativeReals)
    model.P_export = pyo.Var(model.Buses, model.T, domain=pyo.NonNegativeReals)
    model.y_import = pyo.Var(model.Buses, model.T, domain=pyo.Binary)
    model.y_export = pyo.Var(model.Buses, model.T, domain=pyo.Binary)
    model.Q_ch = pyo.Var(model.Buses, model.T, domain=pyo.NonNegativeReals)
    model.Q_dis = pyo.Var(model.Buses, model.T, domain=pyo.NonNegativeReals)
    model.y_ch = pyo.Var(model.Buses, model.T, domain=pyo.Binary)
    model.y_dis = pyo.Var(model.Buses, model.T, domain=pyo.Binary)
    model.E_soc = pyo.Var(model.Buses, model.StateT, domain=pyo.NonNegativeReals,
                          bounds=lambda m, b, t: (0., stor[b]["capacity_kwh"]))
    model.P_line = pyo.Var(model.ELines, model.T, domain=pyo.Reals)
    model.Q_line = pyo.Var(model.ELines, model.T, domain=pyo.Reals)
    model.Q_grid = pyo.Var(model.Buses, model.T, domain=pyo.Reals)
    model.U = pyo.Var(model.Buses, model.T, bounds=(U0 * (1-v_tol), U0 * (1+v_tol)))
    model.theta = pyo.Var(model.Buses, model.T, domain=pyo.Reals)
    model.I_re = pyo.Var(model.ELines, model.T, domain=pyo.Reals)
    model.I_im = pyo.Var(model.ELines, model.T, domain=pyo.Reals)
    model.phi = pyo.Var(model.ELines, model.T, domain=pyo.NonNegativeReals)
    model.chi = pyo.Var(model.ELines, model.T, domain=pyo.NonNegativeReals)
    model.PumpBusLoad = pyo.Expression(model.Buses, model.T, rule=lambda m, b, t: 0.)
    model.GridImportLimit = pyo.Param(initialize=float(network.get("max_grid_import_kw", max(
        1., sum(max(loads[b]) + stor[b]["max_charge_kw"] for b in buses)))), mutable=True)
    model.GridExportLimit = pyo.Param(initialize=float(network.get("max_grid_export_kw", max(
        1., sum(max(pv[b]) + stor[b]["max_discharge_kw"] for b in buses)))), mutable=True)
    model.grid_import_limit_explicit = "max_grid_import_kw" in network
    for b in buses:
        if b not in grid_buses:
            for t in times:
                for var in (model.P_import, model.P_export, model.Q_grid, model.y_import, model.y_export):
                    var[b, t].fix(0)

    def net_in(m, b, t, variable):
        return sum(variable[l, t] for l, (n, k) in endpoints.items() if k == b) - sum(
            variable[l, t] for l, (n, k) in endpoints.items() if n == b)
    model.NodalPBalance = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t:
        m.P_import[b, t] - m.P_export[b, t] + m.P_pv[b, t] + m.Q_dis[b, t] - m.Q_ch[b, t]
        + net_in(m, b, t, m.P_line) == loads[b][t] + m.PumpBusLoad[b, t])
    model.NodalQBalance = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t:
        m.Q_grid[b, t] + net_in(m, b, t, m.Q_line) == qloads[b][t])
    def soc_rule(m, b, t):
        spec = stor[b]
        if t == 0:
            return m.E_soc[b, t] == spec["capacity_kwh"] * spec["initial_soc_frac"]
        return m.E_soc[b, t] == m.E_soc[b, t-1] + dt_h * (
            spec["charge_efficiency"] * m.Q_ch[b, t-1] - m.Q_dis[b, t-1] / spec["discharge_efficiency"])
    model.SoCContinuity = pyo.Constraint(model.Buses, model.StateT, rule=soc_rule)
    model.SoCClosure = pyo.Constraint(model.Buses, rule=lambda m, b:
        m.E_soc[b, m.StateT.last()] >= m.E_soc[b, 0])
    model.ChargeBound = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t:
        m.Q_ch[b, t] <= stor[b]["max_charge_kw"] * m.y_ch[b, t])
    model.DischargeBound = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t:
        m.Q_dis[b, t] <= stor[b]["max_discharge_kw"] * m.y_dis[b, t])
    model.ChargeDischExcl = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t: m.y_ch[b, t] + m.y_dis[b, t] <= 1)
    model.ImportBound = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t: m.P_import[b, t] <= m.GridImportLimit * m.y_import[b, t])
    model.ExportBound = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t: m.P_export[b, t] <= m.GridExportLimit * m.y_export[b, t])
    model.ImportExportExcl = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t: m.y_import[b, t] + m.y_export[b, t] <= 1)
    model.PVCap = pyo.Constraint(model.Buses, model.T, rule=lambda m, b, t: m.P_pv[b, t] <= pv[b][t])
    model.RefAngle = pyo.Constraint(model.T, rule=lambda m, t: m.theta[ref, t] == 0)
    model.RefVoltage = pyo.Constraint(model.T, rule=lambda m, t: m.U[ref, t] == U0)
    coeffs = {l: linearized_ac_coefficients(*calc_line_admittance(R, X), U0)
              for l, (R, X, _) in line_params.items()}
    def delta(m, l, t, variable):
        n, k = endpoints[l]
        return variable[n, t] - variable[k, t]
    model.PLineFlow = pyo.Constraint(model.ELines, model.T, rule=lambda m, l, t:
        m.P_line[l, t] == base_kw * (coeffs[l]["P_dU"] * delta(m, l, t, m.U)
                                     + coeffs[l]["P_dtheta"] * delta(m, l, t, m.theta)))
    model.QLineFlow = pyo.Constraint(model.ELines, model.T, rule=lambda m, l, t:
        m.Q_line[l, t] == base_kw * (coeffs[l]["Q_dU"] * delta(m, l, t, m.U)
                                     + coeffs[l]["Q_dtheta"] * delta(m, l, t, m.theta)))
    # At the flat-voltage expansion point S_pu = U0 * conjugate(I_pu).
    model.IReRule = pyo.Constraint(model.ELines, model.T, rule=lambda m, l, t: m.I_re[l, t] == m.P_line[l, t] / (base_kw * U0))
    model.IImRule = pyo.Constraint(model.ELines, model.T, rule=lambda m, l, t: m.I_im[l, t] == -m.Q_line[l, t] / (base_kw * U0))
    model.PhiPos = pyo.Constraint(model.ELines, model.T, rule=lambda m, l, t: m.phi[l, t] >= m.I_re[l, t])
    model.PhiNeg = pyo.Constraint(model.ELines, model.T, rule=lambda m, l, t: m.phi[l, t] >= -m.I_re[l, t])
    model.ChiPos = pyo.Constraint(model.ELines, model.T, rule=lambda m, l, t: m.chi[l, t] >= m.I_im[l, t])
    model.ChiNeg = pyo.Constraint(model.ELines, model.T, rule=lambda m, l, t: m.chi[l, t] >= -m.I_im[l, t])
    for lname, (_, _, i_max) in line_params.items():
        # Nameplate limits may be orders of magnitude above feeder load. Keep
        # secants useful near operating currents by allowing user-supplied knots.
        pts = create_pwl_current_segments(i_max, n_seg)
        for t in times:
            square_vars = []
            for component in ("phi", "chi"):
                square = pyo.Var(domain=pyo.NonNegativeReals)
                model.add_component(f"{component}_sq_{lname}_{t}", square)
                add_pwl_constraint(model, f"pwl_{component}_{lname}_{t}", getattr(model, component)[lname, t], square, pts)
                square_vars.append(square)
            model.add_component(f"thermal_{lname}_{t}", pyo.Constraint(expr=sum(square_vars) <= i_max ** 2))
    model.energy_cost = pyo.Expression(expr=dt_h * sum(
        tariff[t] * model.P_import[b, t] - export_tariff[t] * model.P_export[b, t]
        for b in buses for t in times))
    # Day-ahead prices and the metered-load hook, kept for the coordination layer.
    model.import_price = list(tariff)
    model.export_price = list(export_tariff)
    model.pump_bus_capacity_kw = {b: 0.0 for b in buses}
    logging.info("Energy sub-model: %d buses, %d lines", len(buses), len(line_params))

# ---------------------------------------------------------------------------
# Nexus coupling
# ---------------------------------------------------------------------------

PUMP_POWER_RESOLUTION_KW = 0.01  # calibrated on-state powers are rounded to 10 W


def _pump_mean_powers(
    inp_file: str,
    T: int,
    pump_names,
    pump_efficiency: float = 1.0,
    epanet_sim: dict = None,
    resolution_kw: float = PUMP_POWER_RESOLUTION_KW,
) -> dict:
    """Per-pump mean electrical power [kW], reusing the shared EPANET sim if present.

    Prefers the pre-simulation cached on data['water']['epanet_sim'] (run once in
    preprocessing). Only when that cache is absent — or lacks headloss — does it
    simulate the network here. Either way the reduction is the Thomas & Sela
    (MILPNet) convention implemented in _mean_pump_powers_from_sim(). The
    result is rounded to ``resolution_kw`` (10 W by default): the calibration
    is not more accurate than that, and rounded coefficients let MILP solvers
    recognise the integrality of the pump-energy objective, which shortens the
    proof of optimality considerably.

    Args:
        inp_file:         EPANET .inp path (used only on the fallback sim path).
        T:                Horizon length in hours.
        pump_names:       Pump identifiers to price (from model.Pumps).
        pump_efficiency:  Wire-to-water efficiency η (default 1.0, MILPNet).
        epanet_sim:       Cached {'flowrate', 'headloss'} tables, or None.

    Returns:
        Dict[pump_name, float] — mean electrical power in kW (0.0 if never running).
    """
    pump_names = list(pump_names)
    if not pump_names:
        return {}

    def rounded(powers):
        if resolution_kw and resolution_kw > 0:
            return {p: round(v / resolution_kw) * resolution_kw for p, v in powers.items()}
        return powers

    if epanet_sim is not None and "headloss" in epanet_sim:
        logging.info("Nexus: reusing shared EPANET pre-simulation for pump power")
        return rounded(_mean_pump_powers_from_sim(
            epanet_sim["flowrate"], epanet_sim["headloss"], pump_names, pump_efficiency
        ))

    # Fallback: simulate on demand (no cache, or cache without headloss).
    import os
    cwd = os.getcwd()
    try:
        wn = wntr.network.WaterNetworkModel(inp_file)
        wn.options.time.duration = max(T - 1, 0) * 3600
        wn.options.time.hydraulic_timestep = 3600
        wn.options.time.report_timestep = 3600
        with tempfile.TemporaryDirectory(prefix="econex-pump-power-") as sim_dir:
            res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=str(Path(sim_dir) / "network"))
        logging.info("Nexus: EPANET on-demand sim for pump power (no shared cache)")
        return rounded(_mean_pump_powers_from_sim(
            res.link["flowrate"], res.link["headloss"], pump_names, pump_efficiency
        ))
    except Exception as exc:
        raise RuntimeError("Pump power calibration failed; provide valid hydraulic simulation data.") from exc
    finally:
        os.chdir(cwd)


def pump_coupling(data: dict, config: dict) -> dict:
    """Calibrated pump powers, pump-to-bus assignment, and per-bus pump capacity.

    The same calibration and assignment rules as _add_nexus_constraints(), made
    available to the coordination layer without building the coupled model:
    ``powers`` [kW] per pump, ``pump_bus`` per pump, ``bus_capacity_kw`` per
    electrical bus (sum of the powers of the pumps it supplies), and
    ``pump_buses`` (buses with at least one pump).
    """
    if "water" not in data or "energy" not in data:
        raise ValueError("pump_coupling needs both water and energy data.")
    T = int(config.get("T", 24))
    nexus_cfg = config.get("nexus", {}) or {}
    pump_eff = float(nexus_cfg.get("pump_efficiency", 1.0))
    if not 0 < pump_eff <= 1:
        raise ValueError("Pump efficiency must lie in (0, 1].")
    inp_file = data["water"]["inp_file"]
    pumps = list(wntr.network.WaterNetworkModel(inp_file).pump_name_list)
    powers = dict(nexus_cfg.get("pump_power_kw", {}) or {})
    if not powers:
        powers = _pump_mean_powers(inp_file, T, pumps, pump_eff, data["water"].get("epanet_sim"))
    if any(p not in powers or not np.isfinite(powers[p]) or powers[p] <= 0 for p in pumps):
        raise ValueError("Every pump needs positive calibrated power; use nexus.pump_power_kw "
                         "for pumps that do not operate in the reference simulation.")
    buses = list(data["energy"]["buses"])
    ref_bus = data["energy"].get("network", {}).get("reference_bus", buses[0])
    assignment = dict(nexus_cfg.get("pump_bus", {}) or {})
    pump_bus = {}
    for p in pumps:
        b = assignment.get(p, ref_bus)
        if b not in buses:
            raise ValueError(f"Pump {p!r} is mapped to unknown electrical bus {b!r}.")
        pump_bus[p] = b
    capacity = {b: 0.0 for b in buses}
    for p, b in pump_bus.items():
        capacity[b] += float(powers[p])
    return {"powers": {p: float(powers[p]) for p in pumps}, "pump_bus": pump_bus,
            "bus_capacity_kw": capacity, "pump_buses": [b for b in buses if capacity[b] > 0]}


def _add_nexus_constraints(model: pyo.ConcreteModel, data: dict, config: dict) -> None:
    """Couple the water and energy sub-models through pump electrical demand.

    First nexus link (Thomas & Sela / MILPNet convention): each pump draws a
    fixed electrical power when ON. That power, mean(ρg·Q·ΔH) from a WNTR
    pre-simulation, is multiplied by the pump's binary ON status and injected as
    additional demand at its assigned electrical bus — entering the nodal active-power balance via the
    model.PumpBusLoad expression created in _add_energy_submodel().

    Because the pump electricity is now priced through the energy import tariff
    (energy_cost = Σ tariff·P_import), the standalone pump electricity cost in
    model.water_cost is dropped here to avoid double counting; only the demand
    slack penalty is retained on the water side.

    Config (config['nexus']):
        pump_efficiency:  η wire-to-water efficiency (default 1.0, MILPNet).
        pump_bus:         {pump_name: bus_name}. Pumps absent from the map fall
                          back to the reference bus (model.Buses[0]).

    Args:
        model:  Shared ConcreteModel — must contain both sub-models (run_nexus
                forces run_water and run_energy true in build_model()).
        data:   Preprocessing dict; data['water']['inp_file'] is the EPANET file.
        config: Unified configuration dict.
    """
    if not hasattr(model, "Pumps") or not hasattr(model, "PumpBusLoad"):
        raise ValueError(
            "Nexus coupling requires both water and energy sub-models. "
            "Ensure run_nexus=true builds water (Status) and energy (PumpBusLoad)."
        )

    nexus_cfg = config.get("nexus", {}) or {}
    pump_eff = float(nexus_cfg.get("pump_efficiency", 1.0))
    pump_bus = dict(nexus_cfg.get("pump_bus", {}) or {})

    buses = list(model.Buses)
    if not buses:
        raise ValueError("Nexus coupling needs at least one electrical bus.")
    ref_bus = model.reference_bus

    # Mean electrical power per pump [kW]. Reuses the shared EPANET pre-simulation
    # cached in preprocessing (data['water']['epanet_sim']) when available, so the
    # network is not simulated a second time just for the coupling.
    inp_file = data["water"]["inp_file"]
    epanet_sim = data["water"].get("epanet_sim")
    mean_powers = dict(nexus_cfg.get("pump_power_kw", {}) or {})
    if not mean_powers:
        mean_powers = _pump_mean_powers(
            inp_file, len(list(model.T)), model.Pumps, pump_eff, epanet_sim)
    if not 0 < pump_eff <= 1:
        raise ValueError("Pump efficiency must lie in (0, 1].")
    if any(p not in mean_powers or not np.isfinite(mean_powers[p]) or mean_powers[p] <= 0
           for p in model.Pumps):
        raise ValueError("Every pump needs positive calibrated power; use nexus.pump_power_kw "
                         "for pumps that do not operate in the reference simulation.")
    if not model.grid_import_limit_explicit:
        model.GridImportLimit.set_value(pyo.value(model.GridImportLimit) + sum(mean_powers.values()))

    # Resolve each pump to a bus; unmapped pumps go to the reference bus.
    bus_pumps = {b: [] for b in buses}
    for p in model.Pumps:
        b = pump_bus.get(p, ref_bus)
        if b not in bus_pumps:
            raise ValueError(f"Pump {p!r} is mapped to unknown electrical bus {b!r}.")
        bus_pumps[b].append(p)
        if p not in pump_bus:
            logging.info(f"Nexus: pump '{p}' not in pump_bus map; attached to reference bus '{ref_bus}'")

    # Inject  Σ_p  mean_power_p · Status[p, t]  as electrical demand at each bus.
    # Updating the named Expression body propagates into the already-built nodal balance.
    model.pump_mean_power_kw = dict(mean_powers)  # stash for postprocessing/logging
    model.pump_bus = {p: b for b, pump_list in bus_pumps.items() for p in pump_list}
    model.pump_bus_capacity_kw = {b: sum(mean_powers[p] for p in bus_pumps[b]) for b in buses}
    for b in buses:
        for t in model.T:
            model.PumpBusLoad[b, t].set_value(
                sum(mean_powers.get(p, 0.0) * model.Status[p, t] for p in bus_pumps[b])
            )

    # Drop standalone pump electricity cost from the water objective (now priced via
    # the energy import tariff); keep the demand-slack penalty and any tie-break.
    if hasattr(model, "water_cost"):
        model.water_cost.set_value(
            model.level_tracking_cost
            + sum(1e9 * (model.SlackPos[n, t] + model.SlackNeg[n, t])
                  for n in model.Junctions for t in model.T)
        )

    total_kw = sum(mean_powers.values())
    logging.info(
        f"Nexus coupling active: {len(list(model.Pumps))} pumps, "
        f"total mean pump load {total_kw:.2f} kW (η={pump_eff}); "
        f"pump electricity now priced through the energy import tariff."
    )


# ---------------------------------------------------------------------------
# Shared objective
# ---------------------------------------------------------------------------

def _build_objective(model: pyo.ConcreteModel) -> None:
    """Sum all registered sub-model cost expressions into one objective."""
    cost_terms = []
    if hasattr(model, "water_cost"):
        cost_terms.append(model.water_cost)
    if hasattr(model, "energy_cost"):
        cost_terms.append(model.energy_cost)

    if not cost_terms:
        logging.warning("No cost expressions found — using zero objective.")
        model.objective = pyo.Objective(expr=0, sense=pyo.minimize)
        return

    model.objective = pyo.Objective(expr=sum(cost_terms), sense=pyo.minimize)
