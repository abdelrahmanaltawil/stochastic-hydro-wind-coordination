"""Two-operator coordination layer: tariff baselines, the day-ahead revenue-sharing
contract, and the limited-information price coordination (Lagrangian) scheme.

The integrated day-ahead problem of :mod:`src.algorithm_tasks` couples the water
block (MILPNet) and the energy block (electrical part of Morvaj et al.) through
one linear interface: the pump demand ``PumpBusLoad[b, t] = sum_p Pbar_p u[p, t]``
at each supply bus. Here the two blocks are owned by different firms,

* the local water operator W, who decides the pump statuses ``u`` subject to the
  hydraulic block and sees only the electricity price it is charged, and
* the local energy operator E, who owns PV, battery, feeder, and the grid
  connection, faces the day-ahead prices, and serves whatever pump load W meters.

Relaxing the interface with an *internal price* ``pi[b, t]`` separates the
integrated problem into W's ordinary price-based pump-scheduling problem and
E's dispatch problem with a price-responsive "accepted" pump load. The master
problem (Algorithm 1 of the manuscript; a proximal bundle rule by default, a box
cutting-plane or subgradient rule optionally) exchanges only the price vector
and the metered load schedule. The contract functions account for the baseline
bill, the coordination gain, and the payoffs of the revenue-sharing contract
with a fixed fee.
"""
import copy
import logging
import math
import time

import numpy as np
import pyomo.environ as pyo

from . import algorithm_tasks as algorithm


# ---------------------------------------------------------------------------
# Solver wrapper
# ---------------------------------------------------------------------------

def solve(model, solver, timeout, options=None, logfile=None):
    """Solve and return (model, record); record has no objective when infeasible."""
    start = time.perf_counter()
    model, results, _ = algorithm.solve_model(model, solver, timeout, logfile=logfile, options=options)
    record = {
        "status": str(results.solver.status),
        "termination": str(results.solver.termination_condition),
        "time_s": time.perf_counter() - start,
        "objective": None, "lower_bound": None, "upper_bound": None,
    }
    if len(results.solution):
        record["objective"] = float(pyo.value(model.objective))
        for source, target in (("lower_bound", "lower_bound"), ("upper_bound", "upper_bound")):
            try:
                bound = float(getattr(results.problem, source))
                record[target] = bound if math.isfinite(bound) else None
            except (AttributeError, TypeError, ValueError):
                record[target] = None
        if record["lower_bound"] is None:
            # Without a reported dual bound only an optimal termination certifies
            # the incumbent; any other status leaves the bound unknown.
            record["lower_bound"] = record["objective"] if record["termination"] == "optimal" else -math.inf
    elif record["termination"] not in ("infeasible", "infeasibleOrUnbounded", "unbounded"):
        raise RuntimeError(f"Solver returned no incumbent ({record['termination']}).")
    return model, record


def dual_bound_gap(lower, upper):
    """Relative gap (upper - lower) / |upper| with the usual guards."""
    if lower is None or upper is None:
        return None
    return max(0.0, upper - lower) / max(abs(upper), 1e-10)


# ---------------------------------------------------------------------------
# Shared data preparation
# ---------------------------------------------------------------------------

def prepare(data, config):
    """Return (data, config, coupling) with explicit connection limits.

    The integrated model derives loose grid-connection limits from the installed
    loads and assets when the configuration gives none, and the nexus coupling
    widens the import limit by the pump capacity. The subproblems of the
    coordination scheme are built without the nexus block, so the limits are
    made explicit here once, from the integrated model, and copied into the
    energy data used by every build. The water-side pump prices use the same
    calibrated on-state powers as the coupling.
    """
    data = copy.deepcopy(data)
    config = copy.deepcopy(config)
    config["run_water"] = config["run_energy"] = config["run_nexus"] = True
    config.setdefault("water", {})["pump_efficiency"] = config.get("nexus", {}).get("pump_efficiency", 1.0)
    coupling = algorithm.pump_coupling(data, config)
    config.setdefault("nexus", {})["pump_power_kw"] = dict(coupling["powers"])
    data["water"]["pump_power_kw"] = dict(coupling["powers"])
    network = data["energy"].setdefault("network", {})
    if "max_grid_import_kw" not in network or "max_grid_export_kw" not in network:
        probe = algorithm.build_model(data, config)
        network.setdefault("max_grid_import_kw", float(pyo.value(probe.GridImportLimit)))
        network.setdefault("max_grid_export_kw", float(pyo.value(probe.GridExportLimit)))
    return data, config, coupling


def horizon(config):
    return list(range(int(config.get("T", 24))))


def bus_price_from_series(series, coupling, config):
    """Expand a horizon price series into {(bus, t): price} at every pump bus."""
    return {(b, t): float(series[t]) for b in coupling["pump_buses"] for t in horizon(config)}


def pump_price(price_by_bus, coupling, config):
    """Map a bus price to the per-pump price table used by the water block."""
    return {p: [price_by_bus[(coupling["pump_bus"][p], t)] for t in horizon(config)]
            for p in coupling["powers"]}


def load_from_schedule(schedule, coupling, config):
    """Metered pump load {(bus, t): kW} implied by a status schedule."""
    load = {(b, t): 0.0 for b in coupling["bus_capacity_kw"] for t in horizon(config)}
    for (p, t), status in schedule.items():
        load[(coupling["pump_bus"][p], t)] += coupling["powers"][p] * float(status)
    return load


# ---------------------------------------------------------------------------
# Operator subproblems
# ---------------------------------------------------------------------------

def water_operator(data, config, price_by_bus, coupling, *, solver, timeout, options=None,
                   tracking_weight=0.0, logfile=None):
    """W's price-based pump-scheduling problem (P_W) for the internal price.

    Returns the status schedule, the metered load per bus, W's energy cost at
    the given prices (the value of P_W), and the solver's proven lower bound,
    which certifies the dual function value when the MILP is not solved to
    optimality within the time limit.
    """
    cfg = copy.deepcopy(config)
    cfg.update(run_water=True, run_energy=False, run_nexus=False)
    cfg["water"]["pump_energy_price"] = pump_price(price_by_bus, coupling, config)
    cfg["water"]["level_tracking_weight"] = float(tracking_weight)
    model = algorithm.build_model({"water": data["water"]}, cfg)
    model, record = solve(model, solver, timeout, options, logfile)
    if record["objective"] is None and record["termination"] not in ("infeasible", "infeasibleOrUnbounded"):
        # A capped solve without an incumbent: allow one longer attempt.
        model = algorithm.build_model({"water": data["water"]}, cfg)
        model, record = solve(model, solver, 4 * timeout, options, logfile)
    if record["objective"] is None:
        raise RuntimeError(f"Water operator subproblem is {record['termination']}.")
    schedule = {(p, t): int(round(pyo.value(model.Status[p, t]))) for p in model.Pumps for t in model.T}
    load = load_from_schedule(schedule, coupling, config)
    tracking = float(pyo.value(model.level_tracking_cost))
    return {
        "schedule": schedule, "load": load, "model": model,
        "energy_cost": float(pyo.value(model.pump_energy_cost)),
        "tracking_cost": tracking,
        "value": float(pyo.value(model.objective)),
        "lower_bound": record["lower_bound"], "termination": record["termination"],
        "time_s": record["time_s"],
        "on_hours": int(sum(schedule.values())),
    }


def energy_operator(data, config, price_by_bus, coupling, *, accepted=None, solver, timeout,
                    options=None, logfile=None):
    """E's dispatch problem (P_E) with a price-responsive accepted pump load.

    The accepted load ``w[b, t]`` replaces the pump demand in the nodal balance
    and is bounded by the pump capacity of the bus; E is paid the internal
    price for it. With ``accepted`` given, the load is fixed (E's response to
    a metered schedule) and the objective is E's day-ahead procurement cost.
    """
    cfg = copy.deepcopy(config)
    cfg.update(run_water=False, run_energy=True, run_nexus=False)
    model = algorithm.build_model({"energy": data["energy"]}, cfg)
    capacity = coupling["bus_capacity_kw"]
    dt_h = float(pyo.value(model.dt)) / 3600.0
    model.W_accept = pyo.Var(model.Buses, model.T, bounds=lambda m, b, t: (0.0, capacity.get(b, 0.0)))
    for b in model.Buses:
        for t in model.T:
            model.PumpBusLoad[b, t].set_value(model.W_accept[b, t])
            if accepted is not None:
                model.W_accept[b, t].fix(float(accepted.get((b, t), 0.0)))
    price = {(b, t): float(price_by_bus.get((b, t), 0.0)) for b in model.Buses for t in model.T}
    model.internal_price = price
    model.price_revenue = pyo.Expression(expr=dt_h * sum(
        price[b, t] * model.W_accept[b, t] for b in model.Buses for t in model.T))
    model.del_component("objective")
    model.objective = pyo.Objective(expr=model.energy_cost - model.price_revenue, sense=pyo.minimize)
    model, record = solve(model, solver, timeout, options, logfile)
    result = {"model": model, "termination": record["termination"], "time_s": record["time_s"],
              "feasible": record["objective"] is not None}
    if not result["feasible"]:
        return result
    result.update({
        "accepted": {(b, t): float(pyo.value(model.W_accept[b, t])) for b in model.Buses for t in model.T},
        "energy_cost": float(pyo.value(model.energy_cost)),
        "settlement": -float(pyo.value(model.energy_cost)),
        "price_revenue": float(pyo.value(model.price_revenue)),
        "value": float(pyo.value(model.objective)),
        "lower_bound": record["lower_bound"],
        "import_kwh": dt_h * sum(pyo.value(model.P_import[b, t]) for b in model.Buses for t in model.T),
        "export_kwh": dt_h * sum(pyo.value(model.P_export[b, t]) for b in model.Buses for t in model.T),
    })
    return result


def energy_response(data, config, load, coupling, **kwargs):
    """E's best response to a metered pump-load schedule (primal recovery)."""
    zero = {key: 0.0 for key in load}
    return energy_operator(data, config, zero, coupling, accepted=load, **kwargs)


def marginal_cost_prices(data, config, load, coupling, *, solver, timeout, options=None):
    """E's marginal cost of serving a metered pump-load schedule, hour by hour.

    The energy operator dispatches against the metered load, fixes the integer
    modes of that dispatch, and reads the multipliers of its nodal balances:
    the cost of one more kilowatt of pump load at each pump bus and hour. It
    uses only E's own data and the metered schedule, so it is a legitimate
    first message of the coordination scheme (the recommended warm start): by
    Proposition 3 it already equals the internal price in every hour in which
    coordination does not change the state of the connection.
    """
    response = energy_response(data, config, load, coupling, solver=solver, timeout=timeout, options=options)
    if not response["feasible"]:
        raise RuntimeError("Energy operator cannot serve the metered schedule.")
    lp = response["model"].clone()
    for var in lp.component_data_objects(pyo.Var, active=True):
        if var.is_integer():
            var.fix(round(pyo.value(var)))
    pyo.TransformationFactory("core.relax_integer_vars").apply_to(lp)
    lp.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    lp, record = solve(lp, solver, timeout)
    if record["objective"] is None:
        raise RuntimeError("Restricted LP of the energy operator's response is infeasible.")
    dt_h = float(pyo.value(lp.dt)) / 3600.0
    return {(b, t): float(lp.dual[lp.NodalPBalance[b, t]]) / dt_h
            for b in coupling["pump_buses"] for t in lp.T}


def dual_at_prices(data, config, price_by_bus, coupling, *, solver, timeout, water_options=None,
                   energy_options=None):
    """Value of the dual function at a given internal price (both subproblems).

    Used to evaluate D at the restricted-LP prices of the integrated optimum:
    D(pi*) equal to the integrated cost means the Lagrangian relaxation of the
    interface has no duality gap on the instance, so any residual certified gap
    of the coordination runs is a search limitation, not a structural one.
    """
    w = water_operator(data, config, price_by_bus, coupling, solver=solver, timeout=timeout, options=water_options)
    e = energy_operator(data, config, price_by_bus, coupling, solver=solver, timeout=timeout, options=energy_options)
    if not e["feasible"]:
        raise RuntimeError("Energy operator subproblem infeasible at the given prices.")
    response = energy_response(data, config, w["load"], coupling, solver=solver, timeout=timeout, options=energy_options)
    return {"dual_value": w["value"] + e["value"], "dual_certified": w["lower_bound"] + e["lower_bound"],
            "water_value": w["value"], "water_lower_bound": w["lower_bound"], "energy_value": e["value"],
            "water_on_hours": sorted(t for (_, t), status in w["schedule"].items() if status),
            "water_termination": w["termination"], "water_time_s": w["time_s"],
            "primal_from_response": response["energy_cost"] if response["feasible"] else None}


def flat_tariff_baseline(data, config, coupling, tariff, *, solver, timeout, options=None):
    """W's schedule under a flat retail tariff: energy-minimal, then level holding.

    Under a flat tariff every schedule with the same number of pump-hours costs
    W the same, so the tariff selects only the on-hour count. Operating
    practice fixes the timing: the discrete analogue of float operation keeps
    the tank as close as possible to its initial level. The baseline is
    therefore computed in two stages: (1) the least number of pump-hours that
    serves demand and replenishes the tank; (2) among schedules with that many
    hours, the one minimizing the summed absolute level deviation. Stage 2 is
    a separate MILP so that the tie-break is resolved exactly rather than
    within the solver's optimality tolerance.
    """
    price = bus_price_from_series([float(tariff)] * len(horizon(config)), coupling, config)
    stage1 = water_operator(data, config, price, coupling, solver=solver, timeout=timeout, options=options)
    cfg = copy.deepcopy(config)
    cfg.update(run_water=True, run_energy=False, run_nexus=False)
    cfg["water"]["pump_energy_price"] = pump_price(price, coupling, config)
    cfg["water"]["level_tracking_weight"] = 1.0
    model = algorithm.build_model({"water": data["water"]}, cfg)
    model.OnHourBudget = pyo.Constraint(expr=sum(
        model.Status[p, t] for p in model.Pumps for t in model.T) <= stage1["on_hours"])
    model.del_component("objective")
    model.objective = pyo.Objective(expr=model.level_tracking_cost, sense=pyo.minimize)
    model, record = solve(model, solver, timeout, options)
    if record["objective"] is None:
        raise RuntimeError("Level-holding stage of the baseline is infeasible.")
    schedule = {(p, t): int(round(pyo.value(model.Status[p, t]))) for p in model.Pumps for t in model.T}
    load = load_from_schedule(schedule, coupling, config)
    return {
        "schedule": schedule, "load": load, "model": model, "tariff": float(tariff),
        "on_hours": int(sum(schedule.values())), "min_on_hours": stage1["on_hours"],
        "energy_cost": float(pyo.value(model.pump_energy_cost)),
        "bill": flat_tariff_bill(load, tariff, config),
        "level_deviation_mh": float(pyo.value(model.level_tracking_cost)),
        "stage1_time_s": stage1["time_s"], "stage2_time_s": record["time_s"],
        "stage1_termination": stage1["termination"], "stage2_termination": record["termination"],
    }


# ---------------------------------------------------------------------------
# Limited-information price coordination (Algorithm 1)
# ---------------------------------------------------------------------------

def _bundle_master(cuts, keys, center, tau, bounds, cap):
    """Proximal bundle master: max theta - ||pi - center||^2 / (2 tau) s.t. cuts.

    The quadratic term keeps the proposal near the incumbent price (the
    "stabilization" that plain cutting planes lack); ``tau`` is the proximal
    step, enlarged after serious steps and reduced after null steps. Solved as
    a small QP with SLSQP; returns (model value at the proposal, proposal).
    """
    from scipy.optimize import minimize
    n = len(keys)
    index = {key: i for i, key in enumerate(keys)}
    rows, rhs = [], []
    for value, price, imbalance in cuts:
        g = np.array([imbalance[key] for key in keys])
        p = np.array([price[key] for key in keys])
        rows.append(g)
        rhs.append(value - g @ p)
    A, b = np.array(rows), np.array(rhs)          # theta <= b_k + A_k pi
    c = np.array([center[key] for key in keys])
    scale = max(1.0, max(abs(v) for v in b) if len(b) else 1.0)

    def objective(x):
        theta, pi = x[0], x[1:]
        return (-theta + (pi - c) @ (pi - c) / (2.0 * tau)) / scale

    def gradient(x):
        pi = x[1:]
        return np.concatenate([[-1.0], (pi - c) / tau]) / scale

    constraints = [{"type": "ineq", "fun": lambda x, i=i: (b[i] + A[i] @ x[1:] - x[0]) / scale,
                    "jac": lambda x, i=i: np.concatenate([[-1.0], A[i]]) / scale} for i in range(len(b))]
    x0 = np.concatenate([[min(b + A @ c) if len(b) else 0.0], c])
    bnds = [(None, cap)] + [(bounds[0], bounds[1])] * n
    result = minimize(objective, x0, jac=gradient, bounds=bnds, constraints=constraints, method="SLSQP",
                      options={"maxiter": 500, "ftol": 1e-12})
    x = result.x if result.success or np.all(np.isfinite(result.x)) else x0
    pi = np.clip(x[1:], bounds[0], bounds[1])
    model_value = float(min(b + A @ pi)) if len(b) else float("inf")
    return min(model_value, cap), {key: float(pi[index[key]]) for key in keys}


def _cutting_plane_master(cuts, keys, center, radius, bounds, cap):
    """Trust-region cutting-plane master: max theta s.t. the dual model cuts.

    ``cuts`` are (value, price, imbalance) triples; every cut is the affine
    majorant theta <= value + g'(pi - pi_k) of the polyhedral dual function.
    The price stays within ``bounds`` and within ``radius`` of ``center``
    (infinity norm), and theta is capped by the best primal value. Solved with
    scipy's HiGHS LP interface; returns (theta, price).
    """
    from scipy.optimize import linprog
    n = len(keys)
    index = {key: i for i, key in enumerate(keys)}
    # variables: [theta, pi_0 ... pi_{n-1}]; maximize theta -> minimize -theta
    c = np.zeros(n + 1)
    c[0] = -1.0
    rows, rhs = [], []
    for value, price, imbalance in cuts:
        row = np.zeros(n + 1)
        row[0] = 1.0
        offset = value
        for key in keys:
            g = imbalance[key]
            row[index[key] + 1] = -g
            offset -= g * price[key]
        rows.append(row)
        rhs.append(offset)
    lo = [(-math.inf, cap)]
    for key in keys:
        low = max(bounds[0], center[key] - radius)
        high = min(bounds[1], center[key] + radius)
        lo.append((low, max(high, low)))
    result = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=lo, method="highs")
    if not result.success:
        raise RuntimeError(f"Cutting-plane master failed: {result.message}")
    theta = float(result.x[0])
    price = {key: float(result.x[index[key] + 1]) for key in keys}
    return theta, price


def price_coordination(data, config, coupling, *, solver, timeout, price0, master="bundle",
                       step_rule="polyak", step_scale=1.0, max_iter=60, tol=1e-4,
                       price_resolution=1e-4, trust_radius=5e-4, price_bounds=None,
                       water_options=None, energy_options=None, primal_target=None,
                       subproblem_timeout=None, callback=None):
    """Dual coordination of W and E through the internal price (Algorithm 1).

    Each iteration: E posts ``pi``; W solves (P_W) and meters its load ``l``;
    E solves (P_E) and forms the imbalance ``g = l - w`` against the load it
    accepted at ``pi``; the dual value ``D(pi)`` is the sum of the two
    subproblem values (a lower bound on the integrated optimum when both are
    solved to optimality; the solvers' proven bounds give a certified bound
    otherwise); E also dispatches against the metered schedule, which yields a
    feasible integrated point (primal recovery) and an upper bound.

    Three master rules update the price. ``master="subgradient"`` moves along
    the imbalance with Polyak, diminishing, or constant steps (the rule of the
    two-agent LP model). ``master="bundle"`` (default) keeps every (value,
    price, imbalance) triple as an affine majorant of the polyhedral dual
    function and maximizes that model minus a proximal term around the best
    price found so far (serious/null steps; ``trust_radius`` is the proximal
    step, enlarged after serious and reduced after null steps).
    ``master="cutting_plane"`` replaces the proximal term by a box of radius
    ``trust_radius``. With integer pump commitments the bundle rule certifies
    the dual bound in far fewer iterations than subgradient steps. All rules
    use only the load schedules W sends and E's own subproblem, so exactly
    ``2 * |pump buses| * T`` scalars cross the boundary per iteration. Prices
    are exchanged with finite ``price_resolution``. ``subproblem_timeout``
    caps each operator's solve (default: ``timeout``); a capped water solve
    returns its incumbent and proven bound, which keeps every cut and the
    certified bound valid.
    """
    times = horizon(config)
    sub_timeout = subproblem_timeout or timeout
    buses = list(coupling["pump_buses"])
    keys = [(b, t) for b in buses for t in times]

    def quantize(values):
        if not price_resolution:
            return dict(values)
        return {k: round(v / price_resolution) * price_resolution for k, v in values.items()}

    price = quantize({key: float(price0[key]) for key in keys})
    if price_bounds is None:
        top = max(max(data["energy"]["tariff"]), max(data["energy"]["export_tariff"]), 0.0)
        price_bounds = (0.0, 2.0 * top + 1e-3)
    best_upper, best_lower = math.inf, -math.inf
    best = None
    history, cuts = [], []
    center, center_value, center_certified, radius, predicted = dict(price), -math.inf, -math.inf, trust_radius, -math.inf
    started = time.perf_counter()
    converged = False
    for k in range(max_iter):
        w_res = water_operator(data, config, price, coupling, solver=solver, timeout=sub_timeout,
                               options=water_options)
        e_res = energy_operator(data, config, price, coupling, solver=solver, timeout=sub_timeout,
                                options=energy_options)
        if not e_res["feasible"]:
            raise RuntimeError("Energy operator subproblem infeasible; check connection limits.")
        dual_value = w_res["value"] + e_res["value"]
        dual_certified = w_res["lower_bound"] + e_res["lower_bound"]
        response = energy_response(data, config, w_res["load"], coupling, solver=solver,
                                   timeout=timeout, options=energy_options)
        primal = response["energy_cost"] if response["feasible"] else math.inf
        imbalance = {key: w_res["load"][key] - e_res["accepted"][key] for key in keys}
        norm = math.sqrt(sum(g * g for g in imbalance.values()))
        best_lower = max(best_lower, dual_certified)
        if primal < best_upper:
            best_upper = primal
            best = {"iteration": k, "schedule": dict(w_res["schedule"]), "load": dict(w_res["load"]),
                    "energy_cost": primal, "price": dict(price), "response": response,
                    "on_hours": w_res["on_hours"]}
        gap = dual_bound_gap(best_lower, best_upper)
        entry = {
            "iteration": k, "price": dict(price), "load": dict(w_res["load"]),
            "accepted": dict(e_res["accepted"]), "imbalance_norm_kw": norm,
            "water_value": w_res["value"], "energy_value": e_res["value"],
            "dual_value": dual_value, "dual_certified": dual_certified,
            "primal_value": primal, "best_upper": best_upper, "best_lower": best_lower,
            "gap": gap, "messages": 2 * len(keys) * (k + 1),
            "water_time_s": w_res["time_s"], "energy_time_s": e_res["time_s"],
            "response_time_s": response["time_s"], "elapsed_s": time.perf_counter() - started,
            "water_termination": w_res["termination"], "energy_termination": e_res["termination"],
            "on_hours": w_res["on_hours"], "step": 0.0, "trust_radius": radius, "serious": None,
        }
        history.append(entry)
        logging.info("coordination k=%d D=%.4f (cert %.4f) primal=%.4f best=[%.4f, %.4f] |g|=%.3f",
                     k, dual_value, dual_certified, primal, best_lower, best_upper, norm)
        if callback:
            callback(entry)
        if norm <= 1e-9 or (gap is not None and gap <= tol):
            converged = True
            break
        if master == "subgradient":
            target = best_upper if primal_target is None else primal_target
            if step_rule == "polyak":
                level = max(target - dual_value, 0.0)
                step = step_scale * level / max(norm * norm, 1e-12)
            elif step_rule == "diminishing":
                step = step_scale / math.sqrt(k + 1)
            elif step_rule == "constant":
                step = step_scale
            else:
                raise ValueError(f"Unknown step rule {step_rule!r}.")
            entry["step"] = step
            price = quantize({key: price[key] + step * imbalance[key] for key in keys})
        elif master in ("cutting_plane", "bundle"):
            cuts.append((dual_value, dict(price), dict(imbalance)))
            # The subproblems may stop at a tolerance or a time cap, so the
            # dual values of the cuts overestimate D. The centre is therefore
            # the price with the best *certified* value, and the centre value
            # is the cut model evaluated there (the tightest upper estimate
            # available), which every new cut can only lower. A serious step
            # moves the centre when the certified value improves on it;
            # a null step tightens the stabilization instead.
            if dual_certified > center_certified:
                serious = True
                center, center_certified = dict(price), dual_certified
                radius = min(2.0 * radius, 10.0 * trust_radius) if k > 0 else radius
            else:
                serious = False
                radius = max(0.5 * radius, 0.02 * trust_radius)
            center_value = min(v + sum(g[key] * (center[key] - p[key]) for key in keys)
                               for v, p, g in cuts)
            entry["serious"] = serious
            entry["center_value"] = center_value
            cap = best_upper if math.isfinite(best_upper) else math.inf
            if master == "cutting_plane":
                predicted, proposal = _cutting_plane_master(cuts, keys, center, radius, price_bounds, cap)
            else:
                predicted, proposal = _bundle_master(cuts, keys, center, radius, price_bounds, cap)
            entry["predicted"] = predicted
            entry["model_gap"] = predicted - center_value
            settled = radius >= trust_radius or radius <= 0.02 * trust_radius * (1 + 1e-9)
            if predicted - center_value <= tol * max(abs(center_value), 1.0) and settled:
                converged = True
                break
            price = quantize(proposal)
            if all(abs(price[key] - center[key]) < 0.5 * (price_resolution or 1e-12) for key in keys):
                # The model proposes the incumbent price again: the dual model is
                # exhausted at this resolution.
                converged = True
                break
        else:
            raise ValueError(f"Unknown master rule {master!r}.")
    return {"history": history, "best": best, "best_upper": best_upper, "best_lower": best_lower,
            "converged": converged, "iterations": len(history),
            "messages": history[-1]["messages"] if history else 0,
            "elapsed_s": time.perf_counter() - started, "master": master, "step_rule": step_rule,
            "center_price": center, "center_value": center_value, "center_certified": center_certified}


# ---------------------------------------------------------------------------
# Internal price characterization (restricted-LP multipliers)
# ---------------------------------------------------------------------------

def restricted_lp_prices(model, solver, timeout=300):
    """Nodal internal prices from the LP obtained by fixing the integer decisions.

    With every binary fixed at its optimal value the integrated problem is an
    LP whose multiplier on the nodal active-power balance is the marginal cost
    of serving one more kW of pump load at that bus and hour, i.e. the internal
    price of Proposition 3. Returns {(bus, t): price} together with the hourly
    connection state used to classify each hour (import, export, idle, and
    whether a connection limit binds).
    """
    lp = model.clone()
    for var in lp.component_data_objects(pyo.Var, active=True):
        if var.is_integer():
            var.fix(round(pyo.value(var)))
    pyo.TransformationFactory("core.relax_integer_vars").apply_to(lp)
    lp.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    lp, record = solve(lp, solver, timeout)
    if record["objective"] is None:
        raise RuntimeError("Restricted LP infeasible.")
    dt_h = float(pyo.value(lp.dt)) / 3600.0
    prices, state = {}, {}
    import_limit = float(pyo.value(lp.GridImportLimit))
    export_limit = float(pyo.value(lp.GridExportLimit))
    for b in lp.Buses:
        for t in lp.T:
            prices[(b, t)] = float(lp.dual[lp.NodalPBalance[b, t]]) / dt_h
    for t in lp.T:
        imported = sum(pyo.value(lp.P_import[b, t]) for b in lp.GridBuses)
        exported = sum(pyo.value(lp.P_export[b, t]) for b in lp.GridBuses)
        available = sum(float(pyo.value(lp.PVCap[b, t].upper)) for b in lp.Buses)
        dispatched = sum(pyo.value(lp.P_pv[b, t]) for b in lp.Buses)
        curtailed = available - dispatched > 1e-6
        if imported > 1e-6:
            mode = "import"
        elif exported > 1e-6:
            mode = "export"
        else:
            mode = "idle"
        # A limit binds when the connection sits at its rating in the direction
        # it is used; a site that curtails PV while exporting nothing is held
        # by an export rating it cannot use (zero rating or exclusivity).
        state[t] = {"mode": mode, "import_kw": imported, "export_kw": exported,
                    "pv_curtailed_kw": max(0.0, available - dispatched),
                    "import_limit_binding": mode == "import" and imported >= import_limit - 1e-6,
                    "export_limit_binding": (mode == "export" and exported >= export_limit - 1e-6)
                                            or (mode != "export" and curtailed)}
    return {"price": prices, "state": state, "objective": record["objective"]}


def classify_internal_price(price_by_bus, state, import_price, export_price, bus, tol=1e-6):
    """Compare internal prices at one bus with the market prices hour by hour."""
    rows = []
    for t, info in state.items():
        pi = price_by_bus[(bus, t)]
        if info["mode"] == "import":
            reference = import_price[t]
        elif info["mode"] == "export":
            reference = export_price[t]
        else:
            reference = None
        rows.append({"hour": t, "internal_price": pi, "mode": info["mode"],
                     "import_price": import_price[t], "export_price": export_price[t],
                     "connection_binding": info["import_limit_binding"] or info["export_limit_binding"],
                     "within_band": export_price[t] - tol <= pi <= import_price[t] + tol,
                     "equals_reference": reference is not None and abs(pi - reference) <= tol})
    return rows


# ---------------------------------------------------------------------------
# Contract accounting
# ---------------------------------------------------------------------------

def flat_tariff_bill(load, tariff, config):
    """Baseline bill: the metered pump energy priced at a flat retail tariff."""
    return float(tariff) * sum(load.values())  # kW x 1 h per interval


def contract_payoffs(settlement, share, fee):
    """(Pi_W, Pi_E) under the contract: E pays W the share of its settlement minus the fee."""
    transfer = share * settlement - fee
    return transfer, settlement - transfer


def anchored_fee(bill0, settlement0, share):
    """Fee that reproduces the baseline payoffs at the baseline schedule.

    With F = bill0 + share * R0 the contract leaves both operators exactly at
    their baseline payoffs when the baseline schedule is run, so following the
    coordinated schedule yields W a payoff gain of share * G and E a gain of
    (1 - share) * G, whatever the sign of the settlement.
    """
    return bill0 + share * settlement0


def fee_window(bill0, settlement0, settlement_star, share):
    """Fees for which the contract Pareto-dominates the baseline at a given share."""
    low = bill0 + settlement0 - (1.0 - share) * settlement_star
    high = bill0 + share * settlement_star
    return low, high


def payoff_table(bill0, settlement0, settlement_star, shares):
    """Baseline and contract payoffs (currency per day) for a list of shares."""
    gain = settlement_star - settlement0
    rows = [{"arrangement": "baseline", "share": None, "fee": None,
             "water": -bill0, "energy": settlement0 + bill0, "total": settlement0, "gain": 0.0}]
    for share in shares:
        fee = anchored_fee(bill0, settlement0, share)
        water, energy = contract_payoffs(settlement_star, share, fee)
        rows.append({"arrangement": "contract", "share": share, "fee": fee,
                     "water": water, "energy": energy, "total": settlement_star,
                     "gain": water + energy - settlement0,
                     "water_gain": water + bill0, "energy_gain": energy - (settlement0 + bill0)})
    return {"gain": gain, "rows": rows}
