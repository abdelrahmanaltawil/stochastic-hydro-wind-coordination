"""Hydraulic utility functions for EcoNex optimization.

Hazen-Williams calculations, pump curve fitting, and piecewise-linear
approximation helpers shared by the water sub-model and validation tests.
"""

import numpy as np
import pyomo.environ as pyo


# ---------------------------------------------------------------------------
# Hazen-Williams helpers
# ---------------------------------------------------------------------------

def calc_K(L, D, R):
    """Hazen-Williams resistance coefficient K = 10.67·L / (R^1.852 · D^4.8704)."""
    if not all(np.isfinite(v) for v in (L, D, R)) or L < 0 or D <= 0 or R <= 0:
        raise ValueError("Pipe length must be nonnegative, diameter and roughness positive.")
    return (10.67 * L) / ((R ** 1.852) * (D ** 4.8704))


def get_pump_curve_points(pump):
    """Extract (flow, head) breakpoints from an EPANET pump object."""
    pts = pump.get_pump_curve().points
    if len(pts) == 1:
        Q0, H0 = pts[0]
        return [(0, H0 * (4.0 / 3.0)), (Q0, H0), (2 * Q0, 0)]
    return list(pts)


def create_piecewise_pipe_curve(K, max_flow, num_segments=5):
    """(flow, head_loss) breakpoints for Hazen-Williams PWL: dH = sign(Q)·K·|Q|^1.852."""
    if not np.isfinite(max_flow) or max_flow <= 0 or num_segments < 2:
        raise ValueError("Pipe flow ceiling must be positive, with at least two segments.")
    if not np.isfinite(K) or K < 0:
        raise ValueError("Pipe resistance must be finite and nonnegative.")
    # Zero is an exact breakpoint, including when an odd segment count is requested.
    flows = np.unique(np.append(np.linspace(-max_flow, max_flow, num_segments + 1), 0.0))
    head_losses = np.sign(flows) * K * np.abs(flows) ** 1.852
    return [(float(q), float(h)) for q, h in zip(flows, head_losses)]


def pump_head_and_slope(pump, flow):
    """EPANET head-curve value and slope for a fixed-speed pump.

    One/three-point curves use the EPANET power law. Two-point and multi-point
    curves use straight interpolation/extrapolation of the supplied points.
    """
    pts = sorted(pump.get_pump_curve().points)
    if not pts or any(not np.isfinite(q+h) or q < 0 for q, h in pts):
        raise ValueError("Pump curve requires finite points with nonnegative flow.")
    if len(pts) in (1, 3):
        A, B, C = pump.get_head_curve_coefficients()
        if flow < 0 or (flow == 0 and C < 1):
            raise ValueError("Pump flow must lie in the differentiable nonnegative domain.")
        return float(A - B * flow ** C), float(-B * C * flow ** (C - 1))
    if any(pts[i+1][0] <= pts[i][0] or pts[i+1][1] >= pts[i][1] for i in range(len(pts)-1)):
        raise ValueError("Pump curve flows must increase and heads must decrease.")
    i = min(max(int(np.searchsorted([q for q, _ in pts], flow)) - 1, 0), len(pts)-2)
    q1, h1 = pts[i]
    q2, h2 = pts[i+1]
    slope = (h2-h1)/(q2-q1)
    return float(h1 + slope * (flow-q1)), float(slope)


def pump_zero_head_flow(pump):
    """Flow at which the fixed-speed pump curve reaches zero head."""
    pts = sorted(pump.get_pump_curve().points)
    if len(pts) in (1, 3):
        A, B, C = pump.get_head_curve_coefficients()
        if B <= 0:
            raise ValueError("Pump curve must have a finite zero-head flow.")
        return float((A / B) ** (1 / C))
    shutoff, _ = pump_head_and_slope(pump, 0.)
    last_head, last_slope = pump_head_and_slope(pump, pts[-1][0])
    cutoff = pts[-1][0] - last_head / last_slope
    if shutoff <= 0 or cutoff <= 0:
        raise ValueError("Pump curve must have positive shutoff head and flow range.")
    return float(cutoff)


def pump_flow_at_head(pump, head, tol=1e-9):
    """Largest flow at which the (decreasing) pump curve still delivers ``head``.

    Bisection on the curve between zero flow and the zero-head flow; returns
    the zero-head flow when ``head`` is nonpositive and 0 when it exceeds the
    shutoff head.
    """
    cutoff = pump_zero_head_flow(pump)
    if head <= 0:
        return cutoff
    if pump_head_and_slope(pump, 0.)[0] <= head:
        return 0.0
    lo, hi = 0.0, cutoff
    while hi - lo > tol * max(cutoff, 1.0):
        mid = 0.5 * (lo + hi)
        if pump_head_and_slope(pump, mid)[0] >= head:
            lo = mid
        else:
            hi = mid
    return float(lo)


def create_piecewise_pump_curve(pump, num_segments=5, max_flow=None):
    """Nonnegative head/flow breakpoints through shutoff and zero-head flow.

    ``max_flow`` truncates the interpolation domain to [0, max_flow] (the
    hydraulically admissible flow range) while keeping the segment count.
    """
    if num_segments < 1:
        raise ValueError("Pump curve requires at least one segment.")
    pts = sorted(pump.get_pump_curve().points)
    cutoff = pump_zero_head_flow(pump)
    if max_flow is not None:
        if not np.isfinite(max_flow) or max_flow <= 0:
            raise ValueError("Pump flow ceiling must be positive.")
        cutoff = min(cutoff, float(max_flow))
    if len(pts) in (1, 3):
        flows = np.linspace(0, cutoff, num_segments + 1)
    else:
        flows = np.unique([0., cutoff] + [q for q, _ in pts if 0 < q < cutoff])
    return [(float(q), max(0., pump_head_and_slope(pump, q)[0])) for q in flows]


# ---------------------------------------------------------------------------
# Shared PWL constraint builder (used by water and energy sub-models)
# ---------------------------------------------------------------------------

def add_pwl_constraint(model, name, x_var, y_var, points, activation=1):
    """Add a piecewise-linear constraint y = f(x) to a Pyomo model.

    Implements SOS2 logic via explicit binary segment variables so the
    model stays compatible with open-source MILP solvers (GLPK, CBC).

    Args:
        model:  Pyomo ConcreteModel to attach components to.
        name:   Unique prefix for all added components.
        x_var:  Pyomo Var (scalar) for the x-axis.
        y_var:  Pyomo Var (scalar) for the y-axis.
        points: List of (x, y) breakpoint tuples in ascending x order.
        activation: Binary status or 1; when zero, both x and y are zero.
    """
    n = len(points)
    if n < 2 or any(not np.isfinite(x) or not np.isfinite(y) for x, y in points):
        raise ValueError("PWL curves require at least two finite breakpoints.")
    if any(points[k + 1][0] <= points[k][0] for k in range(n - 1)):
        raise ValueError("PWL breakpoint x values must be strictly increasing.")
    indices = range(n)
    segments = range(n - 1)

    x_pts = [p[0] for p in points]
    y_pts = [p[1] for p in points]

    w = pyo.Var(indices, bounds=(0, 1))
    model.add_component(f"{name}_w", w)

    z = pyo.Var(segments, domain=pyo.Binary)
    model.add_component(f"{name}_z", z)

    model.add_component(f"{name}_convex",
                        pyo.Constraint(expr=sum(w[k] for k in indices) == activation))
    model.add_component(f"{name}_z_sum",
                        pyo.Constraint(expr=sum(z[k] for k in segments) == activation))

    model.add_component(f"{name}_sos_start",
                        pyo.Constraint(expr=w[0] <= z[0]))
    for k in range(1, n - 1):
        model.add_component(f"{name}_sos_{k}",
                            pyo.Constraint(expr=w[k] <= z[k - 1] + z[k]))
    model.add_component(f"{name}_sos_end",
                        pyo.Constraint(expr=w[n - 1] <= z[n - 2]))

    model.add_component(f"{name}_x_interp",
                        pyo.Constraint(expr=x_var == sum(w[k] * x_pts[k] for k in indices)))
    model.add_component(f"{name}_y_interp",
                        pyo.Constraint(expr=y_var == sum(w[k] * y_pts[k] for k in indices)))
