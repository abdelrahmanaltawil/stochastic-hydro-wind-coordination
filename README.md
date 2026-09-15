# EcoNex: two-operator water–energy coordination

EcoNex models the day-ahead coordination of two separately owned systems behind one grid connection: a **water operator** (a water distribution network with fixed-speed pumps and elevated storage, formulated with MILPNet) and an **energy operator** (a distribution feeder with photovoltaic generation, a battery, and the market interface, formulated with the electrical part of Morvaj et al.). The two blocks share one metered quantity, the pump load at its supply bus. The repository provides

- the integrated day-ahead MILP (the benchmark a single owner would solve),
- the two operators' subproblems and the limited-information **price coordination** scheme (Lagrangian relaxation of the interface with a bundle or subgradient master; only prices and metered load schedules cross the boundary),
- the accounting of the **revenue-sharing contract** (baseline bills, coordination gain, payoffs, fee window), and
- independent EPANET and OpenDSS replay checks of every schedule.

The repository name reflects an earlier project direction. The study is deterministic; it does not implement stochastic scenarios, wind generation, or hydropower turbines. The older comparison notebook is a separate simplified example.

## Install and run

Use Python 3.10 or newer in an isolated environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.workflow --solver highs
```

HiGHS is installed through `highspy`; a commercial license is unnecessary. Gurobi, GLPK, CBC, and CPLEX are supported when installed separately. `--solver` overrides the configured solver. Solver time limits may produce a feasible incumbent without proving optimality; always inspect `termination_condition` and the solver bounds in the saved summary/log.

> **HiGHS version.** The pinned `highspy==1.11.0` is the version used for the paper run. HiGHS 1.15.1 returned invalid dual bounds on the water operator's subproblem of this model class (its presolve pruned the optimum and certified a bound above a feasible schedule); `presolve=off` avoided it at the cost of speed. Keep the pin, or check any newer version against `tests/` and against a fixed-schedule solve before trusting its bounds.

Both `python -m src.workflow` and `python src/workflow.py` work. Configuration network paths and relative output paths resolve against the repository root. The CLI also accepts `--config`, `--output-dir`, and `--timeout`.

## Reproduce the paper

```sh
python -m src.experiments            # ~3 h with HiGHS on a laptop-class machine
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
```

`data/inputs/paper_case.yaml` defines the synthetic 24-hour study: the SNET water network (one pump, one elevated tank) and the two-bus feeder of `master.dss` with its load scaled to a 100 kW peak, a 200 kWp PV array, a 200 kWh/50 kW battery, a connection rated 150 kW import / 50 kW export, time-of-use purchase prices, and a flat sale price. Its `coordination` section sets the flat retail tariff of the baseline, the contract shares reported, the coordination master and iteration budget, and the connection-limit sweeps. The previous single-operator study is recovered with `load_scale: 1.0`, `capacity_kw: 100`, `grid_export_tariff: 0.0`, and no explicit connection limits.

The experiment command runs, in order: the integrated benchmark with its restricted-LP internal prices; the flat-tariff baseline (energy-minimal pump-hours, then level holding); the pass-through and internal-price tariff benchmarks; the contract accounting (payoff table, fee window, deviation test); the price coordination scheme from warm and cold starts with the bundle master and with the subgradient rule; the import- and export-rating sweeps; and the EPANET/OpenDSS replay of the baseline and coordinated schedules. It writes full-precision tables, solver logs, saved configurations, input/source hashes, dependency versions, and replay results under `data/results/paper_study/` (`study.json` holds every number the paper quotes), and it regenerates `paper/results.tex`, `paper/results_macros.tex` (the numbers quoted in the running text as LaTeX macros), and `paper/figures/*.pdf`. Use `--max-iterations` to shorten the coordination runs when checking the pipeline.

The manuscript uses the IEEEtran document class with the `IEEEtran` bibliography style and the `algorithm2e` package. All ship with TeX Live (`texlive-publishers`, `texlive-science`), MacTeX, and Overleaf. The recorded paper run uses HiGHS 1.11.0 (`requirements-reproducible.txt`); with Gurobi installed, `python -m src.experiments --solver gurobi` reproduces the study faster and the paper text updates through the macros.

## Model and validation

- Water operator (`Q`): junction mass conservation, a configurable pressure floor, signed piecewise-linear Hazen–Williams pipe losses, fixed-speed pump curves interpolated over the hydraulically admissible flow range, on/off decisions with a network-derived big-M, cylindrical tank bounds, and an end-of-horizon reserve requirement. The pump load is priced by a flat tariff (`pump_energy_tariff`), an hourly price series, or a per-pump price table (`pump_energy_price`); an optional `level_tracking_weight` adds the level-holding tie-break used by the baseline.
- Energy operator (`X`): actual OpenDSS bus/line/loadshape extraction with `load_scale`, active and reactive bus balances, a balanced lossless linearization, voltage bounds, linear current-square approximations for ampacity limits, and explicit connection ratings (`max_grid_import_kw`, `max_grid_export_kw`). Grid exchange is restricted to source buses; exports earn `grid_export_tariff`.
- Interface: each running pump draws a calibrated constant electrical power (rounded to 10 W) at its mapped bus. This approximates flow-dependent pump power.
- Coordination (`src/coordination.py`): `water_operator` (W's price-based scheduling problem), `energy_operator` (E's dispatch with a price-responsive accepted load, or its response to a metered schedule), `flat_tariff_baseline`, `price_coordination` (bundle, box cutting-plane, or subgradient master; certified dual bounds from the solvers' proven bounds; primal recovery by E's response), `restricted_lp_prices` (nodal internal prices at the integrated optimum), and the contract accounting functions.

Use `run_water`, `run_energy`, and `run_nexus` to select the components. Nexus mode enables both networks. Asset `bus` fields prevent inadvertently installing the configured capacity at every bus. Empty PV availability means zero PV; empty import tariffs mean 0.10 per kWh.

`src.validation` checks algebraic residuals and replays pump commands and scheduled electrical injections in the nonlinear simulators. It reports pressure, flow, tank, voltage, and line-loading differences separately from MILP feasibility. A MILP-feasible schedule is not automatically feasible in nonlinear physics: within-hour hydraulic behavior and terminal reserve differences require review.

## Outputs and tests

A normal workflow run creates a timestamped directory containing `water_data/`, `energy_data/`, `00_summary.json`, and `metadata/`. Failed workflows save `00_failure.json` and input/run metadata rather than reporting successful completion.

```sh
python -m pytest -q
```

Tests check conservation, final-hour accounting, efficiency, pump coupling, feeder transfer and limits, parsing, result precision, metadata, command-line behavior, the contract accounting identities, the internal-price characterization, and the coordination scheme on a toy site. They do not depend on a Gurobi license.

## Files

| Location | Purpose |
|---|---|
| `src/algorithm_tasks.py` | MILP construction (water block, energy block, interface) and solver handling |
| `src/coordination.py` | Operator subproblems, price coordination, contract accounting |
| `src/preprocessing.py` | EPANET calibration and actual DSS network data |
| `src/postprocessing.py` | Complete solution tables and run metadata |
| `src/workflow.py` | Configurable command-line pipeline |
| `src/experiments.py`, `src/validation.py` | Paper experiments and independent replay |
| `data/inputs/paper_case.yaml` | Reproducible illustrative study |
| `paper/main.tex`, `paper/references.bib` | Manuscript and bibliography |
| `doc/implementation.md` | Implementation assumptions and verification notes |
