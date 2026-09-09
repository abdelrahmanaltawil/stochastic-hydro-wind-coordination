# EcoNex: water–energy scheduling

EcoNex coordinates a water distribution network with an electrical feeder, photovoltaic generation, a battery, and grid exchange over hourly operating intervals. It builds a deterministic mixed-integer linear program (MILP) in Pyomo and includes independent EPANET and OpenDSS replay checks.

The repository name reflects an earlier project direction. The implemented study is **deterministic water–energy coordination**. It does not implement stochastic scenarios, wind generation, hydropower turbines, or decentralized market coordination. The older comparison notebook is a separate simplified example.

## Install and run

Use Python 3.10 or newer in an isolated environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.workflow --solver highs
```

HiGHS is installed through `highspy`; a commercial license is unnecessary. Gurobi, GLPK, CBC, and CPLEX are supported when installed separately. The existing default configuration selects Gurobi; `--solver highs` overrides it. Solver time limits may produce a feasible incumbent without proving optimality. Always inspect `termination_condition` and the solver bounds in the saved summary/log.

Both `python -m src.workflow` and `python src/workflow.py` work. The direct script may be called by its absolute path from another directory. Configuration network paths and relative output paths resolve against the repository root. The CLI also accepts `--config`, `--output-dir`, and `--timeout`.

## Reproduce the paper

```sh
python -m src.experiments
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex
```

`data/inputs/paper_case.yaml` defines the synthetic 24-hour study. The experiment command writes full-precision tables, solver logs, saved configurations, input/source hashes, dependency versions, and replay results under `data/results/paper_study/`. It regenerates `paper/results.tex` and `paper/figures/coordination_results.pdf`. The paper compiles to `paper/main.pdf`; the reviewed delivery copy is `output/pdf/water_energy_coordination.pdf`.

The manuscript uses the IEEEtran document class (IEEE Transactions two-column layout) with the `IEEEtran` bibliography style. Both ship with TeX Live's `texlive-publishers` collection, MacTeX, and Overleaf. The water block of the formulation is imported from MILPNet (Thomas and Sela, 2024) and the energy block from the electrical part of Morvaj et al. (2016); the heat balance of the latter is omitted because the two systems interact only through electrical demand.

The recorded paper run uses Gurobi 12.0.3. To run without a commercial license, use `python -m src.experiments --solver highs`; this case can require a longer time limit to prove optimality with HiGHS. The tested Python dependency versions are in `requirements-reproducible.txt`.

The experiment first minimizes pumping energy under a uniform water-side electricity price. It then compares that fixed pump schedule against joint optimization, with electrical asset dispatch optimized under the same time-varying tariff in both cases. A flat-price case provides a separate tariff sensitivity. The particular independent schedule is recorded because multiple water-only optima may exist. Tariffs and PV profiles are illustrative inputs, not measured observations.

Use `--solver` to choose another installed solver. `--output-dir` selects the study directory, and `--paper-dir` selects the destination for generated manuscript results and figures. The result-writing template describes the supplied illustrative case; use that case when regenerating the paper. It is not an automatic authoring template for arbitrary networks.

## Model and validation

- Water: junction mass conservation, a configurable pressure floor, signed piecewise-linear Hazen–Williams pipe losses, fixed-speed pump curves and on/off decisions, cylindrical tank bounds, and an end-of-horizon reserve requirement. Demand slack is disabled by default.
- Electricity: actual OpenDSS bus/line/loadshape extraction, active and reactive bus balances, a balanced lossless linearization, voltage bounds, and linear current-square approximations for ampacity limits. Grid exchange is restricted to source buses.
- Assets: one configured PV site and battery site; battery efficiency is applied once in inventory dynamics. All 24 operating decisions update storage, producing 25 boundary states.
- Coupling: each running pump draws a calibrated constant electrical power at its mapped bus. This approximates flow-dependent pump power.

Use `run_water`, `run_energy`, and `run_nexus` to select the components. Nexus mode enables both networks. Asset `bus` fields prevent inadvertently installing the configured capacity at every bus. Empty PV availability means zero PV; empty import tariffs mean 0.10 per kWh. The voltage base comes from the DSS circuit, not an unrelated configuration default.

`src.validation` checks algebraic residuals and replays pump commands and scheduled electrical injections in the nonlinear simulators. It reports pressure, flow, tank, voltage, and line-loading differences separately from MILP feasibility. A MILP-feasible schedule is not automatically feasible in nonlinear physics: within-hour hydraulic behavior and terminal reserve differences require review. Unsupported network components are rejected where identified; the supplied small case does not validate every legacy valve/control feature.

## Outputs and tests

A normal workflow run creates a timestamped directory containing `water_data/`, `energy_data/`, `00_summary.json`, and `metadata/`. Battery tables include the terminal boundary; water head tables include the terminal tank head. Failed workflows save `00_failure.json` and input/run metadata rather than reporting successful completion.

```sh
python -m pytest -q
```

Tests check conservation, final-hour accounting, efficiency, pump coupling, feeder transfer and limits, parsing, result precision, metadata, and command-line behavior. They do not depend on a Gurobi license. The older `tests/helpers/validation_utils.py` interface is retained only for historical simulation comparisons; `src.validation` and `src.experiments` are the current replay workflow.

## Files

| Location | Purpose |
|---|---|
| `src/algorithm_tasks.py` | MILP construction and solver handling |
| `src/preprocessing.py` | EPANET calibration and actual DSS network data |
| `src/postprocessing.py` | Complete solution tables and run metadata |
| `src/workflow.py` | Configurable command-line pipeline |
| `src/experiments.py`, `src/validation.py` | Paper experiments and independent replay |
| `data/inputs/paper_case.yaml` | Reproducible illustrative study |
| `paper/main.tex`, `paper/references.bib` | Manuscript and bibliography |
| `doc/implementation.md` | Implementation assumptions and verification notes |
