# Implemented model and verification

The executable specification is `src/algorithm_tasks.py`, with matching equations in `paper/main.tex`. Earlier files in `doc/capsules/` and the original theoretical-background notes describe the development history and can include broader, unimplemented features.

## Conservation and storage

The model uses hourly flows/powers at indices `0..T-1`. `H[tank,t]` is the tank head at the start of interval `t`; `H_terminal[tank]` represents boundary `T`. Battery `E_soc[bus,t]` is defined on `0..T`. Tank and battery inventory updates therefore account for every decision, including the last hour. Optimization runs default to terminal states at least equal to initial states. The water-only tangent diagnostic can disable terminal closure to compare a supplied simulation trajectory.

The water pressure floor is `water.min_pressure_m`. Strict mass balance fixes demand slack to zero unless diagnostic relaxation is explicitly enabled. The paper case uses a 20 m floor. Pipe flow domains are derived from twice the pre-simulation peak absolute flow, with a minimum domain; `water.n_pipe_segments` controls interpolation density. These are modeling bounds, not proven envelopes of every physically possible schedule.

Battery charge and discharge are measured at the AC bus. Charging increases stored energy by efficiency times bus power; discharging removes bus power divided by efficiency. The electrical bus balance does not apply those factors a second time.

## Electrical approximation and network data

Preprocessing reads the DSS load placement, nominal real/reactive powers, daily multipliers, line impedances, ampacities, and voltage base. It assigns configured PV and battery capacities to their selected buses. The original example is a lightly loaded two-bus 115 kV circuit; it is an illustrative input and does not establish performance on congested distribution feeders.

Active and reactive line flows use a flat-voltage first-order series model with fixed topology. Shunt admittance, series losses, unbalance, transformers and their controls are outside this formulation. Real/reactive line current components are approximated from line power at the voltage expansion point. Piecewise-linear overestimates of both squared components are summed for the thermal constraint; no quadratic constraint remains in the MILP.

## Pump coupling and objective

A shared EPANET pre-simulation estimates each pump's mean electrical draw during running hours as `9.81 * mean(Q * head_gain) / efficiency` in kW. Pump status multiplies that fixed coefficient in the mapped bus load. The water-only model prices on-power using a uniform tariff. In nexus mode pump power is paid through grid procurement, and its standalone cost is removed to avoid double counting.

A constant power per on-hour approximates the physical dependence on both head and flow. Hourly hydraulic inventory updates also approximate continuous tank evolution. Nonlinear replay is required to measure these effects; a solver's optimal termination applies to the MILP approximation alone.

## Reproducible evidence

`python -m src.experiments` records each baseline/coupled solve and its actual pump schedule. It checks constraint, variable-bound and integrality residuals before accepting an incumbent, replays the independent and joint schedules, and generates numerical manuscript content directly from the saved values. `provenance.json` identifies dependencies and SHA256 hashes of network/source files. The identical water model, initial inventories, demands and terminal requirements are used in the comparison.

EPANET replay uses a five-minute hydraulic step with hourly reporting and imposed pump commands. OpenDSS replay uses scheduled net active/reactive injections as constant-power snapshots. Results report approximation errors and reserve discrepancies; they are not field validation. Equal-cost water-only schedules may produce different baseline electricity bills. Time-limited solves are explicitly distinguished from proven optima.
