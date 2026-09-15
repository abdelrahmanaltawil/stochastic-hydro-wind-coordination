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

A shared EPANET pre-simulation estimates each pump's mean electrical draw during running hours as `9.81 * mean(Q * head_gain) / efficiency` in kW, rounded to 10 W (`PUMP_POWER_RESOLUTION_KW`). Rounding matters for solver behaviour: with a rounded coefficient the water operator's objective is a common multiple of one number, which MILP solvers detect (objective integrality) and use to prove optimality far faster on the symmetric flat-tariff problem. Pump status multiplies that fixed coefficient in the mapped bus load. The water-only model prices on-power with a flat tariff (`pump_energy_tariff`), an hourly price series, or a per-pump table (`pump_energy_price`); in nexus mode pump power is paid through grid procurement, and its standalone cost is removed to avoid double counting.

Two formulation tightenings were added with the coordination layer and apply to every solve. Heads are bounded by a network-derived ceiling (highest source head plus the sum of the pump shutoff heads, with a 5 % margin) that also serves as the big-M of the pump and valve disjunctions; and each pump curve is interpolated over the hydraulically admissible flow range, the curve flow at the minimum head gain implied by the discharge node's pressure floor (or tank bottom) and the suction node's head ceiling. Neither changes the feasible set.

## Two-operator coordination layer

`src/coordination.py` builds three kinds of model from the same components. `water_operator` is the water block with the metered pump load priced at an internal price; `energy_operator` is the energy block with a bounded price-responsive accepted load `W_accept[b,t]` in the nodal balance (or, with that load fixed, the energy operator's response to a metered schedule); `prepare` makes the connection ratings explicit so that the subproblems and the integrated model see the same limits. `price_coordination` runs the dual scheme with a bundle master (proximal cutting planes on the polyhedral dual function, serious/null steps), a box cutting-plane master, or Polyak/diminishing subgradient steps; prices are exchanged with finite resolution, the certified dual bound uses the solvers' proven bounds, and the primal is recovered from the energy operator's response to each metered schedule. `restricted_lp_prices` fixes the binaries of an integrated optimum and reads the nodal balance multipliers, which are the internal prices of Proposition 3 of the manuscript. `flat_tariff_baseline` computes the baseline in two stages (minimal pump-hours, then level holding) so that the tie-break is exact. The contract functions (`payoff_table`, `fee_window`, `anchored_fee`, `contract_payoffs`) are pure accounting on the metered settlement.

The Lagrangian dual of the interface has a duality gap with integer pump commitments: the certified gap reported by the coordination runs is not a failure to find the schedule (the recovered schedule is compared with the central solve) but the limit of what the operators can certify from their own data.

A constant power per on-hour approximates the physical dependence on both head and flow. Hourly hydraulic inventory updates also approximate continuous tank evolution. Nonlinear replay is required to measure these effects; a solver's optimal termination applies to the MILP approximation alone.

## Reproducible evidence

`python -m src.experiments` records the integrated benchmark, the flat-tariff baseline, the tariff benchmarks, the contract accounting, the coordination runs (full per-iteration histories in `coordination_*.json`), the connection-rating sweeps, and the replays; `study.json` holds every number the manuscript quotes, and `paper/results_macros.tex` exposes them to the running text. It checks constraint, variable-bound and integrality residuals before accepting an incumbent, replays the independent and joint schedules, and generates numerical manuscript content directly from the saved values. `provenance.json` identifies dependencies and SHA256 hashes of network/source files. The identical water model, initial inventories, demands and terminal requirements are used in the comparison.

EPANET replay uses a five-minute hydraulic step with hourly reporting and imposed pump commands. OpenDSS replay uses scheduled net active/reactive injections as constant-power snapshots. Results report approximation errors and reserve discrepancies; they are not field validation. Equal-cost water-only schedules may produce different baseline electricity bills. Time-limited solves are explicitly distinguished from proven optima.
