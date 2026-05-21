# EcoNex Optimization & Simulation Project

This project contains a stochastic model for the daily coordination of pumped storage hydro plants and wind power plants.

## Project Structure

```
├── src/
│   ├── algorithm_tasks.py    # Model formulation & solving
│   ├── preprocessing.py
│   ├── postprocessing.py
│   ├── workflow.py           # Single entry point
│   └── helpers/
│       ├── utils.py
│       ├── energy/
│       └── water/
├── data/
│   ├── inputs/               # Config and network definitions
│   └── results/              # Output from workflow
├── notebooks/
│   ├── energy_visualization.ipynb
│   └── water_visualization.ipynb
├── doc/
│   ├── capsules/
│   └── theoretical_background.md
└── tests/
```

## Usage

### Execution
To run the full pipeline (data build, pyomo optimization, and extraction):
```bash
python src/workflow.py
```
> **Note**: This model uses complex hydraulic constraints (Piecewise Linear). While it is configured to run with `glpk` by default, a commercial solver like **Gurobi** or **CPLEX** is strongly recommended for production use to ensure convergence and performance.

## Installation
```bash
pip install -r requirements.txt
```
