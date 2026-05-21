import nbformat
import sys

nb_path = "notebooks/energy_visualization.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Replace cell 1 (index 1)
nb.cells[1].source = """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

# Add src to path
project_root = Path().resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    plt.style.use(['science', 'ieee', 'grid'])
except:
    plt.style.use('ggplot')

# ── Point to the most recent run ───────────────────────────────────────────
results_dir = project_root / 'data' / 'results'
if results_dir.exists() and results_dir.is_dir():
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and ('ENERGY' in d.name or 'NEXUS' in d.name)], reverse=True)
    if run_dirs:
        run_dir = run_dirs[0]
        print(f"Loading results from: {run_dir.name}")
        energy_dir = run_dir / 'energy_data'
        try:
            with open(run_dir / '00_summary.json') as f:
                summary = json.load(f)
            print(f"Objective Value: {summary.get('objective_value', 'N/A')}")
        except FileNotFoundError:
            print("00_summary.json not found")
    else:
        print("No energy optimization runs found.")
        run_dir = None
else:
    print(f"Results directory not found: {results_dir}")
    run_dir = None"""

# Replace cell 2 (index 2)
nb.cells[2].source = """# ── Day-Ahead Dispatch ──────────────────────────────────────────────────────
if run_dir and (energy_dir / '00_dispatch.csv').exists():
    dispatch = pd.read_csv(energy_dir / '00_dispatch.csv')
    
    # Pivot to get time on index and bus on columns
    # We'll plot total import, export, PV, and storage dispatch over time
    agg_dispatch = dispatch.groupby('time').sum().reset_index()
    
    plt.figure(figsize=(14, 6))
    
    # Plot components
    plt.plot(agg_dispatch['time'], agg_dispatch['P_pv'], label='PV Generation', marker='o')
    plt.plot(agg_dispatch['time'], agg_dispatch['P_import'], label='Grid Import', marker='^')
    plt.plot(agg_dispatch['time'], agg_dispatch['P_export'], label='Grid Export', marker='v')
    plt.plot(agg_dispatch['time'], agg_dispatch['Q_dis'], label='Storage Discharge', marker='s')
    plt.plot(agg_dispatch['time'], agg_dispatch['Q_ch'], label='Storage Charge', marker='D')
    
    plt.title('Day-Ahead Energy Dispatch Profile')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power (kW)')
    plt.xticks(range(24))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Dispatch data not found.")"""

# Replace cell 3 (index 3)
nb.cells[3].source = """# ── State of charge ─────────────────────────────────────────────────────────
if run_dir and (energy_dir / '00_soc.csv').exists():
    soc = pd.read_csv(energy_dir / '00_soc.csv')
    
    # Pivot to plot SOC for each storage bus
    soc_pivot = soc.pivot(index='time', columns='bus', values='E_soc')
    
    plt.figure(figsize=(14, 6))
    soc_pivot.plot(ax=plt.gca(), marker='o', linestyle='-')
    plt.title('Day-Ahead State of Charge (SOC) Trajectory')
    plt.xlabel('Hour of Day')
    plt.ylabel('Stored Energy (kWh)')
    plt.xticks(range(24))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Bus')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("SOC data not found.")"""

# Replace cell 4 (index 4)
nb.cells[4].source = """# ── Voltages ────────────────────────────────────────────────────────────────
if run_dir and (energy_dir / '00_voltages.csv').exists():
    voltages = pd.read_csv(energy_dir / '00_voltages.csv')
    
    v_pivot = voltages.pivot(index='time', columns='bus', values='U')
    
    plt.figure(figsize=(14, 6))
    # Plotting a subset if many buses exist
    cols_to_plot = v_pivot.columns[:10]
    v_pivot[cols_to_plot].plot(ax=plt.gca(), marker='.')
    plt.axhline(y=1.1, color='r', linestyle='--', alpha=0.5, label='Upper Limit (1.1 p.u.)')
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Lower Limit (0.9 p.u.)')
    
    plt.title('Day-Ahead Bus Voltages Profile')
    plt.xlabel('Hour of Day')
    plt.ylabel('Voltage (p.u.)')
    plt.xticks(range(24))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Bus')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Voltage data not found.")"""

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Fixed energy_visualization.ipynb")
