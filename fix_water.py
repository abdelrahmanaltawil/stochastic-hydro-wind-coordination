import nbformat
import sys

nb_path = "notebooks/water_visualization.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Fix cell 3 (index 3)
nb.cells[3].source = """# Find most recent simulation run
results_dir = project_root / 'data' / 'results'

if results_dir.exists() and results_dir.is_dir():
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and ('WATER' in d.name or 'NEXUS' in d.name)], reverse=True)
    if run_dirs:
        run_dir = run_dirs[0]
        print(f"Loading results from: {run_dir.name}")
        
        # Check available files
        files = list((run_dir / 'water_data').glob('*.csv')) if (run_dir / 'water_data').exists() else []
        print("Found files:", [f.name for f in files])
    else:
        print("No optimization runs found.")
        run_dir = None
else:
    print(f"Results directory not found: {results_dir}")
    run_dir = None"""

# Fix cell 5 (index 5)
nb.cells[5].source = """if run_dir:
    # Try to find the input file in the run directory first (as copied by workflow)
    inp_files = list((run_dir / "metadata" / "inputs" / "water").glob("*.inp"))
    if inp_files:
        inp_file_path = inp_files[0]
        print(f"Loading network from: {inp_file_path}")
        wn = wntr.network.WaterNetworkModel(str(inp_file_path))
    
        print(f"Nodes: {wn.num_nodes}")
        print(f"Links: {wn.num_links}")
        print(f"Pumps: {wn.num_pumps}")
        print(f"Valves: {wn.num_valves}")
        print(f"Tanks: {wn.num_tanks}")
    else:
        print("No INP file found in run directory. Check if saving is enabled.")
        wn = None"""

# Fix cell 10 (index 10)
nb.cells[10].source = """def load_and_pivot(filepath, value_col, index_col, columns_col):
    \"\"\"Load CSV strings and pivot to wide format.\"\"\"
    if filepath.exists():
        try:
            df = pd.read_csv(filepath)
            if not df.empty:
                return df.pivot(index=index_col, columns=columns_col, values=value_col)
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")
    return None

if run_dir:
    # Load and pivot CSVs for easy plotting (Index=Time, Columns=ID)
    df_flows = load_and_pivot(run_dir / 'water_data' / '00_flows.csv', 'flow_rate', 'time', 'link')
    if df_flows is not None:
        print(f"Loaded Flows: {df_flows.shape}")
        
    df_heads = load_and_pivot(run_dir / 'water_data' / '00_heads.csv', 'head', 'time', 'node')
    if df_heads is not None:
        print(f"Loaded Heads: {df_heads.shape}")
        
    df_status = load_and_pivot(run_dir / 'water_data' / '00_pump_status.csv', 'status', 'time', 'pump')
    if df_status is not None:
        print(f"Loaded Pump Status: {df_status.shape}")
        
    # Load Metadata
    try:
        with open(run_dir / '00_summary.json') as f:
            summary = json.load(f)
        print(f"Objective Value: {summary.get('objective_value', 'N/A')}")
    except FileNotFoundError:
        print("00_summary.json not found")"""

# Fix cell 12 (index 12)
nb.cells[12].source = """if df_heads is not None:
    plt.figure(figsize=(14, 6))
    # Plot a subset of nodes if too many
    cols_to_plot = df_heads.columns[:10]  
    df_heads[cols_to_plot].plot(ax=plt.gca(), marker='o')
    plt.title('Nodal Heads over Day-Ahead Horizon')
    plt.ylabel('Head (m)')
    plt.xlabel('Hour of Day')
    plt.xticks(range(24))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""

# Fix cell 14 (index 14)
nb.cells[14].source = """if df_status is not None:
    plt.figure(figsize=(14, 4))
    sns.heatmap(df_status.T, cmap='Greens', cbar_kws={'label': 'ON/OFF'}, linewidths=0.1)
    plt.title('Day-Ahead Pump Schedule Optimization')
    plt.xlabel('Hour of Day')
    plt.ylabel('Pump ID')
    plt.xticks(ticks=np.arange(24) + 0.5, labels=np.arange(24), rotation=0)
    plt.tight_layout()
    plt.show()"""

# Fix cell 16 (index 16)
nb.cells[16].source = """if df_flows is not None:
    plt.figure(figsize=(14, 6))
    # Plot a subset of links if too many
    cols_to_plot = df_flows.columns[:10]  
    df_flows[cols_to_plot].plot(ax=plt.gca(), marker='s')
    plt.title('Link Flows over Day-Ahead Horizon')
    plt.ylabel('Flow Rate (GPM)')
    plt.xlabel('Hour of Day')
    plt.xticks(range(24))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Link ID')
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""

with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Fixed water_visualization.ipynb")
