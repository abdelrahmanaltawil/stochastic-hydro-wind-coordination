"""EcoNex Workflow — single entry point.

Orchestrates the full pipeline:
  1. Build network data (water + energy as configured)
  2. Build and solve the Pyomo model
  3. Extract and save results

All parameters are read from data/inputs/config.yaml — no CLI arguments needed.
Set config flags to control what is built:
  run_water: true   → run water sub-model (W1–W24)
  run_energy: false → run energy sub-model (E1–E13)   [default: false until implemented]
  run_nexus: false  → run coupled model               [default: false until nexus is built]
"""

# imports
import logging
import sys
import yaml

# local imports
from helpers import utils
import preprocessing, algorithm_tasks as algorithm, postprocessing 


if __name__ == "__main__":

    # load parameters
    with open("data/inputs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # utilities
    run_dir = utils.create_save_dir(
        base_path=config["paths"]["output_dir"],
        config=config
    )

    # logging and metadata collection
    utils.setup_run_logging(save_path=run_dir / "metadata" / "00_run.log")
    metadata = utils.collect_run_metadata(save_path=run_dir)

    logging.info(f"Starting optimization of {run_dir.name.split(' -- ')[0]} for {config['T']} hours")
 
    # preprocessing
    networks = preprocessing.load_networks(config)

    # build model
    model = algorithm.build_model(networks, config)

    # solve model
    model, solver_results, solver_log_path = algorithm.solve_model(
        model,
        solver=config["solver"]["name"],
        timeout=config["solver"]["timeout"],
        logfile=str(run_dir / "metadata" / "00_solver.log"),
    )

    # extract solution
    solution = postprocessing.extract_solution(model)

    # postprocessing
    summary = postprocessing.create_summary(
        run_id=run_dir.name,
        model=model,
        results=solution,
        solver_results=solver_results
    )

    # save results
    postprocessing.save_data(
        datasets={
            "water_data/00_flows.csv":       solution["water"]["flows"] if config["run_water"] else None,
            "water_data/00_heads.csv":       solution["water"]["heads"] if config["run_water"] else None,
            "water_data/00_pump_status.csv": solution["water"]["pump_status"] if config["run_water"] else None,
            "water_data/00_slack.csv":       solution["water"]["slack"] if config["run_water"] else None,
            "energy_data/00_dispatch.csv":   solution["energy"]["dispatch"] if config["run_energy"] else None,
            "energy_data/00_soc.csv":        solution["energy"]["soc"] if config["run_energy"] else None,
            "energy_data/00_line_flows.csv": solution["energy"]["line_flows"] if config["run_energy"] else None,
            "energy_data/00_voltages.csv":   solution["energy"]["voltages"] if config["run_energy"] else None,
            "00_summary.json":               summary,
        },
        save_path=run_dir,
    )

    # save run metadata
    metadata = postprocessing.save_run_metadata(
                        save_path=run_dir / "metadata",
                        metadata=metadata,
                        experiment_parameters=config,
                        network_files=networks,
                        solver_log_path=solver_log_path,
                        logger=logging.getLogger()
                        )

    logging.info(f"Pipeline completed successfully. Elapsed time: {metadata['execution_duration_min']:.2f} min. Results in: {run_dir}")
