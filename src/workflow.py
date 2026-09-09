"""Run the deterministic water/energy scheduling pipeline from any directory.

Usage: python -m src.workflow --config data/inputs/config.yaml --solver highs
The direct-script entry point (python src/workflow.py) is also supported.
"""

import argparse
import copy
import logging
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src import algorithm_tasks as algorithm, postprocessing, preprocessing
from src.helpers import utils


def load_config(path=None, *, solver=None, output_dir=None, timeout=None):
    """Read configuration; relative network/output paths use the project root."""
    path = Path(path) if path else PROJECT_ROOT / "data/inputs/config.yaml"
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    config = copy.deepcopy(config)
    for flag, default in (("run_water", True), ("run_energy", False), ("run_nexus", False)):
        config.setdefault(flag, default)
        if not isinstance(config[flag], bool):
            raise ValueError(f"{flag} must be true or false.")
    if config["run_nexus"]:
        config["run_water"] = config["run_energy"] = True
    if not (config["run_water"] or config["run_energy"]):
        raise ValueError("Enable at least one water or energy model.")
    horizon = config.setdefault("T", 24)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("T must be a positive integer number of hours.")
    for domain in ("water", "energy"):
        if config[f"run_{domain}"]:
            network = Path(config[domain]["network"]).expanduser()
            config[domain]["network"] = str(
                (network if network.is_absolute() else PROJECT_ROOT / network).resolve()
            )
    settings = config.setdefault("solver", {})
    settings["name"] = solver or settings.get("name", "highs")
    settings["timeout"] = timeout if timeout is not None else settings.get("timeout", 300)
    if settings["timeout"] <= 0:
        raise ValueError("Solver timeout must be positive.")
    paths = config.setdefault("paths", {})
    output = Path(output_dir or paths.get("output_dir", "data/results")).expanduser()
    paths["output_dir"] = str(
        (output if output.is_absolute() else PROJECT_ROOT / output).resolve()
    )
    return config


def run_pipeline(config):
    """Build, solve, and save one run; raise if no usable solution exists."""
    config = copy.deepcopy(config)
    run_dir = utils.create_save_dir(config["paths"]["output_dir"], config).resolve()
    utils.setup_run_logging(run_dir / "metadata/00_run.log")
    logging.getLogger().setLevel(config.get("logging", {}).get("level", "INFO"))
    metadata = utils.collect_run_metadata(run_dir)
    networks = {}
    solver_log = str(run_dir / "metadata/00_solver.log")
    logging.info("Starting %s-hour schedule in %s", config["T"], run_dir)
    try:
        networks = preprocessing.load_networks(config)
        model = algorithm.build_model(networks, config)
        model, solver_results, solver_log = algorithm.solve_model(
            model, solver=config["solver"]["name"],
            timeout=config["solver"]["timeout"], logfile=solver_log,
        )
        if not len(solver_results.solution):
            raise RuntimeError(
                f"Solver returned no feasible incumbent ({solver_results.solver.termination_condition})."
            )
        solution = postprocessing.extract_solution(model)
        if solution["objective"] is None:
            raise RuntimeError("Solver returned no usable solution.")
        summary = postprocessing.create_summary(run_dir.name, model, solution, solver_results)
        datasets = {"00_summary.json": summary}
        for domain in ("water", "energy"):
            for name, rows in solution.get(domain, {}).items():
                datasets[f"{domain}_data/00_{name}.csv"] = rows
        postprocessing.save_data(datasets, run_dir)
    except Exception as exc:
        postprocessing.save_data({"00_failure.json": {
            "status": "failed", "error_type": type(exc).__name__, "message": str(exc),
        }}, run_dir)
        logging.exception("Pipeline failed; diagnostics saved in %s", run_dir)
        raise
    finally:
        postprocessing.save_run_metadata(
            save_path=run_dir / "metadata", metadata=metadata,
            experiment_parameters=config, network_files=networks,
            solver_log_path=solver_log, logger=logging.getLogger(),
        )
    logging.info("Pipeline completed: %s", run_dir)
    return {"run_dir": run_dir, "model": model, "solution": solution,
            "summary": summary, "solver_results": solver_results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--solver", help="Override the configured Pyomo solver")
    parser.add_argument("--output-dir", type=Path, help="Override the results directory")
    parser.add_argument("--timeout", type=float, help="Solver time limit in seconds")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, solver=args.solver,
                             output_dir=args.output_dir, timeout=args.timeout)
        outcome = run_pipeline(config)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Scheduling failed: {exc}\n")
    print(f"Results: {outcome['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
