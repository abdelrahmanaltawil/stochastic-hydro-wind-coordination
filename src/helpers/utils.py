
import logging
import sys
import os
import platform
import getpass
import socket
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def create_save_dir(base_path: str, config: dict) -> Path:
    """Creates the results directory structure with timestamp and unique ID."""

    # generate timestamp and run id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]

    # get active components
    active = [
        name 
        for name, flag in [
            ("WATER", config.get("run_water", False)),
            ("ENERGY", config.get("run_energy", False)),
            ("NEXUS", config.get("run_nexus", False))
        ] 
        if flag
    ]
    base_name = "-".join(active)

    # create save directory
    save_dir = Path(base_path).expanduser().resolve() / f"{base_name} -- {timestamp} -- {run_id}"

    save_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata subdirectory
    (save_dir / "metadata" / "inputs").mkdir(parents=True, exist_ok=True)
    
    return save_dir

def setup_run_logging(save_path: Path) -> None:
    """Configures both console and file handlers for the run."""

    class _FormatSolverLogs(logging.Filter):
        def filter(self, record):
            # A generalized check for Pyomo solver logs (e.g. GUROBI_RUN, GLPK_RUN, or pyomo.solver)
            # without hardcoding any specific solver names.
            is_solver_log = (
                record.name.endswith("_RUN") or 
                record.module.endswith("_RUN") or 
                "solver" in record.name.lower() or 
                "solver" in record.module.lower()
            )
        
            if is_solver_log:
                # Prefix the module name so it displays exactly as the user requested
                if not record.module.startswith("algorithm_tasks - "):
                    record.module = f"algorithm_tasks - {record.module}"
            return True

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] \033[1m%(module)s\033[0m - %(message)s")
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(_FormatSolverLogs())

    # file handler
    log_path = Path(save_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(module)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(_FormatSolverLogs())

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True
    )


def collect_run_metadata(save_path: Path) -> dict:
    """Collects run environment and versioning details."""

    metadata = {
        "experiment_id": save_path.parts[-1].split(" -- ")[-1],
        "execution_start_time": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "working_directory": os.getcwd(),
        "command": " ".join(sys.argv),
    }

    logging.info("Collecting run environment and versioning details...")
    logging.info(f"Experiment ID: {metadata['experiment_id']} (started at {metadata['execution_start_time']})")
    logging.info(f"Experiment results will be saved to: {save_path}\n\n")

    return metadata


def get_git_revision_hash() -> str:
    """Retrieve the current git commit hash for traceability.

    Returns:
        The git ref hash or 'unknown'.
    """
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception as e:
        logger.warning(f"Could not retrieve git hash: {e}")
        return "unknown"
