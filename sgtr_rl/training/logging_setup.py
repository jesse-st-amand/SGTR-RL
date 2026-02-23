"""Logging configuration for SGTR-RL training."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(experiment_name: str, log_dir: str = "logs") -> Path:
    """Configure logging to both terminal and a log file.

    Args:
        experiment_name: Used to name the log file.
        log_dir: Directory for log files.

    Returns:
        Path to the log file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.log"

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Clear any existing handlers
    root.handlers.clear()

    # Terminal handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

    logging.getLogger("sgtr_rl").setLevel(logging.DEBUG)

    logging.info(f"Logging to {log_file}")
    return log_file
