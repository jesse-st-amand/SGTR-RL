"""Data path configuration and resolution.

This module handles finding cached generations from self-rec-framework
and other data directories. It checks multiple locations in order:
1. Path specified in config/data_paths.yaml
2. Default location: data/cached_generations (typically a symlink)
3. Raises error if not found

Usage:
    from sgtr_rl.config import get_data_path, get_dataset_path

    # Get base data path
    data_path = get_data_path()

    # Get specific dataset path
    wikisum_path = get_dataset_path("wikisum")
"""

from pathlib import Path
from typing import Optional
import yaml


class DataPathError(Exception):
    """Raised when data path cannot be found or accessed."""

    pass


def get_project_root() -> Path:
    """Get the project root directory (where pyproject.toml is located)."""
    # Start from this file and go up until we find pyproject.toml
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: assume we're in sgtr_rl/config, so go up 2 levels
    return Path(__file__).resolve().parents[2]


def load_config() -> dict:
    """Load data paths configuration from config/data_paths.yaml.

    Returns:
        dict: Configuration dictionary, or empty dict if file doesn't exist.
    """
    project_root = get_project_root()
    config_file = project_root / "config" / "data_paths.yaml"

    if not config_file.exists():
        return {}

    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
            return config if config else {}
    except Exception as e:
        raise DataPathError(f"Failed to load config file {config_file}: {e}")


def get_data_path() -> Path:
    """Get the base path for cached generations data.

    Checks locations in order:
    1. Path from config/data_paths.yaml (cached_generations_path)
    2. Default location: data/cached_generations

    Returns:
        Path: Absolute path to cached generations directory.

    Raises:
        DataPathError: If data path cannot be found or accessed.

    Example:
        >>> data_path = get_data_path()
        >>> print(data_path)
        /home/user/SGTR-RL/data/cached_generations
    """
    project_root = get_project_root()

    # Try config file first
    config = load_config()
    if "cached_generations_path" in config:
        config_path = Path(config["cached_generations_path"])

        # Handle relative paths (relative to project root)
        if not config_path.is_absolute():
            config_path = project_root / config_path

        if config_path.exists():
            return config_path.resolve()
        else:
            raise DataPathError(
                f"Data path specified in config does not exist: {config_path}\n"
                f"Please check config/data_paths.yaml and ensure the path is correct."
            )

    # Try default location: data/cached_generations
    default_path = project_root / "data" / "cached_generations"
    if default_path.exists():
        return default_path.resolve()

    # If we get here, data not found
    raise DataPathError(
        "Cached generations data not found!\n\n"
        "Please set up data access by either:\n"
        "1. Creating a symlink:\n"
        "   ln -s /path/to/self-rec-framework/data/results data/cached_generations\n"
        "2. Copying data to:\n"
        "   data/cached_generations/\n"
        "3. Setting custom path in config/data_paths.yaml:\n"
        "   cached_generations_path: /your/custom/path\n\n"
        f"Project root: {project_root}\n"
        f"Expected default location: {default_path}\n"
        f"See data/README.md for detailed setup instructions."
    )


def get_dataset_path(dataset_name: str, check_exists: bool = True) -> Path:
    """Get path to a specific dataset within cached generations.

    Args:
        dataset_name: Name of dataset (e.g., "wikisum", "bigcodebench")
        check_exists: If True, raise error if dataset doesn't exist

    Returns:
        Path: Absolute path to dataset directory.

    Raises:
        DataPathError: If dataset path cannot be found (when check_exists=True).

    Example:
        >>> wikisum_path = get_dataset_path("wikisum")
        >>> print(wikisum_path)
        /home/user/SGTR-RL/data/cached_generations/wikisum
    """
    # Check if there's a dataset-specific override in config
    config = load_config()
    if "datasets" in config and dataset_name in config["datasets"]:
        dataset_path = Path(config["datasets"][dataset_name])

        # Handle relative paths
        if not dataset_path.is_absolute():
            project_root = get_project_root()
            dataset_path = project_root / dataset_path

        if check_exists and not dataset_path.exists():
            raise DataPathError(
                f"Dataset path specified in config does not exist: {dataset_path}\n"
                f"Dataset: {dataset_name}\n"
                f"Please check config/data_paths.yaml"
            )

        return dataset_path.resolve()

    # Default: dataset_name subdirectory within cached_generations
    data_path = get_data_path()
    dataset_path = data_path / dataset_name

    if check_exists and not dataset_path.exists():
        raise DataPathError(
            f"Dataset not found: {dataset_name}\n"
            f"Expected location: {dataset_path}\n"
            f"Available datasets: {list_available_datasets()}\n\n"
            "If this dataset should exist, please check:\n"
            "1. Symlink is correctly set up (ls -la data/cached_generations)\n"
            "2. Dataset exists in self-rec-framework/data/results\n"
            "3. Dataset name spelling is correct"
        )

    return dataset_path.resolve()


def list_available_datasets() -> list[str]:
    """List available datasets in cached_generations directory.

    Returns:
        list[str]: List of dataset names (subdirectory names).

    Example:
        >>> datasets = list_available_datasets()
        >>> print(datasets)
        ['wikisum', 'bigcodebench', 'pku_saferlhf', 'sharegpt']
    """
    try:
        data_path = get_data_path()
        # List subdirectories only
        datasets = [d.name for d in data_path.iterdir() if d.is_dir()]
        return sorted(datasets)
    except DataPathError:
        return []


def get_training_data_path(create: bool = True) -> Path:
    """Get path to training_data directory (for processed DPO triples).

    Args:
        create: If True, create directory if it doesn't exist.

    Returns:
        Path: Absolute path to training_data directory.
    """
    project_root = get_project_root()
    training_path = project_root / "data" / "training_data"

    if create:
        training_path.mkdir(parents=True, exist_ok=True)

    return training_path


def get_checkpoints_path(create: bool = True) -> Path:
    """Get path to checkpoints directory.

    Args:
        create: If True, create directory if it doesn't exist.

    Returns:
        Path: Absolute path to checkpoints directory.
    """
    project_root = get_project_root()
    checkpoints_path = project_root / "data" / "checkpoints"

    if create:
        checkpoints_path.mkdir(parents=True, exist_ok=True)

    return checkpoints_path


def get_results_path(create: bool = True) -> Path:
    """Get path to results directory.

    Args:
        create: If True, create directory if it doesn't exist.

    Returns:
        Path: Absolute path to results directory.
    """
    project_root = get_project_root()
    results_path = project_root / "data" / "results"

    if create:
        results_path.mkdir(parents=True, exist_ok=True)

    return results_path


def verify_data_setup() -> dict:
    """Verify data setup and return status information.

    Returns:
        dict: Status information including paths and available datasets.

    Example:
        >>> status = verify_data_setup()
        >>> print(f"Data found: {status['data_found']}")
        >>> print(f"Datasets: {status['datasets']}")
    """
    status = {
        "project_root": str(get_project_root()),
        "config_file_exists": (get_project_root() / "config" / "data_paths.yaml").exists(),
        "data_found": False,
        "data_path": None,
        "datasets": [],
        "errors": [],
    }

    try:
        data_path = get_data_path()
        status["data_found"] = True
        status["data_path"] = str(data_path)
        status["datasets"] = list_available_datasets()
    except DataPathError as e:
        status["errors"].append(str(e))

    return status
