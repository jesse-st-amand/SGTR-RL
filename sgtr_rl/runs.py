"""Unified run directory management for SGTR-RL experiments."""

import shutil
from datetime import datetime
from pathlib import Path

import yaml

from sgtr_rl.config import TrainingConfig

BASE_DIR = Path("results")

# Fields worth tracking in the overrides string
TRACKED_FIELDS = [
    "learning_rate",
    "num_epochs",
    "per_device_train_batch_size",
    "num_rollouts_per_prompt",
    "max_completion_length",
    "lora_rank",
    "seed",
]


def make_run_name(experiment_name: str, overrides_str: str, timestamp: str) -> str:
    """Build run directory name from components.

    Format: {experiment_name}__{overrides}__{timestamp}
    Omits the overrides segment when empty.
    """
    if overrides_str:
        return f"{experiment_name}__{overrides_str}__{timestamp}"
    return f"{experiment_name}__{timestamp}"


def compute_overrides(config: TrainingConfig, yaml_path: str | Path) -> str:
    """Compare config against the original YAML to find CLI-overridden fields.

    Returns a compact string like ``lr=1e-4,rollouts=16`` for fields that differ
    from the YAML defaults. Only tracks fields listed in TRACKED_FIELDS.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        return ""

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    hp = raw.get("hyperparameters", {})
    model_cfg = raw.get("model", {})

    # Map tracked field names to their YAML values
    yaml_values = {
        "learning_rate": hp.get("learning_rate"),
        "num_epochs": hp.get("num_epochs"),
        "per_device_train_batch_size": hp.get("per_device_train_batch_size"),
        "num_rollouts_per_prompt": hp.get("num_rollouts_per_prompt"),
        "max_completion_length": hp.get("max_completion_length"),
        "lora_rank": model_cfg.get("lora_rank"),
        "seed": hp.get("seed"),
    }

    # Short names for the overrides string
    short_names = {
        "learning_rate": "lr",
        "num_epochs": "epochs",
        "per_device_train_batch_size": "bs",
        "num_rollouts_per_prompt": "rollouts",
        "max_completion_length": "max_len",
        "lora_rank": "rank",
        "seed": "seed",
    }

    parts = []
    for field_name in TRACKED_FIELDS:
        yaml_val = yaml_values.get(field_name)
        config_val = getattr(config, field_name, None)
        if yaml_val is not None and config_val != yaml_val:
            parts.append(f"{short_names[field_name]}={config_val}")

    return ",".join(parts)


def create_run_dir(
    config: TrainingConfig,
    yaml_path: str | Path,
    group: str | None = None,
    exists: str = "error",
) -> Path:
    """Create a unified run directory for a training run.

    Args:
        config: The training config (possibly with CLI overrides applied).
        yaml_path: Path to the original experiment YAML (for computing overrides).
        group: Optional grouping subdirectory (e.g. "sweep_lr").
        exists: Policy when a run with the same experiment_name already exists
                in the group: "error", "skip", or "overwrite".

    Returns:
        Path to the created run directory. Also sets config.run_dir.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overrides_str = compute_overrides(config, yaml_path)
    run_name = make_run_name(config.experiment_name, overrides_str, timestamp)

    base = BASE_DIR / group if group else BASE_DIR
    run_dir = base / run_name

    # Check for existing runs with the same experiment_name in this group
    if exists != "new":
        existing = _find_existing_run(base, config.experiment_name)
        if existing:
            if exists == "error":
                raise FileExistsError(
                    f"Run directory already exists for experiment "
                    f"'{config.experiment_name}' in {base}: {existing}. "
                    f"Use --exists=new, --exists=skip, or --exists=overwrite."
                )
            elif exists == "skip":
                config.run_dir = str(existing)
                return existing
            elif exists == "overwrite":
                shutil.rmtree(existing)

    # Create directory structure
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir()

    # Freeze config
    yaml_path = Path(yaml_path)
    if yaml_path.exists():
        with open(yaml_path) as f:
            raw_config = yaml.safe_load(f)
        # Overlay any CLI overrides onto the frozen config
        for field_name in TRACKED_FIELDS:
            config_val = getattr(config, field_name, None)
            if field_name in ("lora_rank",):
                raw_config.setdefault("model", {})[field_name] = config_val
            else:
                raw_config.setdefault("hyperparameters", {})[field_name] = config_val
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(raw_config, f, default_flow_style=False, sort_keys=False)

    # Copy extraction_meta.json from training data dir if it exists
    train_file = Path(config.train_file)
    meta_path = train_file.parent / "extraction_meta.json"
    if meta_path.exists():
        shutil.copy2(meta_path, run_dir / "extraction_meta.json")

    config.run_dir = str(run_dir)
    return run_dir


def _find_existing_run(base: Path, experiment_name: str) -> Path | None:
    """Find an existing run directory matching the experiment name."""
    if not base.exists():
        return None
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith(experiment_name + "__"):
            return child
    return None


def list_runs(base_dir: str = "results", group: str | None = None) -> list[Path]:
    """List existing run directories, sorted by timestamp (oldest first).

    Args:
        base_dir: Top-level results directory.
        group: Optional group subdirectory to list within.

    Returns:
        Sorted list of run directory paths.
    """
    base = Path(base_dir)
    if group:
        base = base / group
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.name)
