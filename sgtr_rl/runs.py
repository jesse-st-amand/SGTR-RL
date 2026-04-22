"""Unified run directory management for SGTR-RL experiments."""

import shutil
from datetime import datetime
from pathlib import Path

import yaml

from sgtr_rl.config import TrainingConfig

BASE_DIR = Path("results")

TRACKED_FIELDS = [
    {"name": "learning_rate", "section": "hyperparameters", "short": "lr"},
    {"name": "num_epochs", "section": "hyperparameters", "short": "epochs"},
    {"name": "batch_size", "section": "hyperparameters", "short": "bs"},
    {"name": "max_steps", "section": "hyperparameters", "short": "steps"},
    {
        "name": "num_rollouts_per_prompt",
        "section": "hyperparameters",
        "short": "rollouts",
    },
    {"name": "max_completion_length", "section": "hyperparameters", "short": "max_len"},
    {"name": "lora_rank", "section": "model", "short": "rank"},
    {"name": "seed", "section": "hyperparameters", "short": "seed"},
    {"name": "max_train_ids", "section": "data", "short": "train_ids"},
    {"name": "subset_seed", "section": "data", "short": "subset_seed"},
    {
        "name": "randomize_train_labels",
        "section": "data",
        "short": "rand_labels",
    },
    {
        "name": "randomize_train_labels_seed",
        "section": "data",
        "short": "label_seed",
    },
    {"name": "eval_trigger", "section": "evaluation", "short": "eval"},
    {"name": "eval_frequency", "section": "evaluation", "short": "eval_freq"},
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

    defaults = TrainingConfig()

    parts = []
    for spec in TRACKED_FIELDS:
        field_name = spec["name"]
        section = spec["section"]
        yaml_val = raw.get(section, {}).get(field_name, getattr(defaults, field_name))
        config_val = getattr(config, field_name, None)
        if config_val != yaml_val:
            parts.append(f"{spec['short']}={config_val}")

    return ",".join(parts)


def create_run_dir(
    config: TrainingConfig,
    yaml_path: str | Path,
    group: str | None = None,
    exists: str = "error",
    base_dir: str | Path | None = None,
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

    base = Path(base_dir) if base_dir is not None else BASE_DIR
    if group:
        base = base / group
    run_dir = base / run_name

    if exists != "new":
        existing = _find_existing_run(base, config.experiment_name, overrides_str)
        if existing:
            if exists == "error":
                raise FileExistsError(
                    f"Run directory already exists for experiment "
                    f"'{config.experiment_name}' in {base}: {existing}. "
                    f"Use --exists=new, --exists=skip, or --exists=overwrite."
                )
            elif exists == "skip":
                config.run_dir = str(existing)
                config.skipped_existing_run = True
                return existing
            elif exists == "overwrite":
                shutil.rmtree(existing)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir()

    yaml_path = Path(yaml_path)
    if yaml_path.exists():
        with open(yaml_path) as f:
            raw_config = yaml.safe_load(f)
        for spec in TRACKED_FIELDS:
            field_name = spec["name"]
            section = spec["section"]
            config_val = getattr(config, field_name, None)
            raw_config.setdefault(section, {})[field_name] = config_val
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(raw_config, f, default_flow_style=False, sort_keys=False)

    _write_run_metadata(config, run_dir)

    config.run_dir = str(run_dir)
    config.skipped_existing_run = False
    return run_dir


def _find_existing_run(base: Path, experiment_name: str, overrides_str: str = "") -> Path | None:
    """Find an existing run directory matching the experiment name and overrides."""
    if not base.exists():
        return None
    for child in base.iterdir():
        if not child.is_dir():
            continue
        parts = child.name.split("__")
        if not parts or parts[0] != experiment_name:
            continue
        if overrides_str:
            if len(parts) >= 3 and parts[1] == overrides_str:
                return child
            continue
        if len(parts) == 2:
            return child
    return None


def list_runs(base_dir: str = "results", group: str | None = None) -> list[Path]:
    """List existing run directories, sorted by timestamp (oldest first)."""
    base = Path(base_dir)
    if group:
        base = base / group
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.name)


def _write_run_metadata(config: TrainingConfig, run_dir: Path) -> None:
    """Copy or synthesize metadata.json for the run directory."""
    train_files = config.train_files or ([config.train_file] if config.train_file else [])
    if not train_files:
        return

    metadata_entries = []
    for train_file in train_files:
        meta_path = Path(train_file).parent / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            metadata_entries.append(yaml.safe_load(f) or {})

    if not metadata_entries:
        return

    output_path = run_dir / "metadata.json"
    if len(metadata_entries) == 1:
        shutil.copy2(Path(train_files[0]).parent / "metadata.json", output_path)
        return

    combined = {
        "combined_sources": True,
        "train_files": train_files,
        "source_metadata": metadata_entries,
    }
    with open(output_path, "w") as f:
        yaml.safe_dump(combined, f, sort_keys=False)
