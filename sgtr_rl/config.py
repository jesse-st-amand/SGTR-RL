"""Training configuration for SGTR-RL experiments."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# YAML section models (validate structure of each config section)
# ---------------------------------------------------------------------------

class _ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "Qwen/Qwen2-1.5B"
    lora_rank: int = 32


class _HyperparameterSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 5e-5
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    seed: int = 42
    num_rollouts_per_prompt: int = 4
    max_completion_length: int = 1024
    sampling_temperature: float = 1.0


class _DataSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_file: str = ""
    val_file: str = ""
    use_system_prompt: bool = False
    evaluator_model: str = ""
    generator_models: list[str] = []
    dataset: str = ""
    subsets: list[str] = []


# ---------------------------------------------------------------------------
# Runtime config models
# ---------------------------------------------------------------------------

class BenchmarkEvalConfig(BaseModel):
    """Configuration for a single benchmark evaluation."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    type: Literal["mmlu", "sgtr"] = "mmlu"
    data_file: str = ""
    schedule: str = "every_epoch"  # "every_epoch" | "every_N_epochs" | "end_only"
    frequency: int = 1  # for "every_N_epochs"
    cot: bool = False
    flip_targets: bool = False  # swap "1"<->"2" at eval time
    num_samples: int | None = None  # deterministic subsample size (None = use all)


class TrainingConfig(BaseModel):
    """Configuration for SGTR-RL training runs."""

    # Core
    algorithm: Literal["grpo", "sft"] = "grpo"
    experiment_name: str = ""

    # Model
    model_name: str = "Qwen/Qwen2-1.5B"
    lora_rank: int = 32

    # Training
    learning_rate: float = 5e-5
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    seed: int = 42

    # GRPO-specific
    num_rollouts_per_prompt: int = 4
    max_completion_length: int = 1024
    sampling_temperature: float = 1.0

    # Data
    train_file: str = ""
    val_file: str = ""

    # Data field names (configurable for external datasets)
    prompt_field: str = "prompt"
    target_field: str = "target"
    id_field: str = "id"

    # If True, prepend system_prompt from training records to conversations
    use_system_prompt: bool = False

    # Run directory (set by runs.create_run_dir)
    run_dir: str = ""

    # Logging
    wandb_project: str | None = None  # W&B project name; None = wandb disabled

    # Benchmark evaluations (e.g. MMLU canary)
    benchmark_evals: list[BenchmarkEvalConfig] = []


# ---------------------------------------------------------------------------
# Known top-level YAML keys (validated in load_training_config)
# ---------------------------------------------------------------------------

_KNOWN_TOP_KEYS = {
    "experiment_name", "description", "algorithm",
    "model", "hyperparameters", "data",
    "wandb_project", "benchmark_evals", "evaluation",
}


def load_training_config(yaml_path: str | Path) -> TrainingConfig:
    """Load a TrainingConfig from an experiment YAML file.

    Maps the nested YAML structure to the flat TrainingConfig.
    Raises ValueError on unknown keys in any section.

    Args:
        yaml_path: Path to the experiment config YAML.

    Returns:
        Populated TrainingConfig instance.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validate top-level keys
    unknown_top = set(cfg.keys()) - _KNOWN_TOP_KEYS
    if unknown_top:
        raise ValueError(f"Unknown top-level config keys: {unknown_top}")

    # Parse and validate each section (extra="forbid" catches typos)
    model = _ModelSection(**cfg.get("model", {}))
    hp = _HyperparameterSection(**cfg.get("hyperparameters", {}))
    data = _DataSection(**cfg.get("data", {}))

    # Parse benchmark_evals section
    bench_cfg = cfg.get("benchmark_evals", {})
    benchmark_evals = []
    if bench_cfg:
        for name, bcfg in bench_cfg.items():
            benchmark_evals.append(BenchmarkEvalConfig(name=name, **bcfg))

    return TrainingConfig(
        algorithm=cfg.get("algorithm", "grpo"),
        experiment_name=cfg.get("experiment_name", ""),
        # Model
        model_name=model.name,
        lora_rank=model.lora_rank,
        # Hyperparameters
        learning_rate=hp.learning_rate,
        num_epochs=hp.num_epochs,
        per_device_train_batch_size=hp.per_device_train_batch_size,
        seed=hp.seed,
        num_rollouts_per_prompt=hp.num_rollouts_per_prompt,
        max_completion_length=hp.max_completion_length,
        sampling_temperature=hp.sampling_temperature,
        # Data
        train_file=data.train_file,
        val_file=data.val_file,
        use_system_prompt=data.use_system_prompt,
        # Logging
        wandb_project=cfg.get("wandb_project"),
        # Benchmarks
        benchmark_evals=benchmark_evals,
    )
