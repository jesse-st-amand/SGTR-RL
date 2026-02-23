"""Training configuration for SGTR-RL experiments."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainingConfig:
    """Configuration for SGTR-RL training runs."""

    # Core
    algorithm: str = "grpo"  # "grpo" | "dpo" | "sft"
    backend: str = "local"  # "local" | "tinker"
    experiment_name: str = ""

    # Model
    model_name: str = "Qwen/Qwen2-1.5B"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    learning_rate: float = 5e-5
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    bf16: bool = True
    seed: int = 42

    # GRPO-specific
    num_rollouts_per_prompt: int = 4  # rollouts per prompt per step
    max_completion_length: int = 1024  # max tokens for generated rollout
    beta: float = 0.1  # KL penalty (if used)

    # Data
    train_file: str = ""
    val_file: str = ""
    output_dir: str = ""

    # Checkpointing
    save_steps: int = 50  # checkpoint every N steps
    eval_steps: int = 50  # run evals every N steps


def load_training_config(yaml_path: str | Path) -> TrainingConfig:
    """Load a TrainingConfig from an experiment YAML file.

    Maps the nested YAML structure to the flat TrainingConfig dataclass.

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

    model_cfg = cfg.get("model", {})
    hp = cfg.get("hyperparameters", {})
    data_cfg = cfg.get("data", {})
    ckpt_cfg = cfg.get("checkpointing", {})
    output_cfg = cfg.get("output", {})

    return TrainingConfig(
        algorithm=cfg.get("algorithm", "grpo"),
        backend=cfg.get("backend", "local"),
        experiment_name=cfg.get("experiment_name", ""),
        # Model
        model_name=model_cfg.get("name", "Qwen/Qwen2-1.5B"),
        lora_rank=model_cfg.get("lora_rank", 32),
        lora_alpha=model_cfg.get("lora_alpha", 64),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        lora_target_modules=model_cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        # Hyperparameters
        learning_rate=hp.get("learning_rate", 5e-5),
        num_epochs=hp.get("num_epochs", 3),
        per_device_train_batch_size=hp.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=hp.get("gradient_accumulation_steps", 4),
        warmup_ratio=hp.get("warmup_ratio", 0.1),
        bf16=hp.get("bf16", True),
        seed=hp.get("seed", 42),
        num_rollouts_per_prompt=hp.get("num_rollouts_per_prompt", 4),
        max_completion_length=hp.get("max_completion_length", 1024),
        beta=hp.get("beta", 0.1),
        # Data
        train_file=data_cfg.get("train_file", ""),
        val_file=data_cfg.get("val_file", ""),
        output_dir=output_cfg.get("checkpoint_dir", ""),
        # Checkpointing
        save_steps=ckpt_cfg.get("save_steps", 50),
        eval_steps=ckpt_cfg.get("eval_steps", 50),
    )
