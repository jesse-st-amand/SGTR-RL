"""Training configuration for SGTR-RL experiments."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BenchmarkEvalConfig:
    """Configuration for a single benchmark evaluation."""

    name: str = ""
    type: str = "mmlu"  # benchmark type
    data_file: str = ""
    schedule: str = "every_epoch"  # "every_epoch" | "every_N_epochs" | "end_only"
    frequency: int = 1  # for "every_N_epochs"
    cot: bool = False


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
    sampling_temperature: float = 1.0  # temperature for rollout sampling

    # Data
    train_file: str = ""
    val_file: str = ""
    output_dir: str = ""

    # Run directory (set by run_dir.create_run_dir)
    run_dir: str = ""

    # Logging
    wandb_project: str | None = None  # W&B project name; None = wandb disabled

    # Checkpointing
    save_steps: int = 50  # checkpoint every N steps
    eval_steps: int = 50  # run evals every N steps

    # Benchmark evaluations (e.g. MMLU canary)
    benchmark_evals: list[BenchmarkEvalConfig] = field(default_factory=list)


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

    # Parse benchmark_evals section
    bench_cfg = cfg.get("benchmark_evals", {})
    benchmark_evals = []
    if bench_cfg:
        for name, bcfg in bench_cfg.items():
            benchmark_evals.append(BenchmarkEvalConfig(
                name=name,
                type=bcfg.get("type", "mmlu"),
                data_file=bcfg.get("data_file", ""),
                schedule=bcfg.get("schedule", "every_epoch"),
                frequency=bcfg.get("frequency", 1),
                cot=bcfg.get("cot", False),
            ))

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
        sampling_temperature=hp.get("sampling_temperature", 1.0),
        # Data
        train_file=data_cfg.get("train_file", ""),
        val_file=data_cfg.get("val_file", ""),
        # Logging
        wandb_project=cfg.get("wandb_project"),
        # Checkpointing
        save_steps=ckpt_cfg.get("save_steps", 50),
        eval_steps=ckpt_cfg.get("eval_steps", 50),
        # Benchmarks
        benchmark_evals=benchmark_evals,
    )
