"""Training configuration for SGTR-RL experiments."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32


class _HyperparameterSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 5e-5
    num_epochs: int = 20
    batch_size: int = 16
    max_steps: int | None = None
    seed: int = 42
    num_rollouts_per_prompt: int = 4
    max_completion_length: int = 1024
    sampling_temperature: float = 1.0


class _DataSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_file: str = ""
    val_file: str = ""
    train_files: list[str] = []
    val_files: list[str] = []
    train_mix_strategy: Literal["concat", "per_id_one_source"] = "concat"
    max_train_ids: int | None = None
    subset_seed: int | None = None
    randomize_train_labels: bool = False
    randomize_train_labels_seed: int | None = None
    use_system_prompt: bool = False
    generator_models: list[str] = []
    dataset: str = ""


class _EvaluationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger: Literal["epoch", "step"] = "epoch"
    frequency: int = 1
    diagnostic_num_examples: int = 0
    diagnostic_example_ids: list[str] = []
    train_diagnostic_num_examples: int = 0
    train_diagnostic_example_ids: list[str] = []


class BenchmarkEvalConfig(BaseModel):
    """Configuration for a single benchmark evaluation."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    type: Literal["mmlu", "sgtr"] = "mmlu"
    data_file: str = ""
    schedule: str = "every_epoch"  # "every_epoch" | "every_N_epochs" | "end_only"
    frequency: int = 1  # for "every_N_epochs"
    cot: bool = False
    num_samples: int | None = None  # deterministic subsample size (None = use all)


class TrainingConfig(BaseModel):
    """Configuration for SGTR-RL training runs."""

    # Core
    algorithm: Literal["grpo", "sft"] = "sft"
    experiment_name: str = ""

    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    lora_rank: int = 32

    # Training
    learning_rate: float = 5e-5
    num_epochs: int = 20
    batch_size: int = 16
    max_steps: int | None = None
    seed: int = 42

    # GRPO-specific
    num_rollouts_per_prompt: int = 4
    max_completion_length: int = 1024
    sampling_temperature: float = 1.0

    # Data
    train_file: str = ""
    val_file: str = ""
    train_files: list[str] = []
    val_files: list[str] = []
    train_mix_strategy: Literal["concat", "per_id_one_source"] = "concat"
    max_train_ids: int | None = None
    subset_seed: int | None = None
    randomize_train_labels: bool = False
    randomize_train_labels_seed: int | None = None

    # Evaluation scheduling
    eval_trigger: Literal["epoch", "step"] = "epoch"
    eval_frequency: int = 1
    eval_diagnostic_num_examples: int = 0
    eval_diagnostic_example_ids: list[str] = []
    train_diagnostic_num_examples: int = 0
    train_diagnostic_example_ids: list[str] = []

    # If True, prepend system_prompt from training records to conversations
    use_system_prompt: bool = False

    # Run directory (set by runs.create_run_dir)
    run_dir: str = ""

    # Logging
    wandb_project: str | None = None  # W&B project name; None = wandb disabled

    # Benchmark evaluations (e.g. MMLU canary)
    benchmark_evals: list[BenchmarkEvalConfig] = []

    # Runtime bookkeeping
    completed_epochs: int = 0
    skipped_existing_run: bool = False

    def model_post_init(self, __context) -> None:
        if not self.train_files and self.train_file:
            self.train_files = [self.train_file]
        if not self.val_files and self.val_file:
            self.val_files = [self.val_file]


_KNOWN_TOP_KEYS = {
    "experiment_name", "description", "algorithm",
    "model", "hyperparameters", "data",
    "wandb_project", "benchmark_evals", "evaluation",
}


def load_training_config(yaml_path: str | Path) -> TrainingConfig:
    """Load a TrainingConfig from an experiment YAML file.

    Maps the nested YAML structure to the flat TrainingConfig.
    Raises ValueError on unknown keys in any section.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    unknown_top = set(cfg.keys()) - _KNOWN_TOP_KEYS
    if unknown_top:
        raise ValueError(f"Unknown top-level config keys: {unknown_top}")

    # Parse and validate each section (extra="forbid" catches typos)
    model = _ModelSection(**cfg.get("model", {}))
    hp = _HyperparameterSection(**cfg.get("hyperparameters", {}))
    data = _DataSection(**cfg.get("data", {}))
    evaluation = _EvaluationSection(**cfg.get("evaluation", {}))

    # Parse benchmark_evals section
    bench_cfg = cfg.get("benchmark_evals", {})
    benchmark_evals = []
    if bench_cfg:
        for name, bcfg in bench_cfg.items():
            benchmark_evals.append(BenchmarkEvalConfig(name=name, **bcfg))

    train_files = _normalize_data_files(
        singular=data.train_file,
        plural=data.train_files,
        label="train",
    )
    val_files = _normalize_data_files(
        singular=data.val_file,
        plural=data.val_files,
        label="val",
    )

    return TrainingConfig(
        algorithm=cfg.get("algorithm", "sft"),
        experiment_name=cfg.get("experiment_name", ""),
        model_name=model.name,
        lora_rank=model.lora_rank,
        learning_rate=hp.learning_rate,
        num_epochs=hp.num_epochs,
        batch_size=hp.batch_size,
        max_steps=hp.max_steps,
        seed=hp.seed,
        num_rollouts_per_prompt=hp.num_rollouts_per_prompt,
        max_completion_length=hp.max_completion_length,
        sampling_temperature=hp.sampling_temperature,
        train_file=train_files[0] if train_files else "",
        val_file=val_files[0] if val_files else "",
        train_files=train_files,
        val_files=val_files,
        train_mix_strategy=data.train_mix_strategy,
        max_train_ids=data.max_train_ids,
        subset_seed=data.subset_seed,
        randomize_train_labels=data.randomize_train_labels,
        randomize_train_labels_seed=data.randomize_train_labels_seed,
        eval_trigger=evaluation.trigger,
        eval_frequency=evaluation.frequency,
        eval_diagnostic_num_examples=evaluation.diagnostic_num_examples,
        eval_diagnostic_example_ids=evaluation.diagnostic_example_ids,
        train_diagnostic_num_examples=evaluation.train_diagnostic_num_examples,
        train_diagnostic_example_ids=evaluation.train_diagnostic_example_ids,
        use_system_prompt=data.use_system_prompt,
        wandb_project=cfg.get("wandb_project"),
        benchmark_evals=benchmark_evals,
    )


def _normalize_data_files(*, singular: str, plural: list[str], label: str) -> list[str]:
    """Resolve legacy singular and new plural data-file fields."""
    if singular and plural:
        raise ValueError(f"Specify only one of data.{label}_file or data.{label}_files")
    if plural:
        return list(plural)
    if singular:
        return [singular]
    return []
