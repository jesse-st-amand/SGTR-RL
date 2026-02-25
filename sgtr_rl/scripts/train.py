"""Run SGTR-RL training.

Usage:
    python -m sgtr_rl.scripts.train \
        --config experiments/02_RL_grpo_IND_ShareGPT/config.yaml \
        [--backend local|tinker] \
        [--group sweep_lr] \
        [--exists error|skip|overwrite] \
        [--resume_from_checkpoint path]
"""

import argparse
import logging

from dotenv import load_dotenv

from sgtr_rl.training.train_config import load_training_config
from sgtr_rl.training.logging_setup import setup_logging
from sgtr_rl.training.run_dir import create_run_dir
from sgtr_rl.training.grpo_trainer import LocalGRPOTrainer, TinkerRLTrainer
from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

logger = logging.getLogger(__name__)

TRAINERS = {
    ("grpo", "local"): LocalGRPOTrainer,
    ("grpo", "tinker"): TinkerRLTrainer,
    ("sft", "tinker"): TinkerSFTTrainer,
}


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="SGTR-RL training")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--backend", default=None, help="Override backend (local|tinker)")
    parser.add_argument("--group", default=None, help="Group subdirectory for run (e.g. sweep_lr)")
    parser.add_argument(
        "--exists",
        default="error",
        choices=["error", "skip", "overwrite"],
        help="Policy when a run already exists (default: error)",
    )
    parser.add_argument(
        "--resume_from_checkpoint", default=None, help="Path to checkpoint to resume from"
    )

    # CLI overrides for tracked hyperparameters
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    config = load_training_config(args.config)

    if args.backend:
        config.backend = args.backend

    # Apply CLI hyperparameter overrides
    for field in [
        "learning_rate", "num_epochs", "per_device_train_batch_size",
        "num_rollouts_per_prompt", "max_completion_length", "lora_rank", "seed",
    ]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(config, field, val)

    # Create unified run directory
    run_dir = create_run_dir(config, args.config, group=args.group, exists=args.exists)

    # Set up logging into the run directory
    setup_logging(config.experiment_name, log_file=run_dir / "train.log")

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Config: algorithm={config.algorithm}, backend={config.backend}")
    logger.info(f"Model: {config.model_name} (LoRA rank={config.lora_rank})")
    logger.info(f"Data: train={config.train_file}, val={config.val_file}")
    logger.info(
        f"Hyperparameters: lr={config.learning_rate}, epochs={config.num_epochs}, "
        f"batch_size={config.per_device_train_batch_size}, "
        f"rollouts={config.num_rollouts_per_prompt}, "
        f"max_completion={config.max_completion_length}"
    )

    key = (config.algorithm, config.backend)
    trainer_cls = TRAINERS.get(key)
    if trainer_cls is None:
        raise ValueError(
            f"No trainer for algorithm={config.algorithm!r}, backend={config.backend!r}. "
            f"Available: {list(TRAINERS.keys())}"
        )

    logger.info(f"Trainer: {trainer_cls.__name__}")
    trainer = trainer_cls(config)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
