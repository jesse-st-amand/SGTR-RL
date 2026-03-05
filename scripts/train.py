"""Run SGTR-RL training.

Usage:
    python -m scripts.train \
        --config experiments/15_sft_pw_rec_vs_qwen/config.yaml \
        [--group sweep_lr] \
        [--exists error|skip|overwrite]
"""

import argparse
import logging

from dotenv import load_dotenv

from sgtr_rl.config import load_training_config
from sgtr_rl.logging_setup import setup_logging
from sgtr_rl.pipeline import run_training
from sgtr_rl.runs import create_run_dir

logger = logging.getLogger(__name__)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="SGTR-RL training")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--group", default=None, help="Group subdirectory for run (e.g. sweep_lr)")
    parser.add_argument(
        "--exists",
        default="new",
        choices=["new", "error", "skip", "overwrite"],
        help="Policy when a run already exists (default: new = always create fresh dir)",
    )

    # CLI overrides for tracked hyperparameters
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    config = load_training_config(args.config)

    # Apply CLI hyperparameter overrides
    for field in [
        "learning_rate", "num_epochs", "batch_size",
        "num_rollouts_per_prompt", "max_completion_length", "lora_rank", "seed",
    ]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(config, field, val)

    run_dir = create_run_dir(config, args.config, group=args.group, exists=args.exists)

    setup_logging(config.experiment_name, log_file=run_dir / "train.log")

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Config: algorithm={config.algorithm}")
    logger.info(f"Model: {config.model_name} (LoRA rank={config.lora_rank})")
    logger.info(f"Data: train={config.train_file}, val={config.val_file}")
    logger.info(
        f"Hyperparameters: lr={config.learning_rate}, epochs={config.num_epochs}, "
        f"batch_size={config.batch_size}, "
        f"rollouts={config.num_rollouts_per_prompt}, "
        f"max_completion={config.max_completion_length}"
    )

    run_training(config)


if __name__ == "__main__":
    main()
