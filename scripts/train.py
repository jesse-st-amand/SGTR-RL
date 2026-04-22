"""Run SGTR-RL training.

Usage:
    python -m scripts.train \
        --config experiments/01_sft_pw_vs_qwen/config.yaml \
        [--runtime runtimes/local_gpu.yaml] \
        [--group sweep_lr] \
        [--exists error|skip|overwrite]
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from scripts.plotting_utils import generate_summary_plot
from sgtr_rl.config import load_training_config
from sgtr_rl.logging_setup import setup_logging
from sgtr_rl.pipeline import run_training
from sgtr_rl.runs import create_run_dir
from sgtr_rl.runtime_config import load_runtime_config

logger = logging.getLogger(__name__)


def _format_data_sources(paths: list[str], fallback: str) -> str:
    if paths:
        if len(paths) == 1:
            return paths[0]
        return "[" + ", ".join(paths) + "]"
    return fallback


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="SGTR-RL training")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--runtime",
        default=None,
        help="Optional runtime config YAML (backend, local GPU, artifact paths, RunPod settings)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["tinker", "local"],
        help="Override runtime backend without editing the runtime YAML",
    )
    parser.add_argument("--group", default=None, help="Group subdirectory for run (e.g. sweep_lr)")
    parser.add_argument(
        "--wandb-project",
        default=None,
        help=(
            "Override wandb project name. Pass an empty string to disable wandb logging for "
            "this run."
        ),
    )
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
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=None)
    parser.add_argument("--max_completion_length", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_train_ids", type=int, default=None)
    parser.add_argument("--subset_seed", type=int, default=None)
    parser.add_argument("--randomize_train_labels", action="store_true")
    parser.add_argument("--randomize_train_labels_seed", type=int, default=None)
    parser.add_argument("--eval_diagnostic_num_examples", type=int, default=None)
    parser.add_argument("--eval_diagnostic_example_ids", nargs="*", default=None)
    parser.add_argument("--train_diagnostic_num_examples", type=int, default=None)
    parser.add_argument("--train_diagnostic_example_ids", nargs="*", default=None)
    parser.add_argument(
        "--resume-manifest",
        default=None,
        help="Path to a resume_manifest.json from a prior Tinker run",
    )
    parser.add_argument(
        "--resume-state-path",
        default=None,
        help="Explicit Tinker state path to resume from with optimizer state",
    )
    parser.add_argument(
        "--resume-completed-epochs",
        type=int,
        default=None,
        help="How many full epochs were already completed in the source run",
    )
    parser.add_argument(
        "--resume-global-step",
        type=int,
        default=None,
        help="Explicit global step to resume metric logging from",
    )

    args = parser.parse_args()

    config = load_training_config(args.config)
    runtime = load_runtime_config(args.runtime)
    if args.backend is not None:
        runtime.backend = args.backend
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project or None

    if args.resume_manifest:
        resume_manifest_path = Path(args.resume_manifest)
        with open(resume_manifest_path, "r") as handle:
            resume_payload = json.load(handle)
        config.resume_state_path = resume_payload.get("state_path")
        config.resume_completed_epochs = resume_payload.get("completed_epochs", 0)
        config.resume_global_step = resume_payload.get("global_step")

    # Apply CLI hyperparameter overrides
    for field in [
        "learning_rate",
        "num_epochs",
        "batch_size",
        "max_steps",
        "num_rollouts_per_prompt",
        "max_completion_length",
        "lora_rank",
        "seed",
        "max_train_ids",
        "subset_seed",
        "randomize_train_labels_seed",
        "eval_diagnostic_num_examples",
        "train_diagnostic_num_examples",
    ]:
        val = getattr(args, field, None)
        if val is not None:
            setattr(config, field, val)
    if args.randomize_train_labels:
        config.randomize_train_labels = True
    if args.eval_diagnostic_example_ids is not None:
        config.eval_diagnostic_example_ids = args.eval_diagnostic_example_ids
    if args.train_diagnostic_example_ids is not None:
        config.train_diagnostic_example_ids = args.train_diagnostic_example_ids
    if args.resume_state_path is not None:
        config.resume_state_path = args.resume_state_path
    if args.resume_completed_epochs is not None:
        config.resume_completed_epochs = args.resume_completed_epochs
    if args.resume_global_step is not None:
        config.resume_global_step = args.resume_global_step

    run_dir = create_run_dir(
        config,
        args.config,
        group=args.group,
        exists=args.exists,
        base_dir=Path(runtime.artifacts.root_dir),
    )
    if config.skipped_existing_run:
        print(f"Skipping existing run: {run_dir}")
        return

    setup_logging(config.experiment_name, log_file=run_dir / "train.log")

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Experiment: {config.experiment_name}")
    logger.info(f"Runtime backend: {runtime.backend}")
    logger.info(f"Config: algorithm={config.algorithm}")
    logger.info(f"Model: {config.model_name} (LoRA rank={config.lora_rank})")
    logger.info(
        "Data: train=%s, val=%s",
        _format_data_sources(config.train_files, config.train_file),
        _format_data_sources(config.val_files, config.val_file),
    )
    if len(config.train_files) > 1:
        logger.info("Train mixing strategy: %s", config.train_mix_strategy)
    if config.max_train_ids is not None:
        subset_seed = config.subset_seed if config.subset_seed is not None else config.seed
        logger.info("Train subset: max_train_ids=%s (seed=%s)", config.max_train_ids, subset_seed)
    if config.randomize_train_labels:
        label_seed = (
            config.randomize_train_labels_seed
            if config.randomize_train_labels_seed is not None
            else config.seed
        )
        logger.info("Train labels randomized (seed=%s)", label_seed)
    logger.info(
        f"Hyperparameters: lr={config.learning_rate}, epochs={config.num_epochs}, "
        f"batch_size={config.batch_size}, "
        f"max_steps={config.max_steps}, "
        f"rollouts={config.num_rollouts_per_prompt}, "
        f"max_completion={config.max_completion_length}"
    )
    logger.info(
        "Evaluation: trigger=%s, frequency=%s",
        config.eval_trigger,
        config.eval_frequency,
    )
    if config.eval_diagnostic_num_examples > 0:
        logger.info(
            "Eval diagnostics: %s examples%s",
            config.eval_diagnostic_num_examples,
            (
                f", ids={config.eval_diagnostic_example_ids}"
                if config.eval_diagnostic_example_ids
                else ""
            ),
        )
    if config.train_diagnostic_num_examples > 0:
        logger.info(
            "Train diagnostics: %s examples%s",
            config.train_diagnostic_num_examples,
            (
                f", ids={config.train_diagnostic_example_ids}"
                if config.train_diagnostic_example_ids
                else ""
            ),
        )
    if config.resume_state_path:
        logger.info(
            "Resume: state=%s completed_epochs=%s global_step=%s",
            config.resume_state_path,
            config.resume_completed_epochs,
            config.resume_global_step,
        )

    run_training(config, runtime)

    try:
        generate_summary_plot(config.run_dir)
    except Exception:
        logger.warning("Failed to generate summary plot", exc_info=True)


if __name__ == "__main__":
    main()
