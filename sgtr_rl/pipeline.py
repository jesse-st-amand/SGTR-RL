"""Training pipeline orchestration for SGTR-RL."""

import logging
from pathlib import Path

from sgtr_rl.config import TrainingConfig
from sgtr_rl.data import load_jsonl, validate_training_data
from sgtr_rl.grpo import train_grpo
from sgtr_rl.sft import train_sft
from sgtr_rl.tinker import save_checkpoint, setup_tinker
from sgtr_rl.tinker_eval import run_benchmark_evals, run_val_eval

logger = logging.getLogger(__name__)


def _load_prompts(config: TrainingConfig) -> list[dict]:
    """Load prompt dataset from JSONL."""
    prompts = load_jsonl(config.train_file)
    logger.info(f"Loaded {len(prompts)} training prompts")

    example = prompts[0]
    prompt = example["prompt"]
    if isinstance(prompt, list):
        display = f"({len(prompt)} messages, multi-turn)"
    elif len(prompt) > 1000:
        display = prompt[:500] + "\n  [...truncated...]\n" + prompt[-200:]
    else:
        display = prompt
    logger.info(
        f"Example training prompt (target={example['target']}):\n  {display}"
    )
    return prompts


def _load_val_prompts(config: TrainingConfig) -> list[dict]:
    """Load validation dataset from JSONL, if configured."""
    if not config.val_file or not Path(config.val_file).exists():
        return []
    prompts = load_jsonl(config.val_file)
    logger.info(f"Loaded {len(prompts)} validation prompts")
    return prompts


def run_training(config: TrainingConfig) -> None:
    """Full training pipeline: setup -> validate -> baseline -> train -> checkpoint."""
    prompts = _load_prompts(config)
    val_prompts = _load_val_prompts(config)
    summary = validate_training_data(prompts, val_prompts)
    logger.info(
        f"Data validation passed: {summary['train_records']} train, "
        f"{summary['val_records']} val, {summary['train_ids']} train IDs, "
        f"{summary['val_ids']} val IDs, format={summary['format']}"
    )

    ctx = setup_tinker(config)

    logger.info("Running epoch 0 baseline evaluation (untrained model)...")
    run_val_eval(
        val_prompts, ctx, step=0, epoch=0,
        run_dir=config.run_dir, use_system_prompt=config.use_system_prompt,
    )
    run_benchmark_evals(
        config.benchmark_evals, ctx, step=0, epoch=0,
        total_epochs=config.num_epochs,
        run_dir=config.run_dir, use_system_prompt=config.use_system_prompt,
    )

    train_fns = {"sft": train_sft, "grpo": train_grpo}
    train_fn = train_fns[config.algorithm]
    global_step = train_fn(config, ctx, prompts, val_prompts)

    save_checkpoint(ctx, config, global_step)

    ctx.ml_logger.close()
    logger.info(f"Training complete. {global_step} steps.")
