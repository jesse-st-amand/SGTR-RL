"""Training pipeline orchestration for SGTR-RL."""

import logging
from pathlib import Path

from sgtr_rl.benchmarks import run_benchmark_evals
from sgtr_rl.config import TrainingConfig
from sgtr_rl.data import load_jsonl, validate_training_data
from sgtr_rl.eval import run_val_eval
from sgtr_rl.grpo import train_grpo
from sgtr_rl.plotting import generate_summary_plot
from sgtr_rl.sft import train_sft
from sgtr_rl.tinker import save_checkpoint, setup_tinker

logger = logging.getLogger(__name__)


def _load_prompts(config: TrainingConfig) -> list[dict]:
    """Load prompt dataset from JSONL."""
    prompts = load_jsonl(config.train_file)
    logger.info(f"Loaded {len(prompts)} training prompts")

    # Log example
    example = prompts[0]
    prompt = example["prompt"]
    if isinstance(prompt, list):
        display = f"({len(prompt)} messages, multi-turn)"
    elif len(prompt) > 1000:
        display = prompt[:500] + "\n  [...truncated...]\n" + prompt[-200:]
    else:
        display = prompt
    logger.info(
        f"Example training prompt (target={example['target']}):\n"
        f"  ---\n  {display}\n  ---"
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
    """Full training pipeline: setup -> validate -> baseline -> train -> checkpoint -> plot."""
    prompts = _load_prompts(config)
    val_prompts = _load_val_prompts(config)

    if config.val_file and Path(config.val_file).exists():
        summary = validate_training_data(
            config.train_file, config.val_file, id_field=config.id_field
        )
        logger.info(
            f"Data validation passed: {summary['train_records']} train, "
            f"{summary['val_records']} val, {summary['train_ids']} train IDs, "
            f"{summary['val_ids']} val IDs, format={summary['format']}"
        )

    ctx = setup_tinker(config)

    # Epoch 0 baseline: evaluate untrained model
    logger.info("Running epoch 0 baseline evaluation (untrained model)...")
    run_val_eval(
        val_prompts, ctx.training_client, ctx.renderer, ctx.eval_params,
        ctx.ml_logger, step=0, epoch=0, run_dir=config.run_dir,
        use_system_prompt=config.use_system_prompt,
    )
    run_benchmark_evals(
        config.benchmark_evals, ctx.training_client, ctx.renderer, ctx.eval_params,
        ctx.ml_logger, step=0, epoch=0, total_epochs=config.num_epochs,
        run_dir=config.run_dir, use_system_prompt=config.use_system_prompt,
    )

    train_fns = {"sft": train_sft, "grpo": train_grpo}
    train_fn = train_fns[config.algorithm]
    global_step = train_fn(config, ctx, prompts, val_prompts)

    save_checkpoint(ctx, config, global_step)

    try:
        generate_summary_plot(config.run_dir)
    except Exception:
        logger.warning("Failed to generate summary plot", exc_info=True)

    ctx.ml_logger.close()
    logger.info(f"Training complete. {global_step} steps.")
