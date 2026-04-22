"""Training pipeline orchestration for SGTR-RL."""

import logging
from pathlib import Path

from sgtr_rl.artifacts import update_run_status
from sgtr_rl.config import TrainingConfig
from sgtr_rl.data import (
    load_jsonl_many,
    randomize_binary_targets,
    subset_records_by_id,
    validate_training_data,
)
from sgtr_rl.grpo import train_grpo
from sgtr_rl.local_sft import train_local_sft
from sgtr_rl.runtime_config import RuntimeConfig
from sgtr_rl.sft import train_sft
from sgtr_rl.tinker import save_checkpoint, setup_tinker, write_resume_manifest
from sgtr_rl.tinker_eval import run_benchmark_evals, run_train_panel_eval, run_val_eval

logger = logging.getLogger(__name__)


def _load_prompts(config: TrainingConfig) -> list[dict]:
    """Load prompt dataset from JSONL."""
    prompts = load_jsonl_many(
        config.train_files,
        strategy=config.train_mix_strategy,
        seed=config.seed,
    )
    original_records = len(prompts)

    if config.max_train_ids is not None:
        subset_seed = config.subset_seed if config.subset_seed is not None else config.seed
        prompts = subset_records_by_id(
            prompts,
            config.max_train_ids,
            seed=subset_seed,
        )
        logger.info(
            "Subset train data to %s unique ids using seed=%s (%s records -> %s)",
            config.max_train_ids,
            subset_seed,
            original_records,
            len(prompts),
        )

    if config.randomize_train_labels:
        label_seed = (
            config.randomize_train_labels_seed
            if config.randomize_train_labels_seed is not None
            else config.seed
        )
        prompts = randomize_binary_targets(prompts, seed=label_seed)
        logger.info("Randomized train labels using seed=%s", label_seed)

    if not prompts:
        raise ValueError("No training prompts loaded after train-data transforms")

    if len(config.train_files) == 1:
        logger.info(f"Loaded {len(prompts)} training prompts")
    else:
        logger.info(
            "Loaded %s training prompts from %s files (strategy=%s)",
            len(prompts),
            len(config.train_files),
            config.train_mix_strategy,
        )
        for path in config.train_files:
            logger.info("  train source: %s", path)

    example = prompts[0]
    prompt = example["prompt"]
    if isinstance(prompt, list):
        display = f"({len(prompt)} messages, multi-turn)"
    elif len(prompt) > 1000:
        display = prompt[:500] + "\n  [...truncated...]\n" + prompt[-200:]
    else:
        display = prompt
    logger.info(f"Example training prompt (target={example['target']}):\n  {display}")
    return prompts


def _load_val_prompts(config: TrainingConfig) -> list[dict]:
    """Load validation dataset from JSONL, if configured."""
    if not config.val_files:
        return []
    missing = [path for path in config.val_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Validation files not found: {missing}")
    prompts = load_jsonl_many(config.val_files)
    if len(config.val_files) == 1:
        logger.info(f"Loaded {len(prompts)} validation prompts")
    else:
        logger.info(
            "Loaded %s validation prompts from %s files",
            len(prompts),
            len(config.val_files),
        )
        for path in config.val_files:
            logger.info("  val source: %s", path)
    return prompts


def _run_tinker_training(
    config: TrainingConfig,
    prompts: list[dict],
    val_prompts: list[dict],
) -> int:
    """Original Tinker-backed training flow."""
    if config.resume_state_path and config.algorithm != "sft":
        raise NotImplementedError("Exact Tinker resume is currently supported for SFT only")

    ctx = setup_tinker(config)

    is_resume = bool(config.resume_state_path) or config.resume_completed_epochs > 0
    if is_resume:
        logger.info(
            "Skipping epoch 0 baseline evaluation for resumed run "
            "(completed_epochs=%s, global_step=%s)",
            config.resume_completed_epochs,
            config.resume_global_step,
        )
    else:
        logger.info("Running epoch 0 baseline evaluation (untrained model)...")
        run_val_eval(
            val_prompts,
            ctx,
            step=0,
            epoch=0,
            run_dir=config.run_dir,
            use_system_prompt=config.use_system_prompt,
            eval_trigger=config.eval_trigger,
            diagnostic_num_examples=config.eval_diagnostic_num_examples,
            diagnostic_example_ids=config.eval_diagnostic_example_ids,
        )
        run_train_panel_eval(
            prompts,
            ctx,
            step=0,
            epoch=0,
            run_dir=config.run_dir,
            use_system_prompt=config.use_system_prompt,
            eval_trigger=config.eval_trigger,
            diagnostic_num_examples=config.train_diagnostic_num_examples,
            diagnostic_example_ids=config.train_diagnostic_example_ids,
        )
        run_benchmark_evals(
            config.benchmark_evals,
            ctx,
            step=0,
            epoch=0,
            total_epochs=config.num_epochs,
            run_dir=config.run_dir,
            use_system_prompt=config.use_system_prompt,
            eval_trigger=config.eval_trigger,
        )

    train_fns = {"sft": train_sft, "grpo": train_grpo}
    train_fn = train_fns[config.algorithm]
    global_step = train_fn(config, ctx, prompts, val_prompts)

    checkpoint_paths = save_checkpoint(
        ctx,
        config,
        global_step,
        epoch=config.completed_epochs or config.num_epochs,
    )
    write_resume_manifest(
        config,
        checkpoint_paths,
        completed_epochs=config.completed_epochs or config.num_epochs,
        global_step=global_step,
    )
    ctx.ml_logger.close()
    return global_step


def run_training(config: TrainingConfig, runtime: RuntimeConfig | None = None) -> None:
    """Full training pipeline for the selected runtime/backend."""
    runtime = runtime or RuntimeConfig()
    prompts = _load_prompts(config)
    val_prompts = _load_val_prompts(config)
    summary = validate_training_data(prompts, val_prompts)
    logger.info(
        f"Data validation passed: {summary['train_records']} train, "
        f"{summary['val_records']} val, {summary['train_ids']} train IDs, "
        f"{summary['val_ids']} val IDs, format={summary['format']}"
    )

    update_run_status(
        config.run_dir,
        "starting",
        backend=runtime.backend,
        algorithm=config.algorithm,
        extra={
            "experiment_name": config.experiment_name,
            "resume_state_path": config.resume_state_path,
            "resume_completed_epochs": config.resume_completed_epochs,
            "resume_global_step": config.resume_global_step,
        },
    )

    try:
        if runtime.backend == "tinker":
            global_step = _run_tinker_training(config, prompts, val_prompts)
        elif runtime.backend == "local":
            if config.algorithm != "sft":
                raise NotImplementedError("Local backend currently supports SFT only")
            global_step = train_local_sft(config, runtime, prompts, val_prompts)
        else:
            raise ValueError(f"Unsupported backend: {runtime.backend}")
    except Exception as exc:
        update_run_status(
            config.run_dir,
            "failed",
            backend=runtime.backend,
            algorithm=config.algorithm,
            error=str(exc),
        )
        raise

    update_run_status(
        config.run_dir,
        "completed",
        backend=runtime.backend,
        algorithm=config.algorithm,
        step=global_step,
        epoch=config.completed_epochs or config.num_epochs,
    )
    logger.info(f"Training complete. {global_step} steps.")
