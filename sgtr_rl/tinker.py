"""Tinker infrastructure: shared context for training functions."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sgtr_rl.artifacts import atomic_write_json
from sgtr_rl.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TinkerContext:
    """Shared Tinker state created once and passed to training functions."""

    training_client: Any
    renderer: Any
    tokenizer: Any
    eval_params: Any
    adam_params: Any
    ml_logger: Any


def setup_tinker(config: TrainingConfig) -> TinkerContext:
    """Create ServiceClient, training client, tokenizer, renderer, params.

    Consolidates all Tinker setup code previously duplicated in both trainers.
    """
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_cookbook.utils import ml_log

    logger.info(
        f"Connecting to Tinker (model={config.model_name}, lora_rank={config.lora_rank})..."
    )
    service_client = tinker.ServiceClient()
    if config.resume_state_path:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            config.resume_state_path
        )
        logger.info("Resumed Tinker training client from state: %s", config.resume_state_path)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        logger.info("Tinker training client created")

    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    eval_params = types.SamplingParams(
        max_tokens=config.max_completion_length,
        stop=renderer.get_stop_sequences(),
        temperature=0.0,
    )
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    log_dir = str(Path(config.run_dir) / "metrics") if config.run_dir else None
    ml_logger = ml_log.setup_logging(
        log_dir=log_dir or ".",
        wandb_project=config.wandb_project,
        wandb_name=config.experiment_name,
        config=config,
        do_configure_logging_module=False,
    )

    return TinkerContext(
        training_client=training_client,
        renderer=renderer,
        tokenizer=tokenizer,
        eval_params=eval_params,
        adam_params=adam_params,
        ml_logger=ml_logger,
    )


def save_checkpoint(
    ctx: TinkerContext,
    config: TrainingConfig,
    global_step: int,
    *,
    epoch: int | None = None,
    name: str = "final",
) -> dict[str, str]:
    """Save checkpoint via tinker-cookbook and return checkpoint paths."""
    from tinker_cookbook import checkpoint_utils

    checkpoint_dir = str(Path(config.run_dir) / "checkpoints") if config.run_dir else (
        f"data/checkpoints/{config.experiment_name}"
    )
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    loop_state: dict[str, int] = {"batch": global_step}
    if epoch is not None:
        loop_state["epoch"] = epoch

    checkpoint_paths = checkpoint_utils.save_checkpoint(
        training_client=ctx.training_client,
        name=name,
        log_path=checkpoint_dir,
        kind="both",
        loop_state=loop_state,
    )
    logger.info("Checkpoint saved to %s", checkpoint_dir)
    return checkpoint_paths


def write_resume_manifest(
    config: TrainingConfig,
    checkpoint_paths: dict[str, str],
    *,
    completed_epochs: int,
    global_step: int,
) -> Path:
    """Write a small manifest for exact future resume from this run."""
    manifest_path = Path(config.run_dir) / "checkpoints" / "resume_manifest.json"
    payload = {
        "experiment_name": config.experiment_name,
        "backend": "tinker",
        "algorithm": config.algorithm,
        "model_name": config.model_name,
        "train_files": config.train_files or ([config.train_file] if config.train_file else []),
        "val_files": config.val_files or ([config.val_file] if config.val_file else []),
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "state_path": checkpoint_paths.get("state_path"),
        "sampler_path": checkpoint_paths.get("sampler_path"),
    }
    atomic_write_json(manifest_path, payload)
    logger.info("Resume manifest written to %s", manifest_path)
    return manifest_path
