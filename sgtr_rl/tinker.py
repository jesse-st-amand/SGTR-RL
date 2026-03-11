"""Tinker infrastructure: shared context for training functions."""

import logging
from dataclasses import dataclass
from typing import Any

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

    from pathlib import Path

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


def save_checkpoint(ctx: TinkerContext, config: TrainingConfig, global_step: int) -> None:
    """Save final checkpoint via tinker-cookbook."""
    from pathlib import Path

    from tinker_cookbook import checkpoint_utils

    checkpoint_dir = str(Path(config.run_dir) / "checkpoints") if config.run_dir else (
        f"data/checkpoints/{config.experiment_name}"
    )
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_utils.save_checkpoint(
        training_client=ctx.training_client,
        name="final",
        log_path=checkpoint_dir,
        kind="both",
        loop_state={"batch": global_step},
    )
    logger.info(f"Checkpoint saved to {checkpoint_dir}")
