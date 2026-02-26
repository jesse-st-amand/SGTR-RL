"""SFT trainer for SGTR-RL using Tinker managed GPU platform."""

import json
import logging
import math
import random
import time
from pathlib import Path

from sgtr_rl.training.train_config import TrainingConfig
from sgtr_rl.training.eval import run_val_eval
from sgtr_rl.training.benchmark_eval import run_benchmark_evals
from sgtr_rl.training.plot_summary import generate_summary_plot
from sgtr_rl.data_processing.validate_data import validate_training_data

logger = logging.getLogger(__name__)

# For binary 1/2 task: model prefers correct answer when P(target) > 0.5
_LOG_HALF = math.log(0.5)


class TinkerSFTTrainer:
    """Supervised fine-tuning trainer using Tinker's cross-entropy loss.

    Trains the model on labeled (prompt, target) pairs using standard
    cross-entropy with loss weights only on the assistant's response tokens.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def _load_prompts(self) -> list[dict]:
        """Load prompt dataset from JSONL."""
        prompts = []
        with open(self.config.train_file, "r") as f:
            for line in f:
                prompts.append(json.loads(line))
        logger.info(f"Loaded {len(prompts)} training prompts")
        return prompts

    def _load_val_prompts(self) -> list[dict]:
        """Load validation dataset from JSONL, if configured."""
        if not self.config.val_file or not Path(self.config.val_file).exists():
            return []
        prompts = []
        with open(self.config.val_file, "r") as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line))
        logger.info(f"Loaded {len(prompts)} validation prompts")
        return prompts

    def train(self, resume_from_checkpoint: str | None = None):
        """Run SFT training via Tinker API.

        For each batch of samples:
        1. Build conversation datums with cross-entropy weights on target tokens only
        2. Call forward_backward with cross_entropy loss
        3. Call optim_step
        """
        import tinker
        from tinker import types
        from tinker_cookbook import checkpoint_utils, model_info, renderers
        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised.common import compute_mean_nll
        from tinker_cookbook.supervised.data import conversation_to_datum
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        from tinker_cookbook.utils import ml_log

        cfg = self.config
        prompts = self._load_prompts()
        val_prompts = self._load_val_prompts()

        # Validate data integrity
        if cfg.val_file and Path(cfg.val_file).exists():
            summary = validate_training_data(cfg.train_file, cfg.val_file)
            logger.info(
                f"Data validation passed: {summary['train_records']} train, "
                f"{summary['val_records']} val, {summary['train_uuids']} train UUIDs, "
                f"{summary['val_uuids']} val UUIDs, format={summary['format']}"
            )

        # Log example
        example = prompts[0]
        prompt_text = example["prompt"]
        if len(prompt_text) > 1000:
            display = prompt_text[:500] + "\n  [...truncated...]\n" + prompt_text[-200:]
        else:
            display = prompt_text
        logger.info(
            f"Example training prompt (target={example['target']}):\n"
            f"  ---\n  {display}\n  ---"
        )

        # Tinker setup
        logger.info(f"Connecting to Tinker (model={cfg.model_name}, lora_rank={cfg.lora_rank})...")
        service_client = tinker.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=cfg.model_name, rank=cfg.lora_rank
        )
        logger.info("Tinker training client created")

        tokenizer = get_tokenizer(cfg.model_name)
        renderer_name = model_info.get_recommended_renderer_name(cfg.model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        logger.info(f"Using renderer: {renderer_name}")

        adam_params = types.AdamParams(
            learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
        )

        eval_params = types.SamplingParams(
            max_tokens=cfg.max_completion_length,
            stop=renderer.get_stop_sequences(),
            temperature=0.0,
        )

        batch_size = cfg.per_device_train_batch_size
        n_batches = len(prompts) // batch_size
        n_epochs = cfg.num_epochs

        checkpoint_dir = str(Path(cfg.run_dir) / "checkpoints") if cfg.run_dir else (
            cfg.output_dir or f"data/checkpoints/{cfg.experiment_name}"
        )
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Metrics logging (wandb + JSON, replaces tensorboard)
        log_dir = str(Path(cfg.run_dir) / "metrics") if cfg.run_dir else None
        ml_logger = ml_log.setup_logging(
            log_dir=log_dir or ".",
            wandb_project=cfg.wandb_project,
            wandb_name=cfg.experiment_name,
            config=cfg,
            do_configure_logging_module=False,
        )

        logger.info(
            f"Training: {n_epochs} epochs, {n_batches} batches/epoch, "
            f"batch_size={batch_size}, total_steps={n_batches * n_epochs}"
        )

        global_step = 0

        # Epoch 0 baseline: evaluate untrained model
        logger.info("Running epoch 0 baseline evaluation (untrained model)...")
        run_val_eval(
            val_prompts, training_client, renderer, eval_params,
            ml_logger, step=0, epoch=0, run_dir=cfg.run_dir,
        )
        run_benchmark_evals(
            cfg.benchmark_evals, training_client, renderer, eval_params,
            ml_logger, step=0, epoch=0, total_epochs=n_epochs,
            run_dir=cfg.run_dir,
        )

        for epoch in range(n_epochs):
            # Shuffle prompts each epoch
            random.shuffle(prompts)

            for batch_idx in range(n_batches):
                t_start = time.time()
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(prompts))
                batch = prompts[batch_start:batch_end]

                # Build datums with cross-entropy weights on assistant tokens only
                datums: list[tinker.Datum] = []

                for item in batch:
                    convo = [
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["target"]},
                    ]
                    datum = conversation_to_datum(
                        convo, renderer, None, TrainOnWhat.LAST_ASSISTANT_MESSAGE
                    )
                    datums.append(datum)

                # Forward-backward + optimizer step
                fwd_bwd_future = training_client.forward_backward(
                    datums, loss_fn="cross_entropy"
                )
                optim_future = training_client.optim_step(adam_params)
                fwd_bwd_result = fwd_bwd_future.result()
                optim_future.result()

                # Compute mean NLL and accuracy from forward pass
                logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
                weights = [d.loss_fn_inputs["weights"] for d in datums]
                train_nll = compute_mean_nll(logprobs, weights)

                # Train accuracy: for binary 1/2 task, the model is correct
                # if P(target_token) > 0.5, i.e. logprob > log(0.5)
                batch_correct = 0
                for lp, w in zip(logprobs, weights):
                    lp_t = lp.to_torch()
                    w_t = w.to_torch()
                    # Answer token is where weight > 0 (single token for "1"/"2")
                    mask = w_t > 0
                    if mask.any():
                        answer_logprob = float(lp_t[mask][0])
                        if answer_logprob > _LOG_HALF:
                            batch_correct += 1
                train_acc = batch_correct / len(datums)

                elapsed = time.time() - t_start

                logger.info(
                    f"[epoch {epoch+1}/{n_epochs}] batch {batch_idx+1}/{n_batches} | "
                    f"nll={train_nll:.4f} | acc={train_acc:.1%} | {elapsed:.1f}s"
                )
                global_step += 1

                ml_logger.log_metrics({
                    "train/nll": train_nll,
                    "train/accuracy": train_acc,
                    "train/batch_time_s": elapsed,
                }, step=global_step)

            logger.info(f"Epoch {epoch+1} complete")

            # Validation evaluation at each epoch boundary
            run_val_eval(
                val_prompts, training_client, renderer, eval_params,
                ml_logger, step=global_step, epoch=epoch + 1, run_dir=cfg.run_dir,
            )
            run_benchmark_evals(
                cfg.benchmark_evals, training_client, renderer, eval_params,
                ml_logger, step=global_step, epoch=epoch + 1,
                total_epochs=n_epochs, run_dir=cfg.run_dir,
            )

        ml_logger.close()

        # Generate summary plot
        try:
            generate_summary_plot(cfg.run_dir)
        except Exception:
            logger.warning("Failed to generate summary plot", exc_info=True)

        # Save final checkpoint
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=checkpoint_dir,
            kind="both",
            loop_state={"batch": global_step},
        )
        logger.info(f"Training complete. {global_step} steps. Checkpoint saved to {checkpoint_dir}")
