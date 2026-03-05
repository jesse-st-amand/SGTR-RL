"""SFT training function for SGTR-RL using Tinker managed GPU platform."""

import logging
import math
import random
import time

from sgtr_rl.benchmarks import run_benchmark_evals
from sgtr_rl.config import TrainingConfig
from sgtr_rl.data import build_conversation
from sgtr_rl.eval import run_val_eval
from sgtr_rl.tinker import TinkerContext

logger = logging.getLogger(__name__)

# For binary 1/2 task: model prefers correct answer when P(target) > 0.5
_LOG_HALF = math.log(0.5)


def train_sft(
    config: TrainingConfig, ctx: TinkerContext,
    prompts: list[dict], val_prompts: list[dict],
) -> int:
    """SFT training loop. Returns final global_step.

    For each batch of samples:
    1. Build conversation datums with cross-entropy weights on target tokens only
    2. Call forward_backward with cross_entropy loss
    3. Call optim_step
    """
    import tinker
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.supervised.data import conversation_to_datum

    cfg = config
    batch_size = cfg.per_device_train_batch_size
    n_batches = len(prompts) // batch_size
    n_epochs = cfg.num_epochs

    random.seed(cfg.seed)

    logger.info(
        f"Training: {n_epochs} epochs, {n_batches} batches/epoch, "
        f"batch_size={batch_size}, total_steps={n_batches * n_epochs}"
    )

    global_step = 0

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
                convo = build_conversation(item, cfg.use_system_prompt)
                convo.append({"role": "assistant", "content": item["target"]})
                datum = conversation_to_datum(
                    convo, ctx.renderer, None, TrainOnWhat.LAST_ASSISTANT_MESSAGE
                )
                datums.append(datum)

            # Forward-backward + optimizer step
            fwd_bwd_future = ctx.training_client.forward_backward(
                datums, loss_fn="cross_entropy"
            )
            optim_future = ctx.training_client.optim_step(ctx.adam_params)
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

            ctx.ml_logger.log_metrics({
                "train/nll": train_nll,
                "train/accuracy": train_acc,
                "train/batch_time_s": elapsed,
            }, step=global_step)

        logger.info(f"Epoch {epoch+1} complete")

        # Validation evaluation at each epoch boundary
        run_val_eval(
            val_prompts, ctx.training_client, ctx.renderer, ctx.eval_params,
            ctx.ml_logger, step=global_step, epoch=epoch + 1, run_dir=cfg.run_dir,
            use_system_prompt=cfg.use_system_prompt,
        )
        run_benchmark_evals(
            cfg.benchmark_evals, ctx.training_client, ctx.renderer, ctx.eval_params,
            ctx.ml_logger, step=global_step, epoch=epoch + 1,
            total_epochs=n_epochs, run_dir=cfg.run_dir,
            use_system_prompt=cfg.use_system_prompt,
        )

    return global_step
