"""SFT training function for SGTR-RL using Tinker managed GPU platform."""

import logging
import math
import random
import time

from sgtr_rl.benchmarks import should_run_training_eval
from sgtr_rl.config import TrainingConfig
from sgtr_rl.data import build_conversation
from sgtr_rl.tinker import TinkerContext
from sgtr_rl.tinker_eval import run_benchmark_evals, run_train_panel_eval, run_val_eval

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

    batch_size = config.batch_size
    n_batches = len(prompts) // batch_size
    n_epochs = config.num_epochs
    max_steps = config.max_steps

    if n_batches == 0:
        raise ValueError(
            f"Not enough training records ({len(prompts)}) for batch_size={batch_size}. "
            "Reduce batch_size or increase train data."
        )

    if config.resume_completed_epochs > n_epochs:
        raise ValueError(
            "resume_completed_epochs cannot exceed num_epochs: "
            f"{config.resume_completed_epochs} > {n_epochs}"
        )

    rng = random.Random(config.seed)
    for _ in range(config.resume_completed_epochs):
        rng.shuffle(prompts)

    total_steps = (
        n_batches * n_epochs
        if max_steps is None
        else min(n_batches * n_epochs, max_steps)
    )

    logger.info(
        "Training: %s total epochs, %s batches/epoch, batch_size=%s, total_steps=%s, "
        "resume_completed_epochs=%s",
        n_epochs,
        n_batches,
        batch_size,
        total_steps,
        config.resume_completed_epochs,
    )

    global_step = config.resume_global_step
    if global_step is None:
        global_step = config.resume_completed_epochs * n_batches

    if config.resume_completed_epochs:
        logger.info(
            "Resuming SFT from epoch %s with starting global_step=%s",
            config.resume_completed_epochs,
            global_step,
        )

    completed_epochs = config.resume_completed_epochs
    stopped_early = False
    train_eval_prompts = list(prompts)

    if max_steps is not None and global_step >= max_steps:
        logger.info("Resume point already satisfies max_steps=%s; skipping training", max_steps)
        config.completed_epochs = completed_epochs
        return global_step

    for epoch in range(config.resume_completed_epochs, n_epochs):
        current_epoch = epoch + 1
        # Shuffle prompts each epoch
        rng.shuffle(prompts)

        for batch_idx in range(n_batches):
            t_start = time.time()
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(prompts))
            batch = prompts[batch_start:batch_end]

            # Build datums with cross-entropy weights on assistant tokens only
            datums: list[tinker.Datum] = []

            for item in batch:
                convo = build_conversation(item, config.use_system_prompt)
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
                f"[epoch {current_epoch}/{n_epochs}] batch {batch_idx+1}/{n_batches} | "
                f"nll={train_nll:.4f} | acc={train_acc:.1%} | {elapsed:.1f}s"
            )
            global_step += 1

            ctx.ml_logger.log_metrics({
                "train/nll": train_nll,
                "train/accuracy": train_acc,
                "train/batch_time_s": elapsed,
            }, step=global_step)

            if config.eval_trigger == "step" and should_run_training_eval(
                trigger=config.eval_trigger,
                frequency=config.eval_frequency,
                step=global_step,
                epoch=current_epoch,
                total_steps=total_steps,
                total_epochs=n_epochs,
            ):
                run_val_eval(
                    val_prompts,
                    ctx,
                    step=global_step,
                    epoch=current_epoch,
                    run_dir=config.run_dir,
                    use_system_prompt=config.use_system_prompt,
                    eval_trigger=config.eval_trigger,
                    diagnostic_num_examples=config.eval_diagnostic_num_examples,
                    diagnostic_example_ids=config.eval_diagnostic_example_ids,
                )
                run_train_panel_eval(
                    train_eval_prompts,
                    ctx,
                    step=global_step,
                    epoch=current_epoch,
                    run_dir=config.run_dir,
                    use_system_prompt=config.use_system_prompt,
                    eval_trigger=config.eval_trigger,
                    diagnostic_num_examples=config.train_diagnostic_num_examples,
                    diagnostic_example_ids=config.train_diagnostic_example_ids,
                )
                run_benchmark_evals(
                    config.benchmark_evals,
                    ctx,
                    step=global_step,
                    epoch=current_epoch,
                    total_epochs=n_epochs,
                    schedule_index=global_step,
                    schedule_total=total_steps,
                    run_dir=config.run_dir,
                    use_system_prompt=config.use_system_prompt,
                    eval_trigger=config.eval_trigger,
                )

            if max_steps is not None and global_step >= max_steps:
                stopped_early = True
                logger.info(
                    "Reached max_steps=%s at epoch %s batch %s/%s",
                    max_steps,
                    current_epoch,
                    batch_idx + 1,
                    n_batches,
                )
                break

        completed_epochs = current_epoch
        if stopped_early and batch_idx + 1 < n_batches:
            logger.info("Stopped during epoch %s after %s batches", current_epoch, batch_idx + 1)
        else:
            logger.info(f"Epoch {current_epoch} complete")

        if config.eval_trigger == "epoch" and should_run_training_eval(
            trigger=config.eval_trigger,
            frequency=config.eval_frequency,
            step=global_step,
            epoch=current_epoch,
            total_steps=total_steps,
            total_epochs=n_epochs,
        ):
            run_val_eval(
                val_prompts,
                ctx,
                step=global_step,
                epoch=current_epoch,
                run_dir=config.run_dir,
                use_system_prompt=config.use_system_prompt,
                eval_trigger=config.eval_trigger,
                diagnostic_num_examples=config.eval_diagnostic_num_examples,
                diagnostic_example_ids=config.eval_diagnostic_example_ids,
            )
            run_train_panel_eval(
                train_eval_prompts,
                ctx,
                step=global_step,
                epoch=current_epoch,
                run_dir=config.run_dir,
                use_system_prompt=config.use_system_prompt,
                eval_trigger=config.eval_trigger,
                diagnostic_num_examples=config.train_diagnostic_num_examples,
                diagnostic_example_ids=config.train_diagnostic_example_ids,
            )
            run_benchmark_evals(
                config.benchmark_evals,
                ctx,
                step=global_step,
                epoch=current_epoch,
                total_epochs=completed_epochs if stopped_early else n_epochs,
                run_dir=config.run_dir,
                use_system_prompt=config.use_system_prompt,
                eval_trigger=config.eval_trigger,
            )

        if stopped_early:
            break

    config.completed_epochs = completed_epochs
    return global_step
