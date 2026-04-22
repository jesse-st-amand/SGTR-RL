"""GRPO training function for SGTR-RL."""

import logging
import time

from sgtr_rl.answer import extract_answer
from sgtr_rl.benchmarks import should_run_training_eval
from sgtr_rl.config import TrainingConfig
from sgtr_rl.data import build_conversation
from sgtr_rl.reward import sgtr_binary_reward
from sgtr_rl.tinker import TinkerContext
from sgtr_rl.tinker_eval import run_benchmark_evals, run_train_panel_eval, run_val_eval

logger = logging.getLogger(__name__)


def _get_reward(completion_text: str, target: str) -> float:
    """Score a single completion against its target."""
    return sgtr_binary_reward([completion_text], [target])[0]


def _log_example_prompt(prompts: list[dict]) -> None:
    """Log an example prompt so the user can verify data looks right."""
    example = prompts[0]
    prompt = example["prompt"]
    if isinstance(prompt, list):
        # Multi-turn: show role/content summary
        parts = [f"  [{m['role']}] {m['content'][:200]}..." if len(m['content']) > 200
                 else f"  [{m['role']}] {m['content']}" for m in prompt]
        display = f"({len(prompt)} messages)\n" + "\n".join(parts)
    else:
        if len(prompt) > 1000:
            display = prompt[:500] + "\n  [...truncated...]\n" + prompt[-200:]
        else:
            display = prompt
    logger.info(
        f"Example training prompt (target={example['target']}):\n"
        f"  ---\n  {display}\n  ---"
    )


def train_grpo(
    config: TrainingConfig, ctx: TinkerContext,
    prompts: list[dict], val_prompts: list[dict],
) -> int:
    """GRPO training loop. Returns final global_step.

    For each batch of prompts:
    1. Sample ``num_rollouts_per_prompt`` completions per prompt
    2. Score each completion with the binary SGTR reward
    3. Compute GRPO advantages (per-group mean centering)
    4. Build Tinker Datum objects and run forward_backward + optim_step
    """
    import torch
    from tinker import types
    from tinker.types.tensor_data import TensorData
    from tinker_cookbook import renderers

    _log_example_prompt(prompts)

    group_size = config.num_rollouts_per_prompt
    batch_size = config.batch_size
    n_batches = len(prompts) // batch_size
    n_epochs = config.num_epochs

    sampling_params = types.SamplingParams(
        max_tokens=config.max_completion_length,
        stop=ctx.renderer.get_stop_sequences(),
        temperature=config.sampling_temperature,
    )

    logger.info(
        f"Training: {n_epochs} epochs, {n_batches} batches/epoch, "
        f"batch_size={batch_size}, group_size={group_size}, "
        f"total_steps={n_batches * n_epochs}, "
        f"temperature={config.sampling_temperature}"
    )

    global_step = 0
    logged_first_output = False
    cumulative_correct = 0
    cumulative_total = 0
    train_eval_prompts = list(prompts)

    for epoch in range(n_epochs):
        epoch_rewards: list[float] = []

        for batch_idx in range(n_batches):
            t_start = time.time()
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(prompts))
            batch = prompts[batch_start:batch_end]

            # Get fresh sampling client from current weights
            sampling_client = ctx.training_client.save_weights_and_get_sampling_client()

            # Build model inputs and fire off sampling requests
            futures = []
            model_inputs = []
            for item in batch:
                convo = build_conversation(item, config.use_system_prompt)
                model_input = ctx.renderer.build_generation_prompt(convo)
                future = sampling_client.sample(
                    prompt=model_input,
                    num_samples=group_size,
                    sampling_params=sampling_params,
                )
                futures.append(future)
                model_inputs.append(model_input)

            # Collect results, compute rewards and advantages, build datums
            datums: list[types.Datum] = []
            batch_rewards: list[float] = []
            batch_correct = 0
            batch_total = 0
            groups_skipped = 0
            batch_advantages: list[float] = []
            batch_answers: dict[str, int] = {"1": 0, "2": 0, "other": 0}

            for group_idx, (future, prompt_input, item) in enumerate(
                zip(futures, model_inputs, batch)
            ):
                sample_result = future.result()
                target = item["target"]

                group_rewards: list[float] = []
                group_tokens: list[list[int]] = []
                group_logprobs: list[list[float]] = []
                group_contents: list[str] = []

                for sequence in sample_result.sequences:
                    group_tokens.append(sequence.tokens)
                    assert sequence.logprobs is not None
                    group_logprobs.append(sequence.logprobs)

                    parsed_msg, _ = ctx.renderer.parse_response(sequence.tokens)
                    content = renderers.get_text_content(parsed_msg)
                    group_contents.append(content)
                    reward = _get_reward(content, target)
                    group_rewards.append(reward)

                    # Track answer distribution
                    answer = extract_answer(content)
                    if answer in ("1", "2"):
                        batch_answers[answer] += 1
                    else:
                        batch_answers["other"] += 1

                    # Log the very first output
                    if not logged_first_output:
                        logger.info(
                            f"Example model output (first sample of training):\n"
                            f"  completion: {content!r}\n"
                            f"  target: {target!r}, reward: {reward}"
                        )
                        logged_first_output = True

                batch_correct += sum(int(r == 1.0) for r in group_rewards)
                batch_total += len(group_rewards)

                # GRPO: center advantages within the group
                mean_reward = sum(group_rewards) / len(group_rewards)
                group_advantages = [r - mean_reward for r in group_rewards]
                batch_rewards.append(mean_reward)

                # Skip groups where all rewards are identical (no signal)
                if all(a == 0.0 for a in group_advantages):
                    groups_skipped += 1
                    continue

                batch_advantages.extend(group_advantages)

                # Log detailed group info for first group with signal each batch
                if len(batch_advantages) == len(group_advantages):
                    n_correct = sum(int(r == 1.0) for r in group_rewards)
                    sample_content = group_contents[0]
                    if len(sample_content) > 120:
                        sample_content = sample_content[:60] + "..." + sample_content[-40:]
                    logger.debug(
                        f"  group {group_idx}: target={target} "
                        f"rewards={[int(r) for r in group_rewards]} "
                        f"({n_correct}/{len(group_rewards)} correct) "
                        f"advantages=[{', '.join(f'{a:+.2f}' for a in group_advantages)}] "
                        f"sample_completion={sample_content!r}"
                    )

                # Build training datums
                ob_len = prompt_input.length - 1
                for tokens, logprobs, advantage in zip(
                    group_tokens, group_logprobs, group_advantages
                ):
                    model_input = prompt_input.append(
                        types.EncodedTextChunk(tokens=tokens[:-1])
                    )
                    target_tokens = [0] * ob_len + tokens
                    padded_logprobs = [0.0] * ob_len + logprobs
                    padded_advantages = (
                        [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                    )
                    assert (
                        model_input.length
                        == len(target_tokens)
                        == len(padded_logprobs)
                        == len(padded_advantages)
                    )
                    datums.append(
                        types.Datum(
                            model_input=model_input,
                            loss_fn_inputs={
                                "target_tokens": TensorData.from_torch(
                                    torch.tensor(target_tokens)
                                ),
                                "logprobs": TensorData.from_torch(
                                    torch.tensor(padded_logprobs)
                                ),
                                "advantages": TensorData.from_torch(
                                    torch.tensor(padded_advantages)
                                ),
                            },
                        )
                    )

            # Training step
            if datums:
                fwd_bwd_future = ctx.training_client.forward_backward(
                    datums, loss_fn="importance_sampling"
                )
                optim_future = ctx.training_client.optim_step(ctx.adam_params)
                fwd_bwd_future.result()
                optim_future.result()

            avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            epoch_rewards.extend(batch_rewards)
            cumulative_correct += batch_correct
            cumulative_total += batch_total
            elapsed = time.time() - t_start
            batch_acc = batch_correct / batch_total if batch_total else 0.0
            running_acc = cumulative_correct / cumulative_total if cumulative_total else 0.0

            adv_abs_mean = (
                sum(abs(a) for a in batch_advantages) / len(batch_advantages)
                if batch_advantages else 0.0
            )

            ans = batch_answers
            logger.info(
                f"[epoch {epoch+1}/{n_epochs}] batch {batch_idx+1}/{n_batches} | "
                f"reward={avg_reward:.3f} | acc={batch_acc:.1%} "
                f"(running={running_acc:.1%}) | "
                f"datums={len(datums)} | "
                f"skipped={groups_skipped}/{len(batch)} groups | "
                f"adv_mag={adv_abs_mean:.3f} | "
                f"answers={{1:{ans['1']},2:{ans['2']},"
                f"?:{ans['other']}}} | {elapsed:.1f}s"
            )
            global_step += 1

            ctx.ml_logger.log_metrics({
                "train/reward": avg_reward,
                "train/accuracy": batch_acc,
                "train/running_accuracy": running_acc,
                "train/datums": len(datums),
                "train/groups_skipped": groups_skipped,
                "train/advantage_magnitude": adv_abs_mean,
                "train/answers_1_pct": batch_answers["1"] / max(batch_total, 1),
                "train/answers_other_pct": batch_answers["other"] / max(batch_total, 1),
                "train/batch_time_s": elapsed,
            }, step=global_step)

            if config.eval_trigger == "step" and should_run_training_eval(
                trigger=config.eval_trigger,
                frequency=config.eval_frequency,
                step=global_step,
                epoch=epoch + 1,
                total_steps=n_batches * n_epochs,
                total_epochs=n_epochs,
            ):
                run_val_eval(
                    val_prompts,
                    ctx,
                    step=global_step,
                    epoch=epoch + 1,
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
                    epoch=epoch + 1,
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
                    epoch=epoch + 1,
                    total_epochs=n_epochs,
                    schedule_index=global_step,
                    schedule_total=n_batches * n_epochs,
                    run_dir=config.run_dir,
                    use_system_prompt=config.use_system_prompt,
                    eval_trigger=config.eval_trigger,
                )

        epoch_avg = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0.0
        logger.info(
            f"Epoch {epoch+1} complete | avg reward={epoch_avg:.3f} | "
            f"running acc={cumulative_correct}/{cumulative_total} "
            f"= {cumulative_correct/cumulative_total:.1%}" if cumulative_total else
            f"Epoch {epoch+1} complete | avg reward={epoch_avg:.3f}"
        )
        ctx.ml_logger.log_metrics({"train/epoch_reward": epoch_avg}, step=global_step)

        if config.eval_trigger == "epoch" and should_run_training_eval(
            trigger=config.eval_trigger,
            frequency=config.eval_frequency,
            step=global_step,
            epoch=epoch + 1,
            total_steps=n_batches * n_epochs,
            total_epochs=n_epochs,
        ):
            run_val_eval(
                val_prompts,
                ctx,
                step=global_step,
                epoch=epoch + 1,
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
                epoch=epoch + 1,
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
                epoch=epoch + 1,
                total_epochs=n_epochs,
                run_dir=config.run_dir,
                use_system_prompt=config.use_system_prompt,
                eval_trigger=config.eval_trigger,
            )

    return global_step
