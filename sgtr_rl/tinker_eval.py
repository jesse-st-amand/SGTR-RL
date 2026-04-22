"""Tinker-based evaluation: validation accuracy/NLL and benchmark evals.

All functions in this module interact with Tinker's sampling/training clients.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable

from sgtr_rl.answer import extract_answer
from sgtr_rl.benchmarks import (
    extract_mmlu_answer,
    format_mmlu_prompt,
    load_benchmark_data,
    should_run_benchmark,
    subsample,
)
from sgtr_rl.data import build_conversation
from sgtr_rl.eval_diagnostics import (
    build_prompt_preview,
    select_binary_diagnostic_items,
    summarize_binary_margin_rows,
)
from sgtr_rl.metrics import (
    log_binary_eval_result,
    log_split_metrics,
    log_val_metrics,
    log_val_result,
    make_eval_artifact_name,
    save_split_diagnostics,
    save_split_predictions,
    save_val_diagnostics,
    save_val_predictions,
)
from sgtr_rl.tinker import TinkerContext

logger = logging.getLogger(__name__)


def _collect_completions(
    items: list[dict],
    sampling_client: Any,
    renderer: Any,
    eval_params: Any,
    build_prompt: Callable[[dict], list[dict]],
) -> list[tuple[str, Any]]:
    """Fire async sampling for all items and return (content, sequence) pairs.

    This is the shared Tinker sampling loop used by all evaluate_* functions.
    """
    from tinker_cookbook import renderers as r

    futures = []
    for item in items:
        convo = build_prompt(item)
        model_input = renderer.build_generation_prompt(convo)
        futures.append(sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=eval_params,
        ))

    completions = []
    for future in futures:
        result = future.result()
        seq = result.sequences[0]
        parsed_msg, _ = renderer.parse_response(seq.tokens)
        content = r.get_text_content(parsed_msg)
        completions.append((content, seq))
    return completions


def _binary_choice_token_ids(tokenizer: Any) -> tuple[int, int]:
    token_1 = tokenizer.encode("1", add_special_tokens=False)
    token_2 = tokenizer.encode("2", add_special_tokens=False)
    if len(token_1) != 1 or len(token_2) != 1:
        raise ValueError("Expected '1' and '2' to encode to single tokens")
    return token_1[0], token_2[0]


def compute_binary_margin_diagnostics(
    items: list[dict],
    *,
    sampling_client: Any,
    renderer: Any,
    tokenizer: Any,
    eval_params: Any,
    use_system_prompt: bool = False,
) -> dict:
    """Measure log p(1) - log p(2) on a fixed panel of examples."""
    import tinker

    token_1, token_2 = _binary_choice_token_ids(tokenizer)
    scoring_params = tinker.types.SamplingParams(
        max_tokens=1,
        stop=eval_params.stop,
        temperature=0.0,
    )

    pending: list[tuple[dict, Any, Any]] = []
    for item in items:
        convo = build_conversation(item, use_system_prompt)
        model_input = renderer.build_generation_prompt(convo)
        prompt_1 = model_input.append(tinker.types.EncodedTextChunk(tokens=[token_1]))
        prompt_2 = model_input.append(tinker.types.EncodedTextChunk(tokens=[token_2]))
        pending.append(
            (
                item,
                sampling_client.sample(
                    prompt=prompt_1,
                    num_samples=1,
                    sampling_params=scoring_params,
                    include_prompt_logprobs=True,
                ),
                sampling_client.sample(
                    prompt=prompt_2,
                    num_samples=1,
                    sampling_params=scoring_params,
                    include_prompt_logprobs=True,
                ),
            )
        )

    rows = []
    for item, future_1, future_2 in pending:
        result_1 = future_1.result()
        result_2 = future_2.result()
        prompt_logprobs_1 = result_1.prompt_logprobs or []
        prompt_logprobs_2 = result_2.prompt_logprobs or []
        logprob_1 = prompt_logprobs_1[-1] if prompt_logprobs_1 else None
        logprob_2 = prompt_logprobs_2[-1] if prompt_logprobs_2 else None
        if logprob_1 is None or logprob_2 is None:
            continue

        margin = float(logprob_1 - logprob_2)
        predicted_by_margin = "1" if margin >= 0 else "2"
        rows.append(
            {
                "id": item.get("id", ""),
                "target": item["target"],
                "predicted_by_margin": predicted_by_margin,
                "correct_by_margin": predicted_by_margin == item["target"],
                "logprob_1": float(logprob_1),
                "logprob_2": float(logprob_2),
                "margin_1_minus_2": margin,
                "prompt_preview": build_prompt_preview(item),
            }
        )

    return {
        "examples": rows,
        "summary": summarize_binary_margin_rows(rows),
    }


def compute_val_nll(
    val_prompts: list[dict], ctx: TinkerContext,
    use_system_prompt: bool = False,
) -> float:
    """Compute mean NLL on validation set via forward pass.

    Builds SFT-style datums and runs a forward-only loss computation.
    """
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.supervised.data import conversation_to_datum

    datums = []
    for item in val_prompts:
        convo = build_conversation(item, use_system_prompt)
        convo.append({"role": "assistant", "content": item["target"]})
        datum = conversation_to_datum(
            convo, ctx.renderer, None, TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        datums.append(datum)

    forward_result = ctx.training_client.forward(
        datums, loss_fn="cross_entropy"
    ).result()

    logprobs = [x["logprobs"] for x in forward_result.loss_fn_outputs]
    weights = [d.loss_fn_inputs["weights"] for d in datums]
    return compute_mean_nll(logprobs, weights)


def evaluate_val(
    val_prompts: list[dict], sampling_client: Any, renderer: Any, eval_params: Any,
    use_system_prompt: bool = False,
) -> dict:
    """Run greedy evaluation on validation set.

    Returns:
        Dict with accuracy, correct, total, answer distribution,
        and per-sample predictions.
    """
    def build_prompt(item):
        return build_conversation(item, use_system_prompt)

    completions = _collect_completions(
        val_prompts, sampling_client, renderer, eval_params, build_prompt,
    )

    correct = 0
    answers = {"1": 0, "2": 0, "other": 0}
    predictions = []

    for (content, seq), item in zip(completions, val_prompts):
        answer = extract_answer(content)
        target = item["target"]
        is_correct = answer == target
        if is_correct:
            correct += 1
        answers[answer if answer in ("1", "2") else "other"] += 1

        predictions.append({
            "id": item.get("id", ""),
            "prediction": answer,
            "target": target,
            "correct": is_correct,
            "logprob": seq.logprobs[0] if seq.logprobs else None,
        })

    total = len(val_prompts)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct, "total": total,
        "answers": answers, "predictions": predictions,
    }


def evaluate_benchmark(
    data: list[dict], sampling_client: Any, renderer: Any, eval_params: Any,
    cot: bool,
) -> dict:
    """Run MMLU benchmark evaluation on loaded data.

    Returns:
        Dict with accuracy, correct, total, per-subject breakdown,
        answer distribution, and per-item predictions.
    """
    def build_prompt(item):
        return [{"role": "user", "content": format_mmlu_prompt(item, cot=cot)}]

    completions = _collect_completions(
        data, sampling_client, renderer, eval_params, build_prompt,
    )

    correct = 0
    answers = {"A": 0, "B": 0, "C": 0, "D": 0, "other": 0}
    subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}
    predictions = []

    for (content, _seq), item in zip(completions, data):
        answer = extract_mmlu_answer(content)
        target = item["answer"]
        subject = item["subject"]
        is_correct = answer == target
        if is_correct:
            correct += 1
        answers[answer if answer in ("A", "B", "C", "D") else "other"] += 1

        subject_correct[subject] = subject_correct.get(subject, 0) + int(is_correct)
        subject_total[subject] = subject_total.get(subject, 0) + 1

        predictions.append({
            "question": item["question"][:200],
            "subject": subject,
            "prediction": answer,
            "target": target,
            "correct": is_correct,
            "completion": content[:500],
        })

    total = len(data)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct, "total": total,
        "answers": answers,
        "subject_accuracy": {
            s: subject_correct[s] / subject_total[s]
            for s in sorted(subject_total)
        },
        "predictions": predictions,
    }


def evaluate_sgtr_benchmark(
    data: list[dict], sampling_client: Any, renderer: Any, eval_params: Any,
    use_system_prompt: bool = False,
) -> dict:
    """Run SGTR benchmark evaluation on loaded data.

    Like evaluate_val() but for cross-eval: accuracy grouped by format.
    """
    def build_prompt(item):
        return build_conversation(item, use_system_prompt)

    completions = _collect_completions(
        data, sampling_client, renderer, eval_params, build_prompt,
    )

    correct = 0
    answers = {"1": 0, "2": 0, "other": 0}
    format_correct: dict[str, int] = {}
    format_total: dict[str, int] = {}
    predictions = []

    for (content, _seq), item in zip(completions, data):
        answer = extract_answer(content)
        target = item["target"]
        fmt = item.get("format", "unknown")
        is_correct = answer == target
        if is_correct:
            correct += 1
        answers[answer if answer in ("1", "2") else "other"] += 1

        format_correct[fmt] = format_correct.get(fmt, 0) + int(is_correct)
        format_total[fmt] = format_total.get(fmt, 0) + 1

        predictions.append({
            "prompt": item["prompt"][:200] if isinstance(item["prompt"], str) else "(multi-turn)",
            "format": fmt,
            "prediction": answer,
            "target": target,
            "correct": is_correct,
            "completion": content[:500],
        })

    total = len(data)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct, "total": total,
        "answers": answers,
        "format_accuracy": {
            f: format_correct[f] / format_total[f]
            for f in sorted(format_total)
        },
        "predictions": predictions,
    }


def _save_benchmark_predictions(
    result: dict,
    cfg,
    *,
    epoch: int,
    step: int,
    eval_trigger: str,
    run_dir: str,
    extra_fields: dict,
) -> None:
    """Save benchmark predictions to JSON."""
    pred_dir = Path(run_dir) / "benchmark_predictions" / cfg.name
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / (
        f"{make_eval_artifact_name(epoch=epoch, step=step, eval_trigger=eval_trigger)}.json"
    )
    payload = {
        "epoch": epoch,
        "step": step,
        "eval_trigger": eval_trigger,
        "name": cfg.name,
        "type": cfg.type,
        **extra_fields,
        "accuracy": result["accuracy"],
        "predictions": result["predictions"],
    }
    # Include any per-type breakdown
    for key in ("format_accuracy", "subject_accuracy"):
        if key in result:
            payload[key] = result[key]
    with open(pred_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.debug(f"  benchmark predictions saved to {pred_path}")


def _log_benchmark_result(cfg_name: str, result: dict) -> None:
    """Log benchmark result and return per-answer metrics dict."""
    ans = result["answers"]
    parts = ",".join(f"{k}:{v}" for k, v in ans.items() if k != "other")
    logger.info(
        f"  benchmark/{cfg_name}: {result['correct']}/{result['total']} "
        f"= {result['accuracy']:.1%} | answers={{{parts},?:{ans['other']}}}"
    )


def _answer_pct_metrics(cfg_name: str, result: dict) -> dict[str, float]:
    """Build per-answer percentage metrics for a benchmark result."""
    total = max(result["total"], 1)
    metrics = {f"benchmark/{cfg_name}/accuracy": result["accuracy"]}
    for key, count in result["answers"].items():
        suffix = "other_pct" if key == "other" else f"answers_{key}_pct"
        metrics[f"benchmark/{cfg_name}/{suffix}"] = count / total
    return metrics


def run_val_eval(
    val_prompts: list[dict],
    ctx: TinkerContext,
    step: int,
    epoch: int,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
    eval_trigger: str = "epoch",
    diagnostic_num_examples: int = 0,
    diagnostic_example_ids: list[str] | None = None,
) -> dict | None:
    """Run full validation: accuracy, NLL, logging, and prediction saving."""
    if not val_prompts:
        return None

    val_sampling = ctx.training_client.save_weights_and_get_sampling_client()
    val_result = evaluate_val(
        val_prompts, val_sampling, ctx.renderer, ctx.eval_params, use_system_prompt,
    )
    val_result["nll"] = compute_val_nll(val_prompts, ctx, use_system_prompt)
    log_val_result(val_result)
    log_val_metrics(ctx.ml_logger, val_result, step=step)
    if run_dir:
        save_val_predictions(
            val_result,
            run_dir,
            epoch=epoch,
            step=step,
            eval_trigger=eval_trigger,
        )
        diagnostic_items = select_binary_diagnostic_items(
            val_prompts,
            num_examples=diagnostic_num_examples,
            example_ids=diagnostic_example_ids,
        )
        if diagnostic_items:
            diagnostic_payload = compute_binary_margin_diagnostics(
                diagnostic_items,
                sampling_client=val_sampling,
                renderer=ctx.renderer,
                tokenizer=ctx.tokenizer,
                eval_params=ctx.eval_params,
                use_system_prompt=use_system_prompt,
            )
            diagnostic_payload.update(
                {
                    "epoch": epoch,
                    "step": step,
                    "eval_trigger": eval_trigger,
                }
            )
            save_val_diagnostics(
                diagnostic_payload,
                run_dir,
                epoch=epoch,
                step=step,
                eval_trigger=eval_trigger,
            )
            summary = diagnostic_payload["summary"]
            ctx.ml_logger.log_metrics(
                {
                    "val_diag/accuracy": float(summary["accuracy"]),
                    "val_diag/mean_margin_1_minus_2": float(
                        summary["mean_margin_1_minus_2"]
                    ),
                    "val_diag/predicted_1_pct": (
                        float(summary["predicted_1_count"])
                        / max(int(summary["num_examples"]), 1)
                    ),
                },
                step=step,
            )
    return val_result


def run_train_panel_eval(
    train_prompts: list[dict],
    ctx: TinkerContext,
    step: int,
    epoch: int,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
    eval_trigger: str = "epoch",
    diagnostic_num_examples: int = 0,
    diagnostic_example_ids: list[str] | None = None,
) -> dict | None:
    """Run a fixed train-panel eval to compare with raw batch train loss."""
    diagnostic_items = select_binary_diagnostic_items(
        train_prompts,
        num_examples=diagnostic_num_examples,
        example_ids=diagnostic_example_ids,
    )
    if not diagnostic_items:
        return None

    train_sampling = ctx.training_client.save_weights_and_get_sampling_client()
    train_result = evaluate_val(
        diagnostic_items,
        train_sampling,
        ctx.renderer,
        ctx.eval_params,
        use_system_prompt,
    )
    train_result["nll"] = compute_val_nll(diagnostic_items, ctx, use_system_prompt)
    log_binary_eval_result("train_panel", train_result)
    log_split_metrics(ctx.ml_logger, train_result, step=step, prefix="train_panel")

    if run_dir:
        save_split_predictions(
            train_result,
            run_dir,
            split_name="train_panel",
            epoch=epoch,
            step=step,
            eval_trigger=eval_trigger,
        )
        diagnostic_payload = compute_binary_margin_diagnostics(
            diagnostic_items,
            sampling_client=train_sampling,
            renderer=ctx.renderer,
            tokenizer=ctx.tokenizer,
            eval_params=ctx.eval_params,
            use_system_prompt=use_system_prompt,
        )
        diagnostic_payload.update(
            {
                "epoch": epoch,
                "step": step,
                "eval_trigger": eval_trigger,
            }
        )
        save_split_diagnostics(
            diagnostic_payload,
            run_dir,
            split_name="train_panel",
            epoch=epoch,
            step=step,
            eval_trigger=eval_trigger,
        )
        summary = diagnostic_payload["summary"]
        ctx.ml_logger.log_metrics(
            {
                "train_panel_diag/accuracy": float(summary["accuracy"]),
                "train_panel_diag/mean_margin_1_minus_2": float(
                    summary["mean_margin_1_minus_2"]
                ),
                "train_panel_diag/predicted_1_pct": (
                    float(summary["predicted_1_count"])
                    / max(int(summary["num_examples"]), 1)
                ),
            },
            step=step,
        )
    return train_result


def run_benchmark_evals(
    configs,
    ctx: TinkerContext,
    step: int,
    epoch: int,
    total_epochs: int,
    schedule_index: int | None = None,
    schedule_total: int | None = None,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
    eval_trigger: str = "epoch",
) -> None:
    """Run all configured benchmark evals that are due this epoch."""
    if not configs:
        return

    current_index = schedule_index if schedule_index is not None else epoch
    total_index = schedule_total if schedule_total is not None else total_epochs
    due = [
        cfg for cfg in configs
        if should_run_benchmark(cfg.schedule, cfg.frequency, current_index, total_index)
    ]
    if not due:
        return

    sampling_client = ctx.training_client.save_weights_and_get_sampling_client()
    run_benchmark_configs(
        due,
        sampling_client=sampling_client,
        renderer=ctx.renderer,
        eval_params=ctx.eval_params,
        ml_logger=ctx.ml_logger,
        step=step,
        epoch=epoch,
        run_dir=run_dir,
        use_system_prompt=use_system_prompt,
        eval_trigger=eval_trigger,
    )


def run_benchmark_configs(
    configs,
    *,
    sampling_client: Any,
    renderer: Any,
    eval_params: Any,
    ml_logger: Any,
    step: int,
    epoch: int,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
    eval_trigger: str = "epoch",
) -> None:
    """Run a concrete list of benchmark configs immediately."""
    if not configs:
        return

    # Accumulate all metrics and log once to avoid W&B step-overwrite issue
    all_metrics: dict[str, float] = {}

    for cfg in configs:
        logger.info(
            f"Running benchmark eval: {cfg.name} (type={cfg.type}, epoch={epoch})"
        )

        data = load_benchmark_data(cfg.data_file)
        data = subsample(data, cfg.num_samples)

        if cfg.type == "sgtr":
            result = evaluate_sgtr_benchmark(
                data, sampling_client, renderer, eval_params,
                use_system_prompt=use_system_prompt,
            )
            extra = {}
        else:
            if not cfg.cot:
                from tinker import types
                mmlu_params = types.SamplingParams(
                    max_tokens=128,
                    stop=eval_params.stop,
                    temperature=eval_params.temperature,
                )
            else:
                mmlu_params = eval_params
            result = evaluate_benchmark(
                data, sampling_client, renderer, mmlu_params, cot=cfg.cot,
            )
            extra = {"cot": cfg.cot}

        _log_benchmark_result(cfg.name, result)
        all_metrics.update(_answer_pct_metrics(cfg.name, result))

        if run_dir:
            _save_benchmark_predictions(
                result,
                cfg,
                epoch=epoch,
                step=step,
                eval_trigger=eval_trigger,
                run_dir=run_dir,
                extra_fields=extra,
            )

    if all_metrics:
        ml_logger.log_metrics(all_metrics, step=step)
