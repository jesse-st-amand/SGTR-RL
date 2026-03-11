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
from sgtr_rl.metrics import log_val_metrics, log_val_result, save_val_predictions
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


def compute_val_nll(
    val_prompts: list[dict], ctx: TinkerContext,
    use_system_prompt: bool = False,
) -> float:
    """Compute mean NLL on validation set via forward pass.

    Builds SFT-style datums, runs forward_backward to get loss, then clears
    accumulated gradients with a zero-lr optimizer step.
    """
    from tinker import types
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

    fwd_bwd_result = ctx.training_client.forward_backward(
        datums, loss_fn="cross_entropy"
    ).result()

    # Clear accumulated gradients without changing weights
    zero_adam = types.AdamParams(
        learning_rate=0.0, beta1=0.9, beta2=0.95, eps=1e-8
    )
    ctx.training_client.optim_step(zero_adam).result()

    logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
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
    result: dict, cfg, epoch: int, run_dir: str, extra_fields: dict,
) -> None:
    """Save benchmark predictions to JSON."""
    pred_dir = Path(run_dir) / "benchmark_predictions" / cfg.name
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"epoch_{epoch}.json"
    payload = {
        "epoch": epoch,
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
        save_val_predictions(val_result, run_dir, epoch)
    return val_result


def run_benchmark_evals(
    configs,
    ctx: TinkerContext,
    step: int,
    epoch: int,
    total_epochs: int,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
) -> None:
    """Run all configured benchmark evals that are due this epoch."""
    if not configs:
        return

    due = [
        cfg for cfg in configs
        if should_run_benchmark(cfg.schedule, cfg.frequency, epoch, total_epochs)
    ]
    if not due:
        return

    sampling_client = ctx.training_client.save_weights_and_get_sampling_client()

    # Accumulate all metrics and log once to avoid W&B step-overwrite issue
    all_metrics: dict[str, float] = {}

    for cfg in due:
        logger.info(
            f"Running benchmark eval: {cfg.name} (type={cfg.type}, epoch={epoch})"
        )

        data = load_benchmark_data(cfg.data_file)
        data = subsample(data, cfg.num_samples)

        if cfg.type == "sgtr":
            result = evaluate_sgtr_benchmark(
                data, sampling_client, ctx.renderer, ctx.eval_params,
                use_system_prompt=use_system_prompt,
            )
            extra = {}
        else:
            if not cfg.cot:
                from tinker import types
                mmlu_params = types.SamplingParams(
                    max_tokens=128,
                    stop=ctx.eval_params.stop,
                    temperature=ctx.eval_params.temperature,
                )
            else:
                mmlu_params = ctx.eval_params
            result = evaluate_benchmark(
                data, sampling_client, ctx.renderer, mmlu_params, cot=cfg.cot,
            )
            extra = {"cot": cfg.cot}

        _log_benchmark_result(cfg.name, result)
        all_metrics.update(_answer_pct_metrics(cfg.name, result))

        if run_dir:
            _save_benchmark_predictions(result, cfg, epoch, run_dir, extra)

    if all_metrics:
        ctx.ml_logger.log_metrics(all_metrics, step=step)
