"""Local-model evaluation helpers for validation and benchmark runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

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

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sgtr_rl.local_sft import LocalTrainingContext


def _generate_completions(
    items: list[dict],
    ctx: LocalTrainingContext,
    *,
    build_prompt: Callable[[dict], list[dict]],
    max_new_tokens: int,
) -> list[str]:
    tokenizer = ctx.tokenizer
    model = ctx.model
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    batch_size = ctx.runtime.local.eval_batch_size
    outputs: list[str] = []
    try:
        model.eval()
        original_use_cache = getattr(model.config, "use_cache", True)
        model.config.use_cache = True
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            prompt_texts = [
                ctx.tokenizer.apply_chat_template(
                    build_prompt(item),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if getattr(ctx.tokenizer, "chat_template", None)
                else "\n\n".join(
                    [f"{msg['role'].upper()}: {msg['content']}" for msg in build_prompt(item)]
                    + ["ASSISTANT:"]
                )
                for item in batch
            ]
            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ctx.runtime.local.max_seq_length,
            )
            encoded = {key: value.to(ctx.device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            input_width = encoded["input_ids"].shape[1]
            for row in generated:
                completion_ids = row[input_width:]
                outputs.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
    finally:
        model.config.use_cache = original_use_cache
        tokenizer.padding_side = old_padding_side
    return outputs


def compute_val_nll(
    val_prompts: list[dict],
    ctx: "LocalTrainingContext",
    *,
    use_system_prompt: bool = False,
) -> float:
    """Compute mean NLL on validation examples."""
    from sgtr_rl.local_sft import _make_training_example

    if not val_prompts:
        return 0.0

    total_loss = 0.0
    total_tokens = 0
    batch_size = ctx.runtime.local.eval_batch_size
    tokenizer = ctx.tokenizer
    tokenizer.padding_side = "right"
    ctx.model.eval()

    for start in range(0, len(val_prompts), batch_size):
        batch = val_prompts[start : start + batch_size]
        features = [
            _make_training_example(
                item,
                tokenizer=tokenizer,
                use_system_prompt=use_system_prompt,
                max_seq_length=ctx.runtime.local.max_seq_length,
            )
            for item in batch
        ]
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids = []
        labels = []
        attention_mask = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)

        tensors = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=ctx.device),
            "labels": torch.tensor(labels, dtype=torch.long, device=ctx.device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=ctx.device),
        }

        with torch.inference_mode():
            outputs = ctx.model(**tensors)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = tensors["labels"][:, 1:].contiguous()
        loss_sum = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        token_count = shift_labels.ne(-100).sum().item()
        total_loss += float(loss_sum.item())
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def evaluate_val(
    val_prompts: list[dict],
    ctx: "LocalTrainingContext",
    *,
    use_system_prompt: bool = False,
) -> dict:
    """Run greedy local validation evaluation."""

    def build_prompt(item: dict) -> list[dict]:
        return build_conversation(item, use_system_prompt)

    completions = _generate_completions(
        val_prompts,
        ctx,
        build_prompt=build_prompt,
        max_new_tokens=ctx.config.max_completion_length,
    )

    correct = 0
    answers = {"1": 0, "2": 0, "other": 0}
    predictions = []
    for content, item in zip(completions, val_prompts):
        answer = extract_answer(content)
        target = item["target"]
        is_correct = answer == target
        if is_correct:
            correct += 1
        answers[answer if answer in ("1", "2") else "other"] += 1
        predictions.append(
            {
                "id": item.get("id", ""),
                "prediction": answer,
                "target": target,
                "correct": is_correct,
                "completion": content[:500],
            }
        )

    total = len(val_prompts)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "answers": answers,
        "predictions": predictions,
    }


def evaluate_benchmark(
    data: list[dict],
    ctx: "LocalTrainingContext",
    *,
    cot: bool,
) -> dict:
    """Run local MMLU evaluation."""

    def build_prompt(item: dict) -> list[dict]:
        return [{"role": "user", "content": format_mmlu_prompt(item, cot=cot)}]

    max_new_tokens = ctx.config.max_completion_length if cot else 128
    completions = _generate_completions(
        data,
        ctx,
        build_prompt=build_prompt,
        max_new_tokens=max_new_tokens,
    )

    correct = 0
    answers = {"A": 0, "B": 0, "C": 0, "D": 0, "other": 0}
    subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}
    predictions = []

    for content, item in zip(completions, data):
        answer = extract_mmlu_answer(content)
        target = item["answer"]
        subject = item["subject"]
        is_correct = answer == target
        if is_correct:
            correct += 1
        answers[answer if answer in ("A", "B", "C", "D") else "other"] += 1
        subject_correct[subject] = subject_correct.get(subject, 0) + int(is_correct)
        subject_total[subject] = subject_total.get(subject, 0) + 1
        predictions.append(
            {
                "question": item["question"][:200],
                "subject": subject,
                "prediction": answer,
                "target": target,
                "correct": is_correct,
                "completion": content[:500],
            }
        )

    total = len(data)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "answers": answers,
        "subject_accuracy": {
            subject: subject_correct[subject] / subject_total[subject]
            for subject in sorted(subject_total)
        },
        "predictions": predictions,
    }


def evaluate_sgtr_benchmark(
    data: list[dict],
    ctx: "LocalTrainingContext",
    *,
    use_system_prompt: bool = False,
) -> dict:
    """Run local SGTR cross-eval benchmark."""

    def build_prompt(item: dict) -> list[dict]:
        return build_conversation(item, use_system_prompt)

    completions = _generate_completions(
        data,
        ctx,
        build_prompt=build_prompt,
        max_new_tokens=ctx.config.max_completion_length,
    )

    correct = 0
    answers = {"1": 0, "2": 0, "other": 0}
    format_correct: dict[str, int] = {}
    format_total: dict[str, int] = {}
    predictions = []

    for content, item in zip(completions, data):
        answer = extract_answer(content)
        target = item["target"]
        fmt = item.get("format", "unknown")
        is_correct = answer == target
        if is_correct:
            correct += 1
        answers[answer if answer in ("1", "2") else "other"] += 1
        format_correct[fmt] = format_correct.get(fmt, 0) + int(is_correct)
        format_total[fmt] = format_total.get(fmt, 0) + 1
        predictions.append(
            {
                "prompt": item["prompt"][:200]
                if isinstance(item["prompt"], str)
                else "(multi-turn)",
                "format": fmt,
                "prediction": answer,
                "target": target,
                "correct": is_correct,
                "completion": content[:500],
            }
        )

    total = len(data)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "answers": answers,
        "format_accuracy": {
            fmt: format_correct[fmt] / format_total[fmt] for fmt in sorted(format_total)
        },
        "predictions": predictions,
    }


def _save_benchmark_predictions(
    result: dict,
    cfg,
    *,
    epoch: int,
    run_dir: str,
    extra_fields: dict,
) -> None:
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
    for key in ("format_accuracy", "subject_accuracy"):
        if key in result:
            payload[key] = result[key]
    with open(pred_path, "w") as f:
        json.dump(payload, f, indent=2)


def _log_benchmark_result(cfg_name: str, result: dict) -> None:
    answers = result["answers"]
    parts = ",".join(f"{key}:{value}" for key, value in answers.items() if key != "other")
    logger.info(
        "  benchmark/%s: %s/%s = %.1f%% | answers={%s,?:%s}",
        cfg_name,
        result["correct"],
        result["total"],
        result["accuracy"] * 100.0,
        parts,
        answers["other"],
    )


def _answer_pct_metrics(cfg_name: str, result: dict) -> dict[str, float]:
    total = max(result["total"], 1)
    metrics = {f"benchmark/{cfg_name}/accuracy": result["accuracy"]}
    for key, count in result["answers"].items():
        suffix = "other_pct" if key == "other" else f"answers_{key}_pct"
        metrics[f"benchmark/{cfg_name}/{suffix}"] = count / total
    return metrics


def run_val_eval(
    val_prompts: list[dict],
    ctx: "LocalTrainingContext",
    *,
    step: int,
    epoch: int,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
) -> dict | None:
    if not val_prompts:
        return None

    val_result = evaluate_val(val_prompts, ctx, use_system_prompt=use_system_prompt)
    val_result["nll"] = compute_val_nll(val_prompts, ctx, use_system_prompt=use_system_prompt)
    log_val_result(val_result)
    log_val_metrics(ctx.metrics_logger, val_result, step=step)
    if run_dir:
        save_val_predictions(val_result, run_dir, epoch)
    return val_result


def run_benchmark_evals(
    configs,
    ctx: "LocalTrainingContext",
    *,
    step: int,
    epoch: int,
    total_epochs: int,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
) -> None:
    if not configs:
        return

    due = [
        cfg
        for cfg in configs
        if should_run_benchmark(cfg.schedule, cfg.frequency, epoch, total_epochs)
    ]
    if not due:
        return

    all_metrics: dict[str, float] = {}
    for cfg in due:
        logger.info("Running benchmark eval: %s (type=%s, epoch=%s)", cfg.name, cfg.type, epoch)
        data = subsample(load_benchmark_data(cfg.data_file), cfg.num_samples)
        if cfg.type == "sgtr":
            result = evaluate_sgtr_benchmark(
                data,
                ctx,
                use_system_prompt=use_system_prompt,
            )
            extra = {}
        else:
            result = evaluate_benchmark(data, ctx, cot=cfg.cot)
            extra = {"cot": cfg.cot}

        _log_benchmark_result(cfg.name, result)
        all_metrics.update(_answer_pct_metrics(cfg.name, result))
        if run_dir:
            _save_benchmark_predictions(
                result,
                cfg,
                epoch=epoch,
                run_dir=run_dir,
                extra_fields=extra,
            )

    if all_metrics:
        ctx.metrics_logger.log_metrics(all_metrics, step=step)
