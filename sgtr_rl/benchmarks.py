"""Benchmark evaluation (MMLU and SGTR) during training.

Provides configurable benchmark evals that run alongside the existing
validation eval loop. Mirrors patterns from eval.py.
"""

import functools
import json
import logging
import random
import re
from pathlib import Path

from sgtr_rl.data import build_conversation, flip_target, load_jsonl

logger = logging.getLogger(__name__)


def _subsample(data: list[dict], num_samples: int | None, seed: int = 42) -> list[dict]:
    """Deterministically subsample data. Returns all data if num_samples is None or >= len."""
    if num_samples is None or num_samples >= len(data):
        return data
    rng = random.Random(seed)
    return rng.sample(data, num_samples)


@functools.lru_cache(maxsize=32)
def _load_benchmark_cached(path: str) -> tuple[dict, ...]:
    """Load and cache benchmark JSONL data. Returns tuple for hashability."""
    items = load_jsonl(path)
    logger.info(f"Loaded {len(items)} benchmark items from {path}")
    return tuple(items)


def load_benchmark_data(path: str) -> list[dict]:
    """Load benchmark JSONL data, caching by path."""
    return list(_load_benchmark_cached(path))



def format_mmlu_prompt(item: dict, cot: bool = False) -> str:
    """Format an MMLU item into a prompt string.

    Uses the same format as inspect-ai's MMLU 0-shot template.

    Args:
        item: Dict with 'question', 'choices', 'subject'.
        cot: If True, ask for chain-of-thought reasoning.

    Returns:
        Formatted prompt string.
    """
    choices_str = "\n".join(
        f"{letter}) {text}"
        for letter, text in zip("ABCD", item["choices"])
    )
    if cot:
        instruction = (
            "Answer the following multiple choice question. The last line of "
            "your response should be of the following format: 'ANSWER: $LETTER' "
            "(without quotes) where LETTER is one of A,B,C,D. Think step by "
            "step before answering."
        )
    else:
        instruction = (
            "Answer the following multiple choice question. The entire content "
            "of your response should be of the following format: 'ANSWER: $LETTER' "
            "(without quotes) where LETTER is one of A,B,C,D."
        )

    return f"{instruction}\n\n{item['question']}\n\n{choices_str}"


def extract_mmlu_answer(text: str) -> str | None:
    """Extract A/B/C/D answer from model completion.

    Uses inspect-ai compatible extraction with anti-cheating:
    1. "ANSWER: X" pattern — reject if multiple ANSWER: lines disagree
    2. Fallback: last standalone A/B/C/D, but reject if all 4 present
    3. Bare single-character response
    """
    text = text.strip()

    # Strategy 1: "ANSWER: X" pattern (inspect-ai format)
    # Find ALL occurrences — reject if they disagree
    answer_matches = re.findall(r"(?i)ANSWER\s*[:=]\s*([A-Da-d])", text)
    if answer_matches:
        unique = set(m.upper() for m in answer_matches)
        if len(unique) == 1:
            return unique.pop()
        return None  # multiple different answers — invalid

    # Strategy 2: last standalone A-D, reject if all 4 letters present
    matches = re.findall(r"\b([A-Da-d])\b", text)
    abcd_matches = [m.upper() for m in matches if m.upper() in "ABCD"]
    if abcd_matches:
        if len(set(abcd_matches)) >= 4:
            return None  # all choices mentioned — not a real answer
        return abcd_matches[-1]

    # Strategy 3: bare single character
    if text.upper() in ("A", "B", "C", "D"):
        return text.upper()

    return None


def evaluate_benchmark(
    data: list[dict],
    sampling_client,
    renderer,
    eval_params,
    cot: bool,
) -> dict:
    """Run benchmark evaluation on loaded data.

    Fires all async sampling requests, then collects results.

    Returns:
        Dict with accuracy, correct, total, per-subject breakdown,
        answer distribution, and per-item predictions.
    """
    from tinker_cookbook import renderers as r

    # Fire all requests
    futures = []
    for item in data:
        prompt_text = format_mmlu_prompt(item, cot=cot)
        convo = [{"role": "user", "content": prompt_text}]
        model_input = renderer.build_generation_prompt(convo)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=eval_params,
        )
        futures.append(future)

    # Collect results
    correct = 0
    answers = {"A": 0, "B": 0, "C": 0, "D": 0, "other": 0}
    subject_correct: dict[str, int] = {}
    subject_total: dict[str, int] = {}
    predictions = []

    for future, item in zip(futures, data):
        result = future.result()
        sequence = result.sequences[0]
        parsed_msg, _ = renderer.parse_response(sequence.tokens)
        content = r.get_text_content(parsed_msg)
        answer = extract_mmlu_answer(content)
        target = item["answer"]
        subject = item["subject"]

        is_correct = answer == target
        if is_correct:
            correct += 1

        if answer in ("A", "B", "C", "D"):
            answers[answer] += 1
        else:
            answers["other"] += 1

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
    accuracy = correct / total if total else 0.0

    subject_accuracy = {
        s: subject_correct[s] / subject_total[s]
        for s in sorted(subject_total)
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "answers": answers,
        "subject_accuracy": subject_accuracy,
        "predictions": predictions,
    }


def evaluate_sgtr_benchmark(
    data: list[dict],
    sampling_client,
    renderer,
    eval_params,
    flip_targets: bool = False,
    use_system_prompt: bool = False,
) -> dict:
    """Run SGTR benchmark evaluation on loaded data.

    Like evaluate_val() from eval.py but designed for cross-eval: prompts are
    pre-built in JSONL, accuracy is grouped by format, and
    flip_targets swaps the comparison target at eval time.

    Returns:
        Dict with accuracy, correct, total, per-format breakdown,
        answer distribution, and per-item predictions.
    """
    from tinker_cookbook import renderers as r

    from sgtr_rl.answer import extract_answer

    # Fire all requests
    futures = []
    for item in data:
        convo = build_conversation(item, use_system_prompt)
        model_input = renderer.build_generation_prompt(convo)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=eval_params,
        )
        futures.append(future)

    # Collect results
    correct = 0
    answers = {"1": 0, "2": 0, "other": 0}
    format_correct: dict[str, int] = {}
    format_total: dict[str, int] = {}
    predictions = []

    for future, item in zip(futures, data):
        result = future.result()
        sequence = result.sequences[0]
        parsed_msg, _ = renderer.parse_response(sequence.tokens)
        content = r.get_text_content(parsed_msg)
        answer = extract_answer(content)
        target = item["target"]
        if flip_targets:
            target = flip_target(target)
        fmt = item.get("format", "unknown")

        is_correct = answer == target
        if is_correct:
            correct += 1

        if answer in ("1", "2"):
            answers[answer] += 1
        else:
            answers["other"] += 1

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
    accuracy = correct / total if total else 0.0

    format_accuracy = {
        f: format_correct[f] / format_total[f]
        for f in sorted(format_total)
    }

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "answers": answers,
        "format_accuracy": format_accuracy,
        "predictions": predictions,
    }


def should_run_benchmark(schedule: str, frequency: int, epoch: int, total_epochs: int) -> bool:
    """Check if a benchmark should run at this epoch.

    Args:
        schedule: "every_epoch", "every_N_epochs", or "end_only".
        frequency: For "every_N_epochs", the N value.
        epoch: Current epoch (0 = baseline, 1 = after first epoch, etc.).
        total_epochs: Total number of training epochs.

    Returns:
        True if the benchmark should run.
    """
    is_final = epoch == total_epochs

    # Always run baseline (epoch 0) so every benchmark has an untrained reference
    if epoch == 0:
        return True

    if schedule == "every_epoch":
        return True
    elif schedule == "end_only":
        return is_final
    elif schedule.startswith("every_") and schedule.endswith("_epochs"):
        # Matches "every_N_epochs", "every_5_epochs", etc.
        if epoch == 0:
            return False
        return (epoch % frequency == 0) or is_final
    else:
        logger.warning(f"Unknown benchmark schedule: {schedule!r}")
        return False


def run_benchmark_evals(
    configs,
    training_client,
    renderer,
    eval_params,
    ml_logger,
    step: int,
    epoch: int,
    total_epochs: int,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
) -> None:
    """Run all configured benchmark evals that are due this epoch."""
    if not configs:
        return

    # Find which benchmarks are due
    due = [
        cfg for cfg in configs
        if should_run_benchmark(cfg.schedule, cfg.frequency, epoch, total_epochs)
    ]
    if not due:
        return

    # Get a single sampling client for all benchmarks
    sampling_client = training_client.save_weights_and_get_sampling_client()

    # Accumulate all metrics and log once to avoid W&B step-overwrite issue
    all_metrics: dict[str, float] = {}

    for cfg in due:
        logger.info(f"Running benchmark eval: {cfg.name} (type={cfg.type}, epoch={epoch})")

        data = load_benchmark_data(cfg.data_file)
        data = _subsample(data, cfg.num_samples)

        if cfg.type == "sgtr":
            result = evaluate_sgtr_benchmark(
                data, sampling_client, renderer, eval_params,
                flip_targets=cfg.flip_targets,
                use_system_prompt=use_system_prompt,
            )

            # Log to console (1/2 distribution)
            total = max(result["total"], 1)
            logger.info(
                f"  benchmark/{cfg.name}: {result['correct']}/{result['total']} "
                f"= {result['accuracy']:.1%} | "
                f"answers={{1:{result['answers']['1']},"
                f"2:{result['answers']['2']},"
                f"?:{result['answers']['other']}}}"
            )

            # Accumulate metrics
            all_metrics.update({
                f"benchmark/{cfg.name}/accuracy": result["accuracy"],
                f"benchmark/{cfg.name}/answers_1_pct": result["answers"]["1"] / total,
                f"benchmark/{cfg.name}/answers_2_pct": result["answers"]["2"] / total,
                f"benchmark/{cfg.name}/answers_other_pct": result["answers"]["other"] / total,
            })

            # Save predictions
            if run_dir:
                pred_dir = Path(run_dir) / "benchmark_predictions" / cfg.name
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_path = pred_dir / f"epoch_{epoch}.json"
                with open(pred_path, "w") as f:
                    json.dump({
                        "epoch": epoch,
                        "name": cfg.name,
                        "type": cfg.type,
                        "flip_targets": cfg.flip_targets,
                        "accuracy": result["accuracy"],
                        "format_accuracy": result.get("format_accuracy", {}),
                        "predictions": result["predictions"],
                    }, f, indent=2)
                logger.debug(f"  benchmark predictions saved to {pred_path}")

        else:
            # MMLU
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

            # Log to console (A/B/C/D distribution)
            total = max(result["total"], 1)
            logger.info(
                f"  benchmark/{cfg.name}: {result['correct']}/{result['total']} "
                f"= {result['accuracy']:.1%} | "
                f"answers={{A:{result['answers']['A']},B:{result['answers']['B']},"
                f"C:{result['answers']['C']},D:{result['answers']['D']},"
                f"?:{result['answers']['other']}}}"
            )

            # Accumulate metrics
            all_metrics.update({
                f"benchmark/{cfg.name}/accuracy": result["accuracy"],
                f"benchmark/{cfg.name}/answers_A_pct": result["answers"]["A"] / total,
                f"benchmark/{cfg.name}/answers_B_pct": result["answers"]["B"] / total,
                f"benchmark/{cfg.name}/answers_C_pct": result["answers"]["C"] / total,
                f"benchmark/{cfg.name}/answers_D_pct": result["answers"]["D"] / total,
                f"benchmark/{cfg.name}/answers_other_pct": result["answers"]["other"] / total,
            })

            # Save predictions
            if run_dir:
                pred_dir = Path(run_dir) / "benchmark_predictions" / cfg.name
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_path = pred_dir / f"epoch_{epoch}.json"
                with open(pred_path, "w") as f:
                    json.dump({
                        "epoch": epoch,
                        "name": cfg.name,
                        "type": cfg.type,
                        "cot": cfg.cot,
                        "accuracy": result["accuracy"],
                        "subject_accuracy": result.get("subject_accuracy", {}),
                        "predictions": result["predictions"],
                    }, f, indent=2)
                logger.debug(f"  benchmark predictions saved to {pred_path}")

    # Log all benchmark metrics in a single call to avoid W&B step-overwrite
    if all_metrics:
        ml_logger.log_metrics(all_metrics, step=step)
