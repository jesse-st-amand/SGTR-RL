"""Benchmark evaluation (MMLU) during training.

Provides configurable benchmark evals that run alongside the existing
validation eval loop. Mirrors patterns from eval.py.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level cache: data_file path -> list of items
_benchmark_cache: dict[str, list[dict]] = {}


def load_benchmark_data(path: str) -> list[dict]:
    """Load benchmark JSONL data, caching by path."""
    if path in _benchmark_cache:
        return _benchmark_cache[path]
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    _benchmark_cache[path] = items
    logger.info(f"Loaded {len(items)} benchmark items from {path}")
    return items


def format_mmlu_prompt(item: dict, cot: bool = False) -> str:
    """Format an MMLU item into a prompt string.

    Args:
        item: Dict with 'question', 'choices', 'subject'.
        cot: If True, ask for chain-of-thought reasoning.

    Returns:
        Formatted prompt string.
    """
    choices_str = "\n".join(
        f"{letter}. {text}"
        for letter, text in zip("ABCD", item["choices"])
    )
    if cot:
        instruction = (
            'Think step by step, then provide your final answer as '
            '"Answer: X" where X is A, B, C, or D.'
        )
    else:
        instruction = "Answer with a single letter (A, B, C, or D)."

    return f"{item['question']}\n\n{choices_str}\n\n{instruction}"


def extract_mmlu_answer(text: str) -> str | None:
    """Extract A/B/C/D answer from model completion.

    Tries:
    1. Explicit "Answer: X" pattern (case-insensitive)
    2. Last standalone A/B/C/D in the text
    3. Bare single-character response
    """
    text = text.strip()

    # Strategy 1: explicit answer pattern
    match = re.search(r"(?i)answer\s*[:=]\s*([A-Da-d])", text)
    if match:
        return match.group(1).upper()

    # Strategy 2: last standalone A-D
    matches = re.findall(r"\b([A-Da-d])\b", text)
    # Filter to only A-D (exclude common words like "a" in lowercase context)
    # Use uppercase matches or matches at end of text
    abcd_matches = [m.upper() for m in matches if m.upper() in "ABCD"]
    if abcd_matches:
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
    elif schedule == "every_N_epochs":
        if epoch == 0:
            return False
        return (epoch % frequency == 0) or is_final
    elif schedule == "end_only":
        return is_final
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
) -> None:
    """Run all configured benchmark evals that are due this epoch.

    Single entry point called from trainers, parallel to run_val_eval.

    Args:
        configs: List of BenchmarkEvalConfig objects.
        training_client: Tinker training client.
        renderer: Tinker renderer.
        eval_params: SamplingParams (temperature=0 for greedy).
        ml_logger: Tinker cookbook ml_logger for wandb/metrics.
        step: Current global training step.
        epoch: Current epoch (0 = baseline).
        total_epochs: Total training epochs.
        run_dir: Run directory for saving predictions.
    """
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

    for cfg in due:
        logger.info(f"Running benchmark eval: {cfg.name} (epoch={epoch}, cot={cfg.cot})")

        data = load_benchmark_data(cfg.data_file)
        result = evaluate_benchmark(
            data, sampling_client, renderer, eval_params, cot=cfg.cot,
        )

        # Log to console
        total = max(result["total"], 1)
        logger.info(
            f"  benchmark/{cfg.name}: {result['correct']}/{result['total']} "
            f"= {result['accuracy']:.1%} | "
            f"answers={{A:{result['answers']['A']},B:{result['answers']['B']},"
            f"C:{result['answers']['C']},D:{result['answers']['D']},"
            f"?:{result['answers']['other']}}}"
        )

        # Log to wandb
        metrics = {
            f"benchmark/{cfg.name}/accuracy": result["accuracy"],
            f"benchmark/{cfg.name}/answers_A_pct": result["answers"]["A"] / total,
            f"benchmark/{cfg.name}/answers_B_pct": result["answers"]["B"] / total,
            f"benchmark/{cfg.name}/answers_C_pct": result["answers"]["C"] / total,
            f"benchmark/{cfg.name}/answers_D_pct": result["answers"]["D"] / total,
            f"benchmark/{cfg.name}/answers_other_pct": result["answers"]["other"] / total,
        }
        ml_logger.log_metrics(metrics, step=step)

        # Save predictions
        if run_dir:
            pred_dir = Path(run_dir) / "benchmark_predictions" / cfg.name
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_path = pred_dir / f"epoch_{epoch}.json"
            with open(pred_path, "w") as f:
                json.dump({
                    "epoch": epoch,
                    "name": cfg.name,
                    "cot": cfg.cot,
                    "accuracy": result["accuracy"],
                    "subject_accuracy": result["subject_accuracy"],
                    "predictions": result["predictions"],
                }, f, indent=2)
            logger.debug(f"  benchmark predictions saved to {pred_path}")
