"""Benchmark data loading, prompt formatting, and scheduling logic.

Pure functions with no Tinker dependencies. Tinker-based evaluation
lives in tinker_eval.py.
"""

import functools
import logging
import random
import re

from sgtr_rl.data import load_jsonl

logger = logging.getLogger(__name__)


def subsample(data: list[dict], num_samples: int | None, seed: int = 42) -> list[dict]:
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
    answer_matches = re.findall(r"(?i)ANSWER\s*[:=]\s*([A-Da-d])", text)
    if answer_matches:
        unique = set(m.upper() for m in answer_matches)
        if len(unique) == 1:
            return unique.pop()
        return None

    # Strategy 2: last standalone A-D, reject if all 4 letters present
    matches = re.findall(r"\b([A-Da-d])\b", text)
    abcd_matches = [m.upper() for m in matches if m.upper() in "ABCD"]
    if abcd_matches:
        if len(set(abcd_matches)) >= 4:
            return None
        return abcd_matches[-1]

    # Strategy 3: bare single character
    if text.upper() in ("A", "B", "C", "D"):
        return text.upper()

    return None


def should_run_benchmark(schedule: str, frequency: int, epoch: int, total_epochs: int) -> bool:
    """Check if a benchmark should run at this epoch.

    Args:
        schedule: "every_epoch", "every_N_epochs", or "end_only".
        frequency: For "every_N_epochs", the N value.
        epoch: Current epoch (0 = baseline, 1 = after first epoch, etc.).
        total_epochs: Total number of training epochs.
    """
    is_final = epoch == total_epochs

    if epoch == 0:
        return True

    if schedule == "every_epoch":
        return True
    elif schedule == "end_only":
        return is_final
    elif schedule.startswith("every_") and schedule.endswith("_epochs"):
        return (epoch % frequency == 0) or is_final
    else:
        logger.warning(f"Unknown benchmark schedule: {schedule!r}")
        return False
