"""Reward functions for SGTR-RL training."""

import re


def sgtr_binary_reward(completions: list[str], targets: list[str]) -> list[float]:
    """Compute binary reward for SGTR completions.

    Extracts the final answer token ("1" or "2") from each completion and
    compares it to the target.  Works with both bare answers ("2") and
    chain-of-thought followed by an answer ("reasoning...\\n\\nAnswer: 2").

    Args:
        completions: Generated text strings from the model.
        targets: Ground-truth answer tokens ("1" or "2"), one per completion.

    Returns:
        List of rewards: 1.0 if the extracted answer matches the target,
        0.0 otherwise (including when no answer is found).
    """
    rewards = []
    for completion, target in zip(completions, targets):
        answer = _extract_answer(completion)
        rewards.append(1.0 if answer == target else 0.0)
    return rewards


def _extract_answer(text: str) -> str | None:
    """Extract the answer token from a completion.

    Tries several strategies in order:
    1. Explicit "Answer: <digit>" pattern (case-insensitive)
    2. Last occurrence of a standalone "1" or "2" in the text
    3. If the entire (stripped) text is just "1" or "2"

    Returns:
        "1" or "2" if found, None otherwise.
    """
    text = text.strip()

    # Strategy 1: explicit answer pattern
    match = re.search(r"(?i)answer\s*[:=]\s*([12])", text)
    if match:
        return match.group(1)

    # Strategy 2: last standalone "1" or "2" (word boundary)
    matches = re.findall(r"\b([12])\b", text)
    if matches:
        return matches[-1]

    # Strategy 3: bare single-character response
    if text in ("1", "2"):
        return text

    return None
