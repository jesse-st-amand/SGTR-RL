"""Reward functions for SGTR-RL training."""

from sgtr_rl.answer import extract_answer


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
        answer = extract_answer(completion)
        rewards.append(1.0 if answer == target else 0.0)
    return rewards
