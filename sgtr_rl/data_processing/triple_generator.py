"""Generate DPO training triples from evaluation samples.

This module creates (prompt, chosen, rejected) triples for DPO training
from SGTR evaluation data.
"""

from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

from .eval_loader import EvalSample, categorize_sample


@dataclass
class DPOTriple:
    """DPO training triple.

    Attributes:
        prompt: The input prompt
        chosen: Preferred response (correct reasoning + answer)
        rejected: Dispreferred response (incorrect reasoning + answer)
        metadata: Additional information about this triple
    """

    prompt: str
    chosen: str
    rejected: str
    metadata: dict


def format_response(sample: EvalSample, include_reasoning: bool = True) -> str:
    """Format a sample's response for DPO training.

    Args:
        sample: EvalSample with reasoning and answer
        include_reasoning: Whether to include CoT reasoning (default True)

    Returns:
        str: Formatted response string

    Example:
        >>> response = format_response(sample)
        # Returns: "{reasoning}\n\nAnswer: {answer}"
    """
    if include_reasoning and sample.reasoning:
        return f"{sample.reasoning}\n\nAnswer: {sample.answer}"
    else:
        return sample.answer


def create_dpo_triples(
    samples: list[EvalSample],
    include_reasoning: bool = True,
    require_same_prompt: bool = True,
) -> list[DPOTriple]:
    """Create DPO training triples from evaluation samples.

    Strategy:
    - Chosen: Correct samples (TP or TN)
    - Rejected: Incorrect samples (FP or FN)
    - Pair by matching prompts when possible

    Args:
        samples: List of EvalSamples from evaluations
        include_reasoning: Whether to include CoT reasoning in responses
        require_same_prompt: If True, only create triples where chosen/rejected
                           have the same prompt. If False, allow mismatched prompts.

    Returns:
        list[DPOTriple]: DPO training triples

    Example:
        >>> triples = create_dpo_triples(samples)
        >>> print(f"Created {len(triples)} DPO triples")
    """
    # Categorize samples
    correct_samples = []  # TP and TN
    incorrect_samples = []  # FP and FN

    for sample in samples:
        category = categorize_sample(sample)
        if category in ["TP", "TN"]:
            correct_samples.append(sample)
        else:  # FP or FN
            incorrect_samples.append(sample)

    triples = []

    if require_same_prompt:
        # Group by prompt for matching
        correct_by_prompt = defaultdict(list)
        incorrect_by_prompt = defaultdict(list)

        for sample in correct_samples:
            correct_by_prompt[sample.prompt].append(sample)

        for sample in incorrect_samples:
            incorrect_by_prompt[sample.prompt].append(sample)

        # Create triples by matching prompts
        for prompt in correct_by_prompt:
            if prompt not in incorrect_by_prompt:
                continue  # No matching incorrect sample for this prompt

            # Pair each correct with each incorrect for this prompt
            for correct_sample in correct_by_prompt[prompt]:
                for incorrect_sample in incorrect_by_prompt[prompt]:
                    triple = DPOTriple(
                        prompt=prompt,
                        chosen=format_response(correct_sample, include_reasoning),
                        rejected=format_response(incorrect_sample, include_reasoning),
                        metadata={
                            "chosen_sample_id": correct_sample.sample_id,
                            "rejected_sample_id": incorrect_sample.sample_id,
                            "chosen_category": categorize_sample(correct_sample),
                            "rejected_category": categorize_sample(incorrect_sample),
                            "evaluator_model": correct_sample.evaluator_model,
                            "chosen_generator": correct_sample.generator_model,
                            "rejected_generator": incorrect_sample.generator_model,
                            "dataset": correct_sample.dataset,
                            "experiment": correct_sample.experiment,
                        },
                    )
                    triples.append(triple)

    else:
        # Pair all correct with all incorrect (no prompt matching)
        # This creates many more triples but might be useful for diversity
        for correct_sample in correct_samples:
            for incorrect_sample in incorrect_samples:
                triple = DPOTriple(
                    prompt=correct_sample.prompt,  # Use correct sample's prompt
                    chosen=format_response(correct_sample, include_reasoning),
                    rejected=format_response(incorrect_sample, include_reasoning),
                    metadata={
                        "chosen_sample_id": correct_sample.sample_id,
                        "rejected_sample_id": incorrect_sample.sample_id,
                        "chosen_category": categorize_sample(correct_sample),
                        "rejected_category": categorize_sample(incorrect_sample),
                        "evaluator_model": correct_sample.evaluator_model,
                        "chosen_generator": correct_sample.generator_model,
                        "rejected_generator": incorrect_sample.generator_model,
                        "dataset": correct_sample.dataset,
                        "experiment": correct_sample.experiment,
                        "prompt_mismatch": correct_sample.prompt != incorrect_sample.prompt,
                    },
                )
                triples.append(triple)

    return triples


def split_triples(
    triples: list[DPOTriple],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: Optional[int] = 42,
) -> tuple[list[DPOTriple], list[DPOTriple]]:
    """Split triples into train and validation sets.

    Args:
        triples: List of DPO triples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        seed: Random seed for reproducibility

    Returns:
        tuple: (train_triples, val_triples)
    """
    import random

    if seed is not None:
        random.seed(seed)

    # Shuffle triples
    shuffled = triples.copy()
    random.shuffle(shuffled)

    # Split
    train_size = int(len(shuffled) * train_ratio)

    train_triples = shuffled[:train_size]
    val_triples = shuffled[train_size:]

    return train_triples, val_triples


def save_triples_jsonl(
    triples: list[DPOTriple],
    output_path: str,
) -> None:
    """Save DPO triples to JSONL file.

    Format compatible with Tinker DPO training:
    {"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}

    Args:
        triples: List of DPO triples
        output_path: Path to output JSONL file
    """
    import json
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for triple in triples:
            record = {
                "prompt": triple.prompt,
                "chosen": triple.chosen,
                "rejected": triple.rejected,
                "metadata": triple.metadata,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {len(triples)} triples to {output_path}")
