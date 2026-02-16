"""Test script for data loading and triple generation.

This script tests loading Qwen3-80B evaluation data and creating DPO triples.

Usage:
    uv run python -m sgtr_rl.scripts.test_data_loading
"""

from pathlib import Path
from sgtr_rl.config import get_dataset_path
from sgtr_rl.data_processing import load_experiment_evals, create_dpo_triples, categorize_sample


def main():
    """Test data loading pipeline."""
    print("=" * 70)
    print("SGTR-RL Data Loading Test")
    print("=" * 70)
    print()

    # Test with WikiSum dataset, Qwen3-80B evaluations
    dataset = "wikisum"
    experiment = "ICML_04_UT_IND-Q_Rec_NPr_FA_Rsn"
    evaluator = "qwen-3.0-80b-thinking"

    print(f"Dataset: {dataset}")
    print(f"Experiment: {experiment}")
    print(f"Evaluator: {evaluator}")
    print()

    # Get experiment directory
    dataset_path = get_dataset_path(dataset)
    experiment_dir = dataset_path / "training_set_1-20" / experiment

    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return 1

    print(f"✅ Found experiment directory: {experiment_dir}")
    print()

    # Load eval files
    print(f"Loading evaluations from {evaluator}...")
    samples = load_experiment_evals(
        experiment_dir,
        evaluator_model=evaluator,
        dataset=dataset,
    )

    print(f"✅ Loaded {len(samples)} samples")
    print()

    # Categorize samples
    categories = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for sample in samples:
        category = categorize_sample(sample)
        categories[category] += 1

    print("Sample Categories:")
    print(f"  TP (True Positive):  {categories['TP']:3d}")
    print(f"  TN (True Negative):  {categories['TN']:3d}")
    print(f"  FP (False Positive): {categories['FP']:3d}")
    print(f"  FN (False Negative): {categories['FN']:3d}")
    print(f"  Correct: {categories['TP'] + categories['TN']}")
    print(f"  Incorrect: {categories['FP'] + categories['FN']}")
    accuracy = (categories["TP"] + categories["TN"]) / len(samples) if samples else 0
    print(f"  Accuracy: {accuracy:.2%}")
    print()

    # Show sample
    if samples:
        print("=" * 70)
        print("Sample Example:")
        print("=" * 70)
        sample = samples[0]
        print(f"ID: {sample.sample_id}")
        print(f"Category: {categorize_sample(sample)}")
        print(f"Evaluator: {sample.evaluator_model}")
        print(f"Generator: {sample.generator_model}")
        print()
        print("Prompt:")
        print(sample.prompt[:200] + "..." if len(sample.prompt) > 200 else sample.prompt)
        print()
        print("Reasoning:")
        print(
            sample.reasoning[:300] + "..."
            if len(sample.reasoning) > 300
            else sample.reasoning
        )
        print()
        print(f"Answer: {sample.answer}")
        print(f"Target: {sample.target}")
        print(f"Correct: {sample.is_correct}")
        print()

    # Create DPO triples
    print("=" * 70)
    print("Creating DPO Triples")
    print("=" * 70)
    print()

    triples = create_dpo_triples(samples, include_reasoning=True, require_same_prompt=True)

    print(f"✅ Created {len(triples)} DPO triples")
    print()

    # Show triple example
    if triples:
        print("=" * 70)
        print("Triple Example:")
        print("=" * 70)
        triple = triples[0]
        print("Prompt:")
        print(triple.prompt[:200] + "..." if len(triple.prompt) > 200 else triple.prompt)
        print()
        print("Chosen (Correct):")
        print(triple.chosen[:300] + "..." if len(triple.chosen) > 300 else triple.chosen)
        print()
        print("Rejected (Incorrect):")
        print(
            triple.rejected[:300] + "..." if len(triple.rejected) > 300 else triple.rejected
        )
        print()
        print("Metadata:")
        for key, value in triple.metadata.items():
            print(f"  {key}: {value}")
        print()

    print("=" * 70)
    print("✅ Data loading test completed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
