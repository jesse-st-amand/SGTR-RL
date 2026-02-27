"""Prepare SGTR training data from static repo data.

Usage:
    python -m sgtr_rl.scripts.prepare_data \
        --evaluator_model ll-3.1-8b \
        --generator_models ll-3.1-8b qwen-2.5-7b \
        --dataset wikisum \
        --subset test_set_1-30 \
        --experiment_config experiments/ICML_02_UT_IND-Q_Rec_NPr_FA_Inst/config.yaml \
        --output data/training_data/ind_wikisum/ \
        [--format ind|pw] \
        [--train_ratio 0.8]
"""

import argparse
import random
from pathlib import Path

from dotenv import load_dotenv

from sgtr_rl.data_processing.prompt_builder import build_sgtr_prompts, save_prompt_dataset


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Prepare SGTR training data")
    parser.add_argument(
        "--evaluator_model", required=True, help="Short model name for the evaluator"
    )
    parser.add_argument(
        "--generator_models", nargs="+", required=True, help="Short model names for generators"
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. wikisum)")
    parser.add_argument("--subset", required=True, help="Data subset (e.g. test_set_1-30)")
    parser.add_argument(
        "--experiment_config", required=True, help="Path to SGTR experiment config YAML"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for train.jsonl / val.jsonl"
    )
    parser.add_argument(
        "--format", default="ind", choices=["ind", "pw"], help="Prompt format (default: ind)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Fraction of data for training (default: 0.8)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    args = parser.parse_args()

    print(f"Building {args.format.upper()} prompts: {args.evaluator_model} vs {args.generator_models}")
    prompts = build_sgtr_prompts(
        evaluator_model=args.evaluator_model,
        generator_models=args.generator_models,
        experiment_config_path=args.experiment_config,
        dataset_name=args.dataset,
        data_subset=args.subset,
        format=args.format,
    )
    print(f"Built {len(prompts)} total prompts")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(prompts)
    split_idx = int(len(prompts) * args.train_ratio)
    train_prompts = prompts[:split_idx]
    val_prompts = prompts[split_idx:]

    output_dir = Path(args.output)
    save_prompt_dataset(train_prompts, output_dir / "train.jsonl")
    save_prompt_dataset(val_prompts, output_dir / "val.jsonl")

    print(f"Train: {len(train_prompts)}, Val: {len(val_prompts)}")


if __name__ == "__main__":
    main()
