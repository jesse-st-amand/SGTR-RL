"""Extract training data from .eval files.

Usage:
    python -m sgtr_rl.scripts.extract_from_eval \
        --eval_dir /tmp/eval_data \
        --output data/training_data/sharegpt_ind/ \
        --format ind \
        [--train_ratio 0.8] \
        [--seed 42]

    python -m sgtr_rl.scripts.extract_from_eval \
        --eval_dir /tmp/eval_data \
        --output data/training_data/sharegpt_pw/ \
        --format pw

The .eval files are zip archives produced by inspect-ai. Each contains
sample JSON files with the full SGTR prompt already constructed, plus the
target answer. This script extracts those directly into training JSONL,
bypassing the need for raw generation data (input.json / data.json).
"""

import argparse
import json
import random
import zipfile
from pathlib import Path


def extract_samples_from_eval(eval_path: Path) -> list[dict]:
    """Extract all samples from a single .eval zip archive."""
    samples = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            with zf.open(name) as f:
                sample = json.loads(f.read())
                samples.append(sample)
    return samples


def eval_sample_to_training(sample: dict) -> dict:
    """Convert an eval sample to a training record."""
    prompt = sample["input"]
    target = str(sample.get("target", sample["metadata"].get("correct_answer")))
    metadata = {
        "uuid": sample["metadata"].get("uuid", ""),
        "dataset_name": sample["metadata"].get("dataset_name", ""),
        "data_subset": sample["metadata"].get("data_subset", ""),
        "source_eval": True,
    }

    # IND-specific metadata
    if "treatment_name" in sample["metadata"]:
        metadata["treatment_name"] = sample["metadata"]["treatment_name"]
        metadata["is_control"] = sample["metadata"].get("is_control", False)
        metadata["format"] = "ind"
    # PW-specific metadata
    elif "treatment_name_1" in sample["metadata"]:
        metadata["treatment_name_1"] = sample["metadata"]["treatment_name_1"]
        metadata["treatment_name_2"] = sample["metadata"]["treatment_name_2"]
        metadata["format"] = "pw"

    return {"prompt": prompt, "target": target, "metadata": metadata}


def collect_eval_files(eval_dir: Path, fmt: str) -> list[Path]:
    """Find all .eval files matching the requested format."""
    subdir = "IND" if fmt == "ind" else "PW"
    files = []
    for subset_dir in sorted(eval_dir.iterdir()):
        if not subset_dir.is_dir() or subset_dir.name.startswith("."):
            continue
        fmt_dir = subset_dir / subdir
        if fmt_dir.exists():
            files.extend(sorted(fmt_dir.glob("*.eval")))
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract training data from .eval files")
    parser.add_argument("--eval_dir", required=True, help="Directory containing eval data")
    parser.add_argument("--output", required=True, help="Output directory for train/val JSONL")
    parser.add_argument(
        "--format", default="ind", choices=["ind", "pw"], help="Prompt format (default: ind)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Fraction for training (default: 0.8)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    eval_files = collect_eval_files(eval_dir, args.format)

    if not eval_files:
        print(f"No .eval files found for format '{args.format}' in {eval_dir}")
        return

    print(f"Found {len(eval_files)} .eval files:")
    for f in eval_files:
        print(f"  {f.relative_to(eval_dir)}")

    # Extract all samples
    all_records = []
    for eval_file in eval_files:
        samples = extract_samples_from_eval(eval_file)
        records = [eval_sample_to_training(s) for s in samples]
        print(f"  Extracted {len(records)} samples from {eval_file.name}")
        all_records.extend(records)

    # Deduplicate by (prompt, target) — eval files may have repeated epochs
    seen = set()
    unique_records = []
    for rec in all_records:
        key = (rec["prompt"], rec["target"])
        if key not in seen:
            seen.add(key)
            unique_records.append(rec)

    print(f"\nTotal: {len(all_records)} raw, {len(unique_records)} after dedup")

    # Check balance
    targets = [r["target"] for r in unique_records]
    for t in sorted(set(targets)):
        print(f"  target={t}: {targets.count(t)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(unique_records)
    split_idx = int(len(unique_records) * args.train_ratio)
    train = unique_records[:split_idx]
    val = unique_records[split_idx:]

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subset, name in [(train, "train.jsonl"), (val, "val.jsonl")]:
        path = output_dir / name
        with open(path, "w") as f:
            for rec in subset:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(subset)} records to {path}")


if __name__ == "__main__":
    main()
