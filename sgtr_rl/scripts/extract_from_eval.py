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
from collections import defaultdict
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


_NO_COT_SUFFIX = "Provide only the number and no additional text."
DEFAULT_COT_SUFFIX = (
    "Think step by step about whether this text matches your writing style, "
    "then give your final answer as a single number (1 or 2) on its own line."
)


def _to_cot_prompt(prompt: str, cot_suffix: str = DEFAULT_COT_SUFFIX) -> str:
    """Replace the no-CoT instruction with a CoT instruction."""
    if _NO_COT_SUFFIX in prompt:
        return prompt.replace(_NO_COT_SUFFIX, cot_suffix)
    return prompt


def eval_sample_to_training(
    sample: dict, cot: bool = False, cot_suffix: str = DEFAULT_COT_SUFFIX,
) -> dict:
    """Convert an eval sample to a training record."""
    prompt = sample["input"]
    if cot:
        prompt = _to_cot_prompt(prompt, cot_suffix)
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
    parser.add_argument(
        "--cot", action="store_true",
        help="Replace no-CoT prompt suffix with CoT instruction",
    )
    parser.add_argument(
        "--cot_suffix", default=None,
        help="Custom CoT instruction suffix (overrides default). Implies --cot.",
    )
    args = parser.parse_args()

    if args.cot_suffix is not None:
        args.cot = True

    eval_dir = Path(args.eval_dir)
    eval_files = collect_eval_files(eval_dir, args.format)

    if not eval_files:
        print(f"No .eval files found for format '{args.format}' in {eval_dir}")
        return

    if args.cot:
        cot_suffix = args.cot_suffix or DEFAULT_COT_SUFFIX
        print(f"CoT enabled. Suffix: {cot_suffix!r}")
    else:
        print("CoT disabled (direct answer prompts)")

    print(f"Found {len(eval_files)} .eval files:")
    for f in eval_files:
        print(f"  {f.relative_to(eval_dir)}")

    # Extract all samples
    all_records = []
    for eval_file in eval_files:
        samples = extract_samples_from_eval(eval_file)
        cot_suffix = args.cot_suffix or DEFAULT_COT_SUFFIX
        records = [eval_sample_to_training(s, cot=args.cot, cot_suffix=cot_suffix) for s in samples]
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

    # Group records by UUID for split
    uuid_to_records = defaultdict(list)
    for rec in unique_records:
        uuid = rec["metadata"].get("uuid", "")
        uuid_to_records[uuid].append(rec)

    # For PW format, verify every UUID has exactly 2 records (both orderings)
    if args.format == "pw":
        bad_uuids = {u: len(recs) for u, recs in uuid_to_records.items() if len(recs) != 2}
        if bad_uuids:
            for u, count in list(bad_uuids.items())[:5]:
                print(f"  ERROR: UUID {u[:12]}... has {count} records (expected 2)")
            raise ValueError(
                f"{len(bad_uuids)} UUIDs don't have exactly 2 records. "
                f"PW format requires both response orderings per UUID."
            )
        print(f"Verified: all {len(uuid_to_records)} UUIDs have exactly 2 records")

    # Shuffle UUIDs (not individual records) and split
    random.seed(args.seed)
    uuids = list(uuid_to_records.keys())
    random.shuffle(uuids)
    split_idx = int(len(uuids) * args.train_ratio)
    train_uuids = set(uuids[:split_idx])
    val_uuids = set(uuids[split_idx:])

    # Verify no UUID overlap
    overlap = train_uuids & val_uuids
    assert not overlap, f"UUID overlap between train/val: {overlap}"

    # Flatten back to records
    train = [rec for u in uuids[:split_idx] for rec in uuid_to_records[u]]
    val = [rec for u in uuids[split_idx:] for rec in uuid_to_records[u]]

    print(f"Split by UUID: {len(train_uuids)} train UUIDs ({len(train)} records), "
          f"{len(val_uuids)} val UUIDs ({len(val)} records)")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for subset, name in [(train, "train.jsonl"), (val, "val.jsonl")]:
        path = output_dir / name
        with open(path, "w") as f:
            for rec in subset:
                f.write(json.dumps(rec) + "\n")
        print(f"Saved {len(subset)} records to {path}")

    # Save extraction metadata
    cot_suffix = args.cot_suffix or DEFAULT_COT_SUFFIX
    meta = {
        "eval_dir": str(eval_dir),
        "format": args.format,
        "cot": args.cot,
        "cot_suffix": cot_suffix if args.cot else None,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "split_by": "uuid",
        "total_raw": len(all_records),
        "total_dedup": len(unique_records),
        "train_uuids": len(train_uuids),
        "val_uuids": len(val_uuids),
        "train_size": len(train),
        "val_size": len(val),
        "eval_files": [str(f) for f in eval_files],
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved extraction metadata to {output_dir / 'extraction_meta.json'}")


if __name__ == "__main__":
    main()
