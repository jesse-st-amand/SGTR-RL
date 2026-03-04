"""Download MMLU from HuggingFace and prepare canary + benchmark JSONL files.

Usage:
    # Default: 20-sample canary + 2000-sample benchmark
    python -m scripts.prepare_mmlu

    # Custom sizes
    python -m scripts.prepare_mmlu --size 3000 --canary-size 20

    # Full dataset (no subsampling)
    python -m scripts.prepare_mmlu --size 0
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path


def stratified_sample(
    by_subject: dict[str, list[dict]], size: int, rng: random.Random,
) -> list[dict]:
    """Stratified proportional sample across subjects."""
    subjects = sorted(by_subject.keys())
    total = sum(len(v) for v in by_subject.values())

    if size <= 0 or size >= total:
        # Return all
        all_items = [item for items in by_subject.values() for item in items]
        rng.shuffle(all_items)
        return all_items

    # Proportional allocation with remainder distribution
    raw_alloc = {s: len(by_subject[s]) / total * size for s in subjects}
    alloc = {s: int(a) for s, a in raw_alloc.items()}
    remainder = size - sum(alloc.values())
    by_frac = sorted(subjects, key=lambda s: raw_alloc[s] - alloc[s], reverse=True)
    for s in by_frac[:remainder]:
        alloc[s] += 1

    sampled = []
    for subj in subjects:
        n = alloc[subj]
        if n == 0:
            continue
        pool = by_subject[subj]
        sampled.extend(rng.sample(pool, min(n, len(pool))))
    rng.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Download MMLU and prepare benchmark JSONL files")
    parser.add_argument(
        "--size", type=int, default=2000,
        help="Benchmark sample size (0 = full dataset)",
    )
    parser.add_argument("--canary-size", type=int, default=20, help="Canary sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default="data/benchmarks", help="Output directory")
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MMLU from cais/mmlu (test split)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"Loaded {len(ds)} questions")

    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    by_subject: dict[str, list[dict]] = {}
    for row in ds:
        item = {
            "question": row["question"],
            "subject": row["subject"],
            "choices": row["choices"],
            "answer": letter_map[row["answer"]],
        }
        by_subject.setdefault(item["subject"], []).append(item)

    subjects = sorted(by_subject.keys())
    total_questions = sum(len(v) for v in by_subject.values())
    print(f"Found {len(subjects)} subjects, {total_questions} questions total")

    rng = random.Random(args.seed)

    # --- Canary subset: 1 question per randomly-chosen subject ---
    canary_size = args.canary_size
    canary_subjects = rng.sample(subjects, min(canary_size, len(subjects)))
    canary = []
    for subj in canary_subjects:
        canary.append(rng.choice(by_subject[subj]))
    rng.shuffle(canary)

    canary_path = out_dir / f"mmlu_{canary_size}.jsonl"
    with open(canary_path, "w") as f:
        for item in canary:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(canary)} canary questions to {canary_path}")

    # --- Benchmark: stratified proportional sample ---
    bench_size = args.size
    benchmark = stratified_sample(by_subject, bench_size, rng)

    actual_size = len(benchmark)
    bench_label = actual_size if bench_size <= 0 else bench_size
    bench_path = out_dir / f"mmlu_{bench_label}.jsonl"
    with open(bench_path, "w") as f:
        for item in benchmark:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {actual_size} benchmark questions to {bench_path}")

    # --- Metadata ---
    subject_counts = Counter(item["subject"] for item in benchmark)
    meta = {
        "seed": args.seed,
        "date": datetime.now().isoformat(),
        "canary_count": len(canary),
        "benchmark_count": actual_size,
        "total_mmlu_questions": total_questions,
        "num_subjects": len(subjects),
        "benchmark_subject_counts": dict(sorted(subject_counts.items())),
    }
    meta_path = out_dir / "mmlu_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
