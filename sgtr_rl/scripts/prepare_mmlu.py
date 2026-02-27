"""Download MMLU from HuggingFace and prepare 20-sample + 500-sample JSONL files."""

import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path


def main():
    from datasets import load_dataset

    seed = 42
    small_size = 20
    large_size = 500
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MMLU from cais/mmlu (test split)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"Loaded {len(ds)} questions")

    # Map numeric answer index to letter
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    # Group by subject
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
    print(f"Found {len(subjects)} subjects")

    rng = random.Random(seed)

    # --- 20-sample subset: 1 question per randomly-chosen subject ---
    small_subjects = rng.sample(subjects, min(small_size, len(subjects)))
    small = []
    for subj in small_subjects:
        small.append(rng.choice(by_subject[subj]))
    rng.shuffle(small)

    small_path = out_dir / "mmlu_20.jsonl"
    with open(small_path, "w") as f:
        for item in small:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(small)} questions to {small_path}")

    # --- Benchmark: stratified proportional sample ---
    total_questions = sum(len(v) for v in by_subject.values())
    # Compute per-subject allocation (at least 1 if subject sampled)
    raw_alloc = {s: len(by_subject[s]) / total_questions * large_size for s in subjects}
    # Round down, then distribute remainders
    alloc = {s: int(a) for s, a in raw_alloc.items()}
    remainder = large_size - sum(alloc.values())
    # Sort by fractional part descending to allocate remainders
    by_frac = sorted(subjects, key=lambda s: raw_alloc[s] - alloc[s], reverse=True)
    for s in by_frac[:remainder]:
        alloc[s] += 1

    large = []
    for subj in subjects:
        n = alloc[subj]
        if n == 0:
            continue
        pool = by_subject[subj]
        sampled = rng.sample(pool, min(n, len(pool)))
        large.extend(sampled)
    rng.shuffle(large)

    large_path = out_dir / "mmlu_500.jsonl"
    with open(large_path, "w") as f:
        for item in large:
            f.write(json.dumps(item) + "\n")
    print(f"Wrote {len(large)} questions to {large_path}")

    # --- Metadata ---
    subject_counts = Counter(item["subject"] for item in large)
    meta = {
        "seed": seed,
        "date": datetime.now().isoformat(),
        "small_count": len(small),
        "large_count": len(large),
        "total_mmlu_questions": total_questions,
        "num_subjects": len(subjects),
        "large_subject_counts": dict(sorted(subject_counts.items())),
    }
    meta_path = out_dir / "mmlu_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
