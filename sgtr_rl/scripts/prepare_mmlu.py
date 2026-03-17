"""Download MMLU from HuggingFace and save as a single JSONL file.

Subsampling for different benchmark sizes (e.g. 20-sample canary vs
2000-sample benchmark) is handled at runtime by the `num_samples`
field in the benchmark eval config.

Usage:
    python -m scripts.prepare_mmlu
    python -m scripts.prepare_mmlu --out-dir data/benchmarks
"""

import argparse
import json
import logging
from pathlib import Path

from sgtr_rl.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download MMLU and save as JSONL")
    parser.add_argument("--out-dir", type=str, default="data/benchmarks", help="Output directory")
    args = parser.parse_args()

    setup_logging("prepare_mmlu")

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MMLU from cais/mmlu (test split)...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    logger.info("Loaded %d questions", len(ds))

    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    out_path = out_dir / "mmlu.jsonl"
    n_subjects = set()
    with open(out_path, "w") as f:
        for row in ds:
            item = {
                "question": row["question"],
                "subject": row["subject"],
                "choices": row["choices"],
                "answer": letter_map[row["answer"]],
            }
            n_subjects.add(item["subject"])
            f.write(json.dumps(item) + "\n")

    logger.info("Wrote %d questions (%d subjects) to %s", len(ds), len(n_subjects), out_path)


if __name__ == "__main__":
    main()
