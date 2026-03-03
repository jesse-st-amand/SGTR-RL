"""Shared utilities for SGTR-RL training pipeline."""

import json


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts. Skips blank lines."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def flip_target(target: str) -> str:
    """Swap "1"<->"2" for label-flipping experiments."""
    return {"1": "2", "2": "1"}.get(target, target)
