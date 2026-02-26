"""Validate training data integrity.

Checks schema, target values, UUID isolation between train/val splits,
and (for PW format) that each UUID has both response orderings.
"""

import json
from collections import defaultdict
from pathlib import Path


def validate_training_data(train_path: str, val_path: str) -> dict:
    """Validate train/val JSONL files for integrity.

    Args:
        train_path: Path to training JSONL file.
        val_path: Path to validation JSONL file.

    Returns:
        Summary dict with counts, UUID counts, and target distribution.

    Raises:
        ValueError: On any integrity failure.
    """
    train_records = _load_jsonl(train_path)
    val_records = _load_jsonl(val_path)

    # Schema validation
    for label, records in [("train", train_records), ("val", val_records)]:
        for i, rec in enumerate(records):
            _validate_record_schema(rec, label, i)

    # Target value validation
    for label, records in [("train", train_records), ("val", val_records)]:
        bad = [
            i for i, rec in enumerate(records)
            if rec["target"] not in ("1", "2")
        ]
        if bad:
            raise ValueError(
                f"{label}: {len(bad)} records have invalid target values "
                f"(expected '1' or '2'). First bad index: {bad[0]}, "
                f"value: {records[bad[0]]['target']!r}"
            )

    # UUID overlap check
    train_uuids = {rec["metadata"]["uuid"] for rec in train_records}
    val_uuids = {rec["metadata"]["uuid"] for rec in val_records}
    overlap = train_uuids & val_uuids
    if overlap:
        raise ValueError(
            f"UUID leak: {len(overlap)} UUIDs appear in both train and val. "
            f"Examples: {list(overlap)[:3]}"
        )

    # PW format: check each (uuid, treatment_pair) has exactly 2 records (both orderings)
    fmt = _detect_format(train_records)
    if fmt == "pw":
        for label, records in [("train", train_records), ("val", val_records)]:
            pair_counts = defaultdict(int)
            for rec in records:
                meta = rec["metadata"]
                uuid = meta["uuid"]
                # Canonical pair key: sorted treatment names
                t1 = meta.get("treatment_name_1", "")
                t2 = meta.get("treatment_name_2", "")
                pair_key = (uuid, tuple(sorted([t1, t2])))
                pair_counts[pair_key] += 1
            bad_pairs = {k: c for k, c in pair_counts.items() if c != 2}
            if bad_pairs:
                examples = list(bad_pairs.items())[:3]
                raise ValueError(
                    f"{label}: {len(bad_pairs)} (uuid, treatment_pair) groups don't have "
                    f"exactly 2 records (both orderings required for PW format). "
                    f"Examples: {examples}"
                )

    # Build summary
    train_targets = defaultdict(int)
    for rec in train_records:
        train_targets[rec["target"]] += 1
    val_targets = defaultdict(int)
    for rec in val_records:
        val_targets[rec["target"]] += 1

    return {
        "train_records": len(train_records),
        "val_records": len(val_records),
        "train_uuids": len(train_uuids),
        "val_uuids": len(val_uuids),
        "train_target_dist": dict(train_targets),
        "val_target_dist": dict(val_targets),
        "format": fmt,
    }


def _load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _validate_record_schema(rec: dict, label: str, index: int) -> None:
    """Validate a single record has required fields."""
    for field in ("prompt", "target", "metadata"):
        if field not in rec:
            raise ValueError(f"{label}[{index}]: missing required field '{field}'")
    if not isinstance(rec["metadata"], dict):
        raise ValueError(f"{label}[{index}]: 'metadata' must be a dict")
    if "uuid" not in rec["metadata"]:
        raise ValueError(f"{label}[{index}]: metadata missing 'uuid'")


def _detect_format(records: list[dict]) -> str:
    """Detect format from metadata, defaulting to 'ind'."""
    for rec in records:
        fmt = rec.get("metadata", {}).get("format")
        if fmt:
            return fmt
    return "ind"
