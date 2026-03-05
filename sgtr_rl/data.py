"""Data loading and validation for SGTR-RL training."""

import json
from collections import defaultdict


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts. Skips blank lines."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records



def validate_training_data(
    train_records: list[dict],
    val_records: list[dict],
) -> dict:
    """Validate train/val records for integrity.

    Args:
        train_records: List of training record dicts.
        val_records: List of validation record dicts.

    Returns:
        Summary dict with counts, ID counts, and target distribution.

    Raises:
        ValueError: On any integrity failure.
    """
    # Schema validation
    for label, records in [("train", train_records), ("val", val_records)]:
        for i, rec in enumerate(records):
            for field in ("prompt", "target", "id"):
                if field not in rec:
                    raise ValueError(
                        f"{label}[{i}]: missing required field '{field}'"
                    )

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

    # ID overlap check
    train_ids = {rec["id"] for rec in train_records}
    val_ids = {rec["id"] for rec in val_records}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            f"ID leak: {len(overlap)} IDs appear in both train and val. "
            f"Examples: {list(overlap)[:3]}"
        )

    # PW format: check each ID has exactly 2 records (both orderings)
    fmt = _detect_format(train_records)
    if fmt == "pw":
        for label, records in [("train", train_records), ("val", val_records)]:
            id_counts = defaultdict(int)
            for rec in records:
                id_counts[rec["id"]] += 1
            bad_ids = {k: c for k, c in id_counts.items() if c != 2}
            if bad_ids:
                examples = list(bad_ids.items())[:3]
                raise ValueError(
                    f"{label}: {len(bad_ids)} IDs don't have exactly 2 records "
                    f"(both orderings required for PW format). "
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
        "train_ids": len(train_ids),
        "val_ids": len(val_ids),
        "train_target_dist": dict(train_targets),
        "val_target_dist": dict(val_targets),
        "format": fmt,
    }


def build_conversation(
    item: dict,
    use_system_prompt: bool = False,
) -> list[dict]:
    """Build a conversation message list from a training record.

    Args:
        item: Training record with 'prompt' and optionally 'system_prompt'.
              prompt can be a string (UT/ICML) or list of message dicts (AT/COLM).
        use_system_prompt: If True, prepend system_prompt from the record.

    Returns:
        List of message dicts (role/content) for the renderer.
    """
    prompt = item["prompt"]

    if isinstance(prompt, list):
        convo = list(prompt)
    else:
        convo = [{"role": "user", "content": prompt}]

    if use_system_prompt:
        sp = item.get("system_prompt")
        if sp:
            convo.insert(0, {"role": "system", "content": sp})

    return convo


def _detect_format(records: list[dict]) -> str:
    """Detect format from records, defaulting to 'ind'."""
    for rec in records:
        fmt = rec.get("format")
        if fmt:
            return fmt
    return "ind"
