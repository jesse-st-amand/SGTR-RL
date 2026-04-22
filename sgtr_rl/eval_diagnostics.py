"""Helpers for fixed-panel evaluation diagnostics."""

from __future__ import annotations

from collections import defaultdict


def select_binary_diagnostic_items(
    items: list[dict],
    *,
    num_examples: int,
    example_ids: list[str] | None = None,
) -> list[dict]:
    """Choose a stable, roughly balanced subset of binary-label items."""
    if num_examples <= 0 or not items:
        return []

    if example_ids:
        selected = []
        for example_id in example_ids:
            for item in items:
                if item.get("id") == example_id:
                    selected.append(item)
                    if len(selected) >= num_examples:
                        return selected
        if selected:
            return selected[:num_examples]

    by_target: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_target[str(item.get("target"))].append(item)

    selected: list[dict] = []
    row_index = 0
    while len(selected) < num_examples:
        added = False
        for target in ("1", "2"):
            group = by_target.get(target, [])
            if row_index < len(group):
                selected.append(group[row_index])
                added = True
                if len(selected) >= num_examples:
                    break
        if not added:
            break
        row_index += 1
    return selected


def build_prompt_preview(item: dict, *, limit: int = 160) -> str:
    """Return a short prompt preview for human-readable diagnostics."""
    prompt = item.get("prompt", "")
    if isinstance(prompt, list):
        parts = []
        for message in prompt:
            role = message.get("role", "?")
            content = str(message.get("content", "")).replace("\n", " ")
            parts.append(f"[{role}] {content}")
        text = " ".join(parts)
    else:
        text = str(prompt).replace("\n", " ")

    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def summarize_binary_margin_rows(rows: list[dict]) -> dict[str, float | int]:
    """Aggregate simple summary stats for binary margin traces."""
    if not rows:
        return {
            "num_examples": 0,
            "mean_margin_1_minus_2": 0.0,
            "predicted_1_count": 0,
            "predicted_2_count": 0,
            "correct_count": 0,
            "accuracy": 0.0,
        }

    predicted_1_count = sum(1 for row in rows if row["predicted_by_margin"] == "1")
    predicted_2_count = sum(1 for row in rows if row["predicted_by_margin"] == "2")
    correct_count = sum(int(row["correct_by_margin"]) for row in rows)
    mean_margin = sum(float(row["margin_1_minus_2"]) for row in rows) / len(rows)

    return {
        "num_examples": len(rows),
        "mean_margin_1_minus_2": mean_margin,
        "predicted_1_count": predicted_1_count,
        "predicted_2_count": predicted_2_count,
        "correct_count": correct_count,
        "accuracy": correct_count / len(rows),
    }
