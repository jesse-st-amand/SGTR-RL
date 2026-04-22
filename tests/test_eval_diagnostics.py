"""Tests for evaluation diagnostic helpers."""

from sgtr_rl.eval_diagnostics import (
    build_prompt_preview,
    select_binary_diagnostic_items,
    summarize_binary_margin_rows,
)


def test_select_binary_diagnostic_items_balances_targets():
    items = [
        {"id": "a", "target": "1", "prompt": "prompt a"},
        {"id": "b", "target": "1", "prompt": "prompt b"},
        {"id": "c", "target": "2", "prompt": "prompt c"},
        {"id": "d", "target": "2", "prompt": "prompt d"},
    ]

    selected = select_binary_diagnostic_items(items, num_examples=4)

    assert [item["id"] for item in selected] == ["a", "c", "b", "d"]


def test_select_binary_diagnostic_items_prefers_explicit_ids():
    items = [
        {"id": "a", "target": "1", "prompt": "prompt a"},
        {"id": "b", "target": "2", "prompt": "prompt b"},
        {"id": "c", "target": "1", "prompt": "prompt c"},
    ]

    selected = select_binary_diagnostic_items(
        items,
        num_examples=2,
        example_ids=["c", "b"],
    )

    assert [item["id"] for item in selected] == ["c", "b"]


def test_build_prompt_preview_handles_multi_turn():
    item = {
        "prompt": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }

    preview = build_prompt_preview(item)

    assert preview == "[user] hello [assistant] world"


def test_summarize_binary_margin_rows_reports_accuracy():
    rows = [
        {"predicted_by_margin": "1", "correct_by_margin": True, "margin_1_minus_2": 0.5},
        {"predicted_by_margin": "2", "correct_by_margin": False, "margin_1_minus_2": -0.25},
    ]

    summary = summarize_binary_margin_rows(rows)

    assert summary["num_examples"] == 2
    assert summary["predicted_1_count"] == 1
    assert summary["predicted_2_count"] == 1
    assert summary["correct_count"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["mean_margin_1_minus_2"] == 0.125
