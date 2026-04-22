"""Tests for sgtr_rl.metrics."""

import json

from sgtr_rl.metrics import (
    make_eval_artifact_name,
    save_split_diagnostics,
    save_split_predictions,
    save_val_diagnostics,
    save_val_predictions,
)


def test_make_eval_artifact_name_uses_epoch_for_epoch_trigger():
    assert make_eval_artifact_name(epoch=3, step=12, eval_trigger="epoch") == "epoch_3"


def test_make_eval_artifact_name_uses_step_for_step_trigger():
    assert make_eval_artifact_name(epoch=3, step=12, eval_trigger="step") == "step_12"


def test_save_val_predictions_uses_step_filename_for_step_trigger(tmp_path):
    run_dir = tmp_path / "run"
    save_val_predictions(
        {
            "accuracy": 0.5,
            "predictions": [{"id": "a", "prediction": "1", "target": "2", "correct": False}],
        },
        str(run_dir),
        epoch=3,
        step=12,
        eval_trigger="step",
    )

    pred_path = run_dir / "val_predictions" / "step_12.json"
    assert pred_path.exists()
    with open(pred_path) as f:
        payload = json.load(f)

    assert payload["epoch"] == 3
    assert payload["step"] == 12
    assert payload["eval_trigger"] == "step"


def test_save_val_diagnostics_uses_step_filename_for_step_trigger(tmp_path):
    run_dir = tmp_path / "run"
    save_val_diagnostics(
        {
            "epoch": 3,
            "step": 12,
            "eval_trigger": "step",
            "summary": {"accuracy": 0.5},
            "examples": [{"id": "a", "margin_1_minus_2": 0.2}],
        },
        str(run_dir),
        epoch=3,
        step=12,
        eval_trigger="step",
    )

    diag_path = run_dir / "val_diagnostics" / "step_12.json"
    assert diag_path.exists()
    with open(diag_path) as f:
        payload = json.load(f)

    assert payload["epoch"] == 3
    assert payload["step"] == 12
    assert payload["eval_trigger"] == "step"
    assert payload["summary"]["accuracy"] == 0.5


def test_save_split_predictions_uses_named_directory(tmp_path):
    run_dir = tmp_path / "run"
    save_split_predictions(
        {
            "accuracy": 1.0,
            "predictions": [{"id": "a", "prediction": "1", "target": "1", "correct": True}],
        },
        str(run_dir),
        split_name="train_panel",
        epoch=2,
        step=8,
        eval_trigger="step",
    )

    pred_path = run_dir / "train_panel_predictions" / "step_8.json"
    assert pred_path.exists()
    with open(pred_path) as f:
        payload = json.load(f)

    assert payload["accuracy"] == 1.0
    assert payload["predictions"][0]["id"] == "a"


def test_save_split_diagnostics_uses_named_directory(tmp_path):
    run_dir = tmp_path / "run"
    save_split_diagnostics(
        {
            "epoch": 2,
            "step": 8,
            "eval_trigger": "step",
            "summary": {"accuracy": 1.0},
            "examples": [{"id": "a", "margin_1_minus_2": 0.7}],
        },
        str(run_dir),
        split_name="train_panel",
        epoch=2,
        step=8,
        eval_trigger="step",
    )

    diag_path = run_dir / "train_panel_diagnostics" / "step_8.json"
    assert diag_path.exists()
    with open(diag_path) as f:
        payload = json.load(f)

    assert payload["summary"]["accuracy"] == 1.0
    assert payload["examples"][0]["id"] == "a"
