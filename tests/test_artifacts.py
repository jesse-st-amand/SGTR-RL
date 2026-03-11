"""Tests for artifact helpers."""

import json

from sgtr_rl.artifacts import append_jsonl, update_run_status


def test_append_jsonl_writes_one_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_jsonl(path, {"step": 1, "train/nll": 0.5})
    assert path.read_text().strip() == '{"step": 1, "train/nll": 0.5}'


def test_update_run_status_preserves_started_at(tmp_path):
    run_dir = tmp_path / "run"
    update_run_status(run_dir, "starting", backend="local", algorithm="sft")
    first = json.loads((run_dir / "status.json").read_text())

    update_run_status(run_dir, "running", backend="local", algorithm="sft", step=3, epoch=1)
    second = json.loads((run_dir / "status.json").read_text())

    assert second["started_at"] == first["started_at"]
    assert second["status"] == "running"
    assert second["step"] == 3
    assert second["epoch"] == 1
