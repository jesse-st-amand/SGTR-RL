"""Tests for runtime/backend dispatch in pipeline.run_training."""

from unittest.mock import patch

import pytest

from sgtr_rl.config import TrainingConfig
from sgtr_rl.runtime_config import RuntimeConfig

SUMMARY = {
    "train_records": 1,
    "val_records": 1,
    "train_ids": 1,
    "val_ids": 1,
    "format": "ind",
}


def _write_jsonl(path, records):
    import json

    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_run_training_dispatches_local_backend(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(train_path, [{"prompt": "q", "target": "1", "id": "train-1"}])
    _write_jsonl(val_path, [{"prompt": "q", "target": "2", "id": "val-1"}])

    config = TrainingConfig(
        algorithm="sft",
        experiment_name="local-test",
        train_file=str(train_path),
        val_file=str(val_path),
        run_dir=str(tmp_path / "run"),
    )
    runtime = RuntimeConfig(backend="local")

    with (
        patch("sgtr_rl.pipeline.train_local_sft", return_value=7) as mock_local,
        patch(
            "sgtr_rl.pipeline.validate_training_data",
            return_value=SUMMARY,
        ),
    ):
        from sgtr_rl.pipeline import run_training

        run_training(config, runtime)

    mock_local.assert_called_once()


def test_run_training_rejects_local_grpo(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(train_path, [{"prompt": "q", "target": "1", "id": "train-1"}])
    _write_jsonl(val_path, [{"prompt": "q", "target": "2", "id": "val-1"}])

    config = TrainingConfig(
        algorithm="grpo",
        experiment_name="local-test",
        train_file=str(train_path),
        val_file=str(val_path),
        run_dir=str(tmp_path / "run"),
    )

    with patch(
        "sgtr_rl.pipeline.validate_training_data",
        return_value=SUMMARY,
    ):
        from sgtr_rl.pipeline import run_training

        with pytest.raises(NotImplementedError):
            run_training(config, RuntimeConfig(backend="local"))
