"""Tests for runtime/backend dispatch in pipeline.run_training."""

from unittest.mock import patch

import pytest

from sgtr_rl.config import TrainingConfig
from sgtr_rl.data import randomize_binary_targets, subset_records_by_id
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


def test_run_training_loads_multiple_train_and_val_files(tmp_path):
    train_a = tmp_path / "dataset_a" / "train.jsonl"
    train_b = tmp_path / "dataset_b" / "train.jsonl"
    val_a = tmp_path / "dataset_a" / "val.jsonl"
    val_b = tmp_path / "dataset_b" / "val.jsonl"
    train_a.parent.mkdir()
    train_b.parent.mkdir()

    for path, target, record_id in [
        (train_a, "1", "shared-train"),
        (train_b, "2", "shared-train"),
        (val_a, "1", "shared-val"),
        (val_b, "2", "shared-val"),
    ]:
        _write_jsonl(path, [{"prompt": "q", "target": target, "id": record_id, "format": "ind"}])

    config = TrainingConfig(
        algorithm="sft",
        experiment_name="multi-source-local-test",
        train_file=str(train_a),
        val_file=str(val_a),
        train_files=[str(train_a), str(train_b)],
        val_files=[str(val_a), str(val_b)],
        run_dir=str(tmp_path / "run"),
    )
    runtime = RuntimeConfig(backend="local")

    with (
        patch("sgtr_rl.pipeline.train_local_sft", return_value=9) as mock_local,
        patch("sgtr_rl.pipeline.validate_training_data", return_value=SUMMARY),
    ):
        from sgtr_rl.pipeline import run_training

        run_training(config, runtime)

    [call_config, _, prompts, val_prompts] = mock_local.call_args.args
    assert call_config.experiment_name == "multi-source-local-test"
    assert len(prompts) == 2
    assert len(val_prompts) == 2
    assert len({prompt["id"] for prompt in prompts}) == 2
    assert len({prompt["id"] for prompt in val_prompts}) == 2


def test_run_training_applies_train_transforms_before_dispatch(tmp_path):
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(
        train_path,
        [
            {"prompt": "q1", "target": "1", "id": "id-1", "format": "pw"},
            {"prompt": "q1", "target": "2", "id": "id-1", "format": "pw"},
            {"prompt": "q2", "target": "1", "id": "id-2", "format": "pw"},
            {"prompt": "q2", "target": "2", "id": "id-2", "format": "pw"},
            {"prompt": "q3", "target": "1", "id": "id-3", "format": "pw"},
            {"prompt": "q3", "target": "2", "id": "id-3", "format": "pw"},
        ],
    )
    _write_jsonl(val_path, [{"prompt": "q", "target": "1", "id": "val-1", "format": "ind"}])

    config = TrainingConfig(
        algorithm="sft",
        experiment_name="transform-test",
        train_file=str(train_path),
        val_file=str(val_path),
        run_dir=str(tmp_path / "run"),
        max_train_ids=1,
        subset_seed=7,
        randomize_train_labels=True,
        randomize_train_labels_seed=11,
    )
    runtime = RuntimeConfig(backend="local")

    with (
        patch("sgtr_rl.pipeline.train_local_sft", return_value=3) as mock_local,
        patch("sgtr_rl.pipeline.validate_training_data", return_value=SUMMARY),
        patch(
            "sgtr_rl.pipeline.subset_records_by_id",
            wraps=subset_records_by_id,
        ) as mock_subset,
        patch(
            "sgtr_rl.pipeline.randomize_binary_targets",
            wraps=randomize_binary_targets,
        ) as mock_randomize,
    ):
        from sgtr_rl.pipeline import run_training

        run_training(config, runtime)

    prompts = mock_local.call_args.args[2]
    assert len(prompts) == 2
    assert len({prompt["id"] for prompt in prompts}) == 1
    assert {prompt["target"] for prompt in prompts} == {"1", "2"}
    mock_subset.assert_called_once()
    mock_randomize.assert_called_once()
