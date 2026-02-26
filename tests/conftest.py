"""Shared fixtures for SGTR-RL test suite."""

import json

import pytest
import yaml


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _pw_record(uuid: str, target: str, **meta) -> dict:
    """Build a minimal PW-format record dict."""
    metadata = {"uuid": uuid, "format": "pw", **meta}
    return {
        "prompt": f"Which response did you write? (uuid={uuid})",
        "target": target,
        "metadata": metadata,
    }


def _ind_record(uuid: str, target: str, **meta) -> dict:
    """Build a minimal IND-format record dict."""
    metadata = {"uuid": uuid, "format": "ind", **meta}
    return {
        "prompt": f"Did you write this response? (uuid={uuid})",
        "target": target,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# JSONL helper
# ---------------------------------------------------------------------------

def write_jsonl(path, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pw_train_val_files(tmp_path):
    """Create valid PW train/val JSONL pair (no UUID overlap, 2 records per UUID)."""
    train_records = [
        _pw_record("train-uuid-1", "1"),
        _pw_record("train-uuid-1", "2"),
        _pw_record("train-uuid-2", "1"),
        _pw_record("train-uuid-2", "2"),
    ]
    val_records = [
        _pw_record("val-uuid-1", "1"),
        _pw_record("val-uuid-1", "2"),
        _pw_record("val-uuid-2", "1"),
        _pw_record("val-uuid-2", "2"),
    ]
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    return str(train_path), str(val_path)


@pytest.fixture()
def sample_config_yaml(tmp_path):
    """Create a minimal experiment YAML config file."""
    config = {
        "experiment_name": "14_sft_pw_test",
        "algorithm": "sft",
        "backend": "tinker",
        "model": {
            "name": "Qwen/Qwen2-1.5B",
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
        },
        "hyperparameters": {
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "per_device_train_batch_size": 4,
            "seed": 42,
        },
        "data": {
            "train_file": "data/training_data/sharegpt_pw/train.jsonl",
            "val_file": "data/training_data/sharegpt_pw/val.jsonl",
        },
        "checkpointing": {
            "save_steps": 50,
            "eval_steps": 50,
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)
