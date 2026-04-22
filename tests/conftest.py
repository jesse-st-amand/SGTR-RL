"""Shared fixtures for SGTR-RL test suite."""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pw_record(id: str, target: str, **extra) -> dict:
    """Build a minimal PW-format record dict (flat schema)."""
    return {
        "prompt": f"Which response did you write? (id={id})",
        "target": target,
        "id": id,
        "format": "pw",
        **extra,
    }


def _ind_record(id: str, target: str, **extra) -> dict:
    """Build a minimal IND-format record dict (flat schema)."""
    return {
        "prompt": f"Did you write this response? (id={id})",
        "target": target,
        "id": id,
        "format": "ind",
        **extra,
    }


def write_jsonl(path, records: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@pytest.fixture()
def pw_train_val_files(tmp_path):
    """Create valid PW train/val JSONL pair (no ID overlap, 2 records per ID)."""
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
        "model": {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "lora_rank": 32,
        },
        "hyperparameters": {
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "batch_size": 4,
            "seed": 42,
        },
        "data": {
            "train_file": "data/training_data/sharegpt_pw/train.jsonl",
            "val_file": "data/training_data/sharegpt_pw/val.jsonl",
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)
