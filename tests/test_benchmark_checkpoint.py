"""Tests for scripts.benchmark_checkpoint."""

import json

import pytest
import yaml

from scripts.benchmark_checkpoint import (
    _load_extra_benchmark_configs,
    _read_tinker_sampler_path,
    _resolve_checkpoint_run,
    _select_benchmarks,
)
from sgtr_rl.config import load_training_config
from sgtr_rl.runtime_config import RuntimeConfig


def _write_base_run_config(path, *, train_file="train.jsonl", val_file="val.jsonl"):
    with open(path, "w") as f:
        yaml.safe_dump(
            {
                "experiment_name": "test-exp",
                "algorithm": "sft",
                "model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "lora_rank": 16},
                "hyperparameters": {},
                "data": {"train_file": train_file, "val_file": val_file},
                "benchmark_evals": {
                    "xeval_dataset_pku": {
                        "type": "sgtr",
                        "data_file": "data/a.jsonl",
                        "schedule": "every_epoch",
                    },
                    "mmlu_20": {
                        "type": "mmlu",
                        "data_file": "data/b.jsonl",
                        "schedule": "every_epoch",
                        "cot": True,
                        "num_samples": 20,
                    },
                },
            },
            f,
        )


def test_read_tinker_sampler_path_reads_latest_entry(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    with open(checkpoint_dir / "checkpoints.jsonl", "w") as f:
        f.write(json.dumps({"sampler_path": "tinker://old/sampler_weights/1"}) + "\n")
        f.write(json.dumps({"sampler_path": "tinker://new/sampler_weights/final"}) + "\n")

    assert _read_tinker_sampler_path(run_dir) == "tinker://new/sampler_weights/final"


def test_resolve_checkpoint_run_tinker(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    _write_base_run_config(run_dir / "config.yaml")
    with open(run_dir / "status.json", "w") as f:
        json.dump({"backend": "tinker", "epoch": 20, "step": 200}, f)
    with open(run_dir / "checkpoints" / "checkpoints.jsonl", "w") as f:
        f.write(json.dumps({"sampler_path": "tinker://abc/sampler_weights/final"}) + "\n")

    resolved = _resolve_checkpoint_run(str(run_dir), runtime_override=None)

    assert resolved.backend == "tinker"
    assert resolved.sampler_path == "tinker://abc/sampler_weights/final"
    assert resolved.epoch == 20
    assert resolved.step == 200


def test_resolve_checkpoint_run_local_uses_manifest_runtime(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints" / "final"
    checkpoint_dir.mkdir(parents=True)
    _write_base_run_config(run_dir / "config.yaml")
    with open(checkpoint_dir / "checkpoint_manifest.json", "w") as f:
        json.dump(
            {
                "backend": "local",
                "epoch": 3,
                "global_step": 17,
                "runtime": RuntimeConfig(backend="local").model_dump(mode="json"),
            },
            f,
        )

    resolved = _resolve_checkpoint_run(str(run_dir), runtime_override=None)

    assert resolved.backend == "local"
    assert resolved.checkpoint_dir == checkpoint_dir
    assert resolved.runtime is not None
    assert resolved.epoch == 3
    assert resolved.step == 17


def test_select_benchmarks_filters_and_validates(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_base_run_config(config_path)
    config = load_training_config(config_path)

    selected = _select_benchmarks(config, ["mmlu_20"])
    assert [cfg.name for cfg in selected] == ["mmlu_20"]

    with pytest.raises(ValueError, match="Unknown benchmark names"):
        _select_benchmarks(config, ["missing"])


def test_load_extra_benchmark_configs_and_select(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_base_run_config(config_path)
    config = load_training_config(config_path)

    extra_path = tmp_path / "extra.yaml"
    with open(extra_path, "w") as f:
        yaml.safe_dump(
            {
                "xeval_vs_opus": {
                    "type": "sgtr",
                    "data_file": "data/opus.jsonl",
                    "schedule": "every_epoch",
                }
            },
            f,
        )

    extra = _load_extra_benchmark_configs(str(extra_path))
    selected = _select_benchmarks(config, ["xeval_vs_opus"], extra_configs=extra)

    assert [cfg.name for cfg in extra] == ["xeval_vs_opus"]
    assert [cfg.name for cfg in selected] == ["xeval_vs_opus"]


def test_load_extra_benchmark_configs_uses_defaults_for_posthoc(tmp_path):
    extra_path = tmp_path / "extra.yaml"
    with open(extra_path, "w") as f:
        yaml.safe_dump(
            {
                "xeval_vs_haiku": {
                    "type": "sgtr",
                    "data_file": "data/haiku.jsonl",
                }
            },
            f,
        )

    [cfg] = _load_extra_benchmark_configs(str(extra_path))

    assert cfg.name == "xeval_vs_haiku"
    assert cfg.schedule == "every_epoch"
    assert cfg.frequency == 1
