"""Tests for sgtr_rl.training.train_config."""

import pytest
import yaml

from sgtr_rl.training.train_config import (
    BenchmarkEvalConfig,
    TrainingConfig,
    load_training_config,
)


# ---------------------------------------------------------------------------
# load_training_config
# ---------------------------------------------------------------------------

class TestLoadTrainingConfig:
    def test_load_full_config(self, tmp_path):
        config = {
            "experiment_name": "test_exp",
            "algorithm": "grpo",
            "backend": "tinker",
            "model": {
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            },
            "hyperparameters": {
                "learning_rate": 1e-4,
                "num_epochs": 5,
                "per_device_train_batch_size": 8,
                "seed": 123,
                "num_rollouts_per_prompt": 8,
                "max_completion_length": 512,
            },
            "data": {
                "train_file": "train.jsonl",
                "val_file": "val.jsonl",
            },
            "checkpointing": {
                "save_steps": 100,
                "eval_steps": 100,
            },
            "wandb_project": "test-project",
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert cfg.experiment_name == "test_exp"
        assert cfg.algorithm == "grpo"
        assert cfg.learning_rate == 1e-4
        assert cfg.lora_rank == 16
        assert cfg.num_rollouts_per_prompt == 8
        assert cfg.wandb_project == "test-project"
        assert cfg.save_steps == 100

    def test_load_minimal_config(self, tmp_path):
        """Missing optional sections use defaults."""
        config = {"experiment_name": "minimal"}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert cfg.experiment_name == "minimal"
        assert cfg.algorithm == "grpo"  # default
        assert cfg.learning_rate == 5e-5  # default
        assert cfg.lora_rank == 32  # default
        assert cfg.benchmark_evals == []

    def test_load_with_benchmarks(self, tmp_path):
        config = {
            "experiment_name": "bench_test",
            "benchmark_evals": {
                "mmlu_20": {
                    "type": "mmlu",
                    "data_file": "data/benchmarks/mmlu_20.jsonl",
                    "schedule": "every_epoch",
                    "cot": False,
                },
                "mmlu_500_cot": {
                    "type": "mmlu",
                    "data_file": "data/benchmarks/mmlu_500.jsonl",
                    "schedule": "end_only",
                    "cot": True,
                },
            },
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert len(cfg.benchmark_evals) == 2
        names = {b.name for b in cfg.benchmark_evals}
        assert "mmlu_20" in names
        assert "mmlu_500_cot" in names

        cot_bench = next(b for b in cfg.benchmark_evals if b.name == "mmlu_500_cot")
        assert cot_bench.cot is True
        assert cot_bench.schedule == "end_only"

    def test_load_with_sgtr_benchmarks(self, tmp_path):
        config = {
            "experiment_name": "sgtr_bench_test",
            "benchmark_evals": {
                "cross_ind_val": {
                    "type": "sgtr",
                    "data_file": "data/training_data/sharegpt_ind/val.jsonl",
                    "schedule": "every_epoch",
                    "flip_targets": True,
                    "num_samples": 50,
                },
                "mmlu_canary": {
                    "type": "mmlu",
                    "data_file": "data/benchmarks/mmlu_500.jsonl",
                    "num_samples": 20,
                },
            },
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert len(cfg.benchmark_evals) == 2

        sgtr_bench = next(b for b in cfg.benchmark_evals if b.name == "cross_ind_val")
        assert sgtr_bench.type == "sgtr"
        assert sgtr_bench.flip_targets is True
        assert sgtr_bench.num_samples == 50

        mmlu_bench = next(b for b in cfg.benchmark_evals if b.name == "mmlu_canary")
        assert mmlu_bench.type == "mmlu"
        assert mmlu_bench.flip_targets is False
        assert mmlu_bench.num_samples == 20

    def test_load_flip_targets_training(self, tmp_path):
        config = {
            "experiment_name": "flip_test",
            "data": {
                "train_file": "train.jsonl",
                "val_file": "val.jsonl",
                "flip_targets": True,
            },
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert cfg.flip_targets is True

    def test_flip_targets_defaults_false(self, tmp_path):
        config = {"experiment_name": "no_flip"}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert cfg.flip_targets is False

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_training_config(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# TrainingConfig defaults
# ---------------------------------------------------------------------------

class TestTrainingConfigDefaults:
    def test_defaults_are_sensible(self):
        cfg = TrainingConfig()
        assert cfg.algorithm == "grpo"
        assert cfg.learning_rate == 5e-5
        assert cfg.num_epochs == 3
        assert cfg.seed == 42
        assert cfg.bf16 is True
        assert cfg.benchmark_evals == []
        assert cfg.wandb_project is None
