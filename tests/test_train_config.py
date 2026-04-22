"""Tests for sgtr_rl.config."""

import pytest
import yaml
from pydantic import ValidationError

from sgtr_rl.config import (
    TrainingConfig,
    load_training_config,
)


class TestLoadTrainingConfig:
    def test_load_full_config(self, tmp_path):
        config = {
            "experiment_name": "test_exp",
            "algorithm": "grpo",
            "model": {
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "lora_rank": 16,
            },
            "hyperparameters": {
                "learning_rate": 1e-4,
                "num_epochs": 5,
                "batch_size": 8,
                "max_steps": 50,
                "seed": 123,
                "num_rollouts_per_prompt": 8,
                "max_completion_length": 512,
            },
            "data": {
                "train_file": "train.jsonl",
                "val_file": "val.jsonl",
                "max_train_ids": 10,
                "subset_seed": 7,
                "randomize_train_labels": True,
                "randomize_train_labels_seed": 9,
            },
            "evaluation": {
                "trigger": "step",
                "frequency": 5,
                "diagnostic_num_examples": 6,
                "diagnostic_example_ids": ["a", "b"],
                "train_diagnostic_num_examples": 4,
                "train_diagnostic_example_ids": ["c", "d"],
            },
            "resume": {
                "state_path": "tinker://run/checkpoint",
                "completed_epochs": 2,
                "global_step": 10,
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
        assert cfg.max_steps == 50
        assert cfg.num_rollouts_per_prompt == 8
        assert cfg.max_train_ids == 10
        assert cfg.subset_seed == 7
        assert cfg.randomize_train_labels is True
        assert cfg.randomize_train_labels_seed == 9
        assert cfg.eval_trigger == "step"
        assert cfg.eval_frequency == 5
        assert cfg.eval_diagnostic_num_examples == 6
        assert cfg.eval_diagnostic_example_ids == ["a", "b"]
        assert cfg.train_diagnostic_num_examples == 4
        assert cfg.train_diagnostic_example_ids == ["c", "d"]
        assert cfg.resume_state_path == "tinker://run/checkpoint"
        assert cfg.resume_completed_epochs == 2
        assert cfg.resume_global_step == 10
        assert cfg.wandb_project == "test-project"

    def test_load_minimal_config(self, tmp_path):
        """Missing optional sections use defaults."""
        config = {"experiment_name": "minimal"}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert cfg.experiment_name == "minimal"
        assert cfg.algorithm == "sft"
        assert cfg.learning_rate == 5e-5
        assert cfg.lora_rank == 32
        assert cfg.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.batch_size == 16
        assert cfg.num_epochs == 20
        assert cfg.max_steps is None
        assert cfg.eval_trigger == "epoch"
        assert cfg.eval_frequency == 1
        assert cfg.eval_diagnostic_num_examples == 0
        assert cfg.eval_diagnostic_example_ids == []
        assert cfg.train_diagnostic_num_examples == 0
        assert cfg.train_diagnostic_example_ids == []
        assert cfg.resume_state_path is None
        assert cfg.resume_completed_epochs == 0
        assert cfg.resume_global_step is None
        assert cfg.benchmark_evals == []

    def test_load_with_benchmarks(self, tmp_path):
        config = {
            "experiment_name": "bench_test",
            "benchmark_evals": {
                "mmlu_20": {
                    "type": "mmlu",
                    "data_file": "data/benchmarks/mmlu.jsonl",
                    "schedule": "every_epoch",
                    "cot": False,
                },
                "mmlu_500_cot": {
                    "type": "mmlu",
                    "data_file": "data/benchmarks/mmlu.jsonl",
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
                    "num_samples": 50,
                },
                "mmlu_canary": {
                    "type": "mmlu",
                    "data_file": "data/benchmarks/mmlu.jsonl",
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
        assert sgtr_bench.num_samples == 50

        mmlu_bench = next(b for b in cfg.benchmark_evals if b.name == "mmlu_canary")
        assert mmlu_bench.type == "mmlu"
        assert mmlu_bench.num_samples == 20

    def test_load_with_system_prompt(self, tmp_path):
        config = {
            "experiment_name": "sp_test",
            "data": {
                "train_file": "train.jsonl",
                "val_file": "val.jsonl",
                "use_system_prompt": True,
            },
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert cfg.use_system_prompt is True

    def test_load_with_multiple_train_and_val_files(self, tmp_path):
        config = {
            "experiment_name": "multi-source",
            "data": {
                "train_files": ["train_a.jsonl", "train_b.jsonl"],
                "val_files": ["val_a.jsonl", "val_b.jsonl"],
                "train_mix_strategy": "per_id_one_source",
                "max_train_ids": 12,
            },
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        cfg = load_training_config(str(path))
        assert cfg.train_files == ["train_a.jsonl", "train_b.jsonl"]
        assert cfg.val_files == ["val_a.jsonl", "val_b.jsonl"]
        assert cfg.train_file == "train_a.jsonl"
        assert cfg.val_file == "val_a.jsonl"
        assert cfg.train_mix_strategy == "per_id_one_source"
        assert cfg.max_train_ids == 12

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_training_config(str(tmp_path / "nonexistent.yaml"))


class TestTrainingConfigDefaults:
    def test_defaults_are_sensible(self):
        cfg = TrainingConfig()
        assert cfg.algorithm == "sft"
        assert cfg.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert cfg.learning_rate == 5e-5
        assert cfg.num_epochs == 20
        assert cfg.batch_size == 16
        assert cfg.max_steps is None
        assert cfg.seed == 42
        assert cfg.eval_trigger == "epoch"
        assert cfg.eval_frequency == 1
        assert cfg.eval_diagnostic_num_examples == 0
        assert cfg.eval_diagnostic_example_ids == []
        assert cfg.train_diagnostic_num_examples == 0
        assert cfg.train_diagnostic_example_ids == []
        assert cfg.resume_state_path is None
        assert cfg.resume_completed_epochs == 0
        assert cfg.resume_global_step is None
        assert cfg.benchmark_evals == []
        assert cfg.wandb_project is None
        assert cfg.use_system_prompt is False


class TestConfigValidation:
    def test_unknown_top_level_key_raises(self, tmp_path):
        config = {"experiment_name": "test", "unkown_key": True}
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Unknown top-level"):
            load_training_config(str(path))

    def test_typo_in_hyperparameters_raises(self, tmp_path):
        config = {
            "experiment_name": "test",
            "hyperparameters": {"lerning_rate": 1e-4},
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValidationError):
            load_training_config(str(path))

    def test_typo_in_model_section_raises(self, tmp_path):
        config = {
            "experiment_name": "test",
            "model": {"naem": "test-model"},
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValidationError):
            load_training_config(str(path))

    def test_typo_in_data_section_raises(self, tmp_path):
        config = {
            "experiment_name": "test",
            "data": {"trian_file": "train.jsonl"},
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValidationError):
            load_training_config(str(path))

    def test_wrong_type_raises(self, tmp_path):
        config = {
            "experiment_name": "test",
            "hyperparameters": {"learning_rate": "not_a_number"},
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValidationError):
            load_training_config(str(path))

    def test_unknown_benchmark_field_raises(self, tmp_path):
        config = {
            "experiment_name": "test",
            "benchmark_evals": {
                "test_bench": {
                    "type": "mmlu",
                    "data_file": "test.jsonl",
                    "unknwon_field": True,
                },
            },
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValidationError):
            load_training_config(str(path))

    def test_both_single_and_plural_data_fields_raise(self, tmp_path):
        config = {
            "experiment_name": "test",
            "data": {
                "train_file": "train.jsonl",
                "train_files": ["train_a.jsonl"],
            },
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="data.train_file or data.train_files"):
            load_training_config(str(path))
