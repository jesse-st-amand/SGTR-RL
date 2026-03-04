"""Tests for sgtr_rl.runs."""

from unittest.mock import patch

import pytest
import yaml

from sgtr_rl.config import TrainingConfig
from sgtr_rl.runs import (
    _find_existing_run,
    compute_overrides,
    create_run_dir,
    make_run_name,
)

# ---------------------------------------------------------------------------
# make_run_name
# ---------------------------------------------------------------------------

class TestMakeRunName:
    def test_make_run_name_with_overrides(self):
        name = make_run_name("14_sft_pw", "lr=1e-4", "20250101_120000")
        assert name == "14_sft_pw__lr=1e-4__20250101_120000"

    def test_make_run_name_no_overrides(self):
        name = make_run_name("14_sft_pw", "", "20250101_120000")
        assert name == "14_sft_pw__20250101_120000"


# ---------------------------------------------------------------------------
# compute_overrides
# ---------------------------------------------------------------------------

class TestComputeOverrides:
    def test_compute_overrides_detects_changes(self, tmp_path):
        yaml_config = {
            "hyperparameters": {"learning_rate": 5e-5, "seed": 42},
            "model": {"lora_rank": 32},
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f)

        config = TrainingConfig(learning_rate=1e-4, seed=42, lora_rank=32)
        result = compute_overrides(config, str(yaml_path))
        assert "lr=0.0001" in result

    def test_compute_overrides_no_changes(self, tmp_path):
        yaml_config = {
            "hyperparameters": {"learning_rate": 5e-5, "seed": 42},
            "model": {"lora_rank": 32},
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_config, f)

        config = TrainingConfig(learning_rate=5e-5, seed=42, lora_rank=32)
        result = compute_overrides(config, str(yaml_path))
        assert result == ""

    def test_compute_overrides_missing_yaml(self, tmp_path):
        config = TrainingConfig()
        result = compute_overrides(config, str(tmp_path / "nonexistent.yaml"))
        assert result == ""


# ---------------------------------------------------------------------------
# create_run_dir
# ---------------------------------------------------------------------------

class TestCreateRunDir:
    def test_create_run_dir_structure(self, tmp_path, sample_config_yaml):
        config = TrainingConfig(
            experiment_name="test_exp",
            train_file=str(tmp_path / "train.jsonl"),
        )
        # Create a dummy train file so the parent dir exists
        (tmp_path / "train.jsonl").write_text("")

        with patch("sgtr_rl.runs.BASE_DIR", tmp_path / "results"):
            run_dir = create_run_dir(config, sample_config_yaml, exists="new")

        assert run_dir.exists()
        assert (run_dir / "checkpoints").exists()
        assert (run_dir / "config.yaml").exists()

    def test_create_run_dir_no_eval_or_tensorboard(self, tmp_path, sample_config_yaml):
        config = TrainingConfig(
            experiment_name="test_exp",
            train_file=str(tmp_path / "train.jsonl"),
        )
        (tmp_path / "train.jsonl").write_text("")

        with patch("sgtr_rl.runs.BASE_DIR", tmp_path / "results"):
            run_dir = create_run_dir(config, sample_config_yaml, exists="new")

        assert not (run_dir / "eval").exists()
        assert not (run_dir / "tensorboard").exists()

    def test_create_run_dir_exists_error(self, tmp_path, sample_config_yaml):
        config = TrainingConfig(
            experiment_name="test_exp",
            train_file=str(tmp_path / "train.jsonl"),
        )
        (tmp_path / "train.jsonl").write_text("")

        with patch("sgtr_rl.runs.BASE_DIR", tmp_path / "results"):
            create_run_dir(config, sample_config_yaml, exists="new")
            with pytest.raises(FileExistsError):
                create_run_dir(config, sample_config_yaml, exists="error")

    def test_create_run_dir_exists_skip(self, tmp_path, sample_config_yaml):
        config = TrainingConfig(
            experiment_name="test_exp",
            train_file=str(tmp_path / "train.jsonl"),
        )
        (tmp_path / "train.jsonl").write_text("")

        with patch("sgtr_rl.runs.BASE_DIR", tmp_path / "results"):
            first_dir = create_run_dir(config, sample_config_yaml, exists="new")
            second_dir = create_run_dir(config, sample_config_yaml, exists="skip")
            assert second_dir == first_dir


# ---------------------------------------------------------------------------
# _find_existing_run
# ---------------------------------------------------------------------------

class TestFindExistingRun:
    def test_find_existing_run(self, tmp_path):
        run_dir = tmp_path / "test_exp__20250101_120000"
        run_dir.mkdir()
        result = _find_existing_run(tmp_path, "test_exp")
        assert result == run_dir

    def test_find_existing_run_not_found(self, tmp_path):
        result = _find_existing_run(tmp_path, "nonexistent")
        assert result is None
