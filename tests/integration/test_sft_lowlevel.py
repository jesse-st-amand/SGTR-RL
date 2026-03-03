"""Low-level integration tests for TinkerSFTTrainer.

Uses the tinker_mocks fixture to inject fake tinker modules, then runs
the real training loop and asserts on pipeline math and orchestration.
"""

import math
from unittest.mock import patch

import numpy as np

from tests.integration.conftest import patch_tinker_modules


class TestSFTSmoke:
    """Basic wiring and import tests."""

    def test_smoke_runs_to_completion(self, sft_config):
        with patch_tinker_modules():
            with patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()


class TestSFTBaseline:
    """Epoch 0 baseline evaluation."""

    def test_epoch_0_baseline_runs(self, sft_config):
        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.training.sft_trainer.run_val_eval") as mock_val_eval, \
                 patch("sgtr_rl.training.sft_trainer.run_benchmark_evals") as mock_bench, \
                 patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                # First call should be epoch=0 (baseline)
                val_calls = mock_val_eval.call_args_list
                assert val_calls[0].kwargs["epoch"] == 0 or val_calls[0][1].get("epoch") == 0 or val_calls[0][0][-1] == 0

                bench_calls = mock_bench.call_args_list
                first_bench = bench_calls[0]
                # epoch is a keyword arg
                assert first_bench.kwargs.get("epoch", None) == 0


class TestSFTForwardBackward:
    """Forward-backward and loss function tests."""

    def test_forward_backward_uses_cross_entropy(self, sft_config):
        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                # Every forward_backward call should use cross_entropy
                for call in mocks["training_client"].forward_backward.call_args_list:
                    assert call[1].get("loss_fn") == "cross_entropy" or call[0][1] == "cross_entropy"


class TestSFTAccuracy:
    """Train accuracy computation tests."""

    def test_train_accuracy_correct_when_logprob_above_threshold(self, sft_config):
        """logprob > log(0.5) → accuracy = 1.0."""
        logprob_value = -0.1  # > log(0.5) ≈ -0.693
        with patch_tinker_modules(logprob_value=logprob_value) as mocks:
            with patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                # Check logged accuracy metrics — all should be 1.0
                for call in mocks["ml_logger"].log_metrics.call_args_list:
                    metrics = call[0][0]
                    if "train/accuracy" in metrics:
                        assert metrics["train/accuracy"] == 1.0

    def test_train_accuracy_zero_when_logprob_below_threshold(self, sft_config):
        """logprob < log(0.5) → accuracy = 0.0."""
        logprob_value = -2.0  # < log(0.5) ≈ -0.693
        with patch_tinker_modules(logprob_value=logprob_value) as mocks:
            with patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                for call in mocks["ml_logger"].log_metrics.call_args_list:
                    metrics = call[0][0]
                    if "train/accuracy" in metrics:
                        assert metrics["train/accuracy"] == 0.0


class TestSFTEvalSchedule:
    """Validation and eval schedule tests."""

    def test_val_eval_at_each_epoch(self, sft_config):
        """run_val_eval called n_epochs+1 times (epoch 0 + each boundary)."""
        with patch_tinker_modules():
            with patch("sgtr_rl.training.sft_trainer.run_val_eval") as mock_val_eval, \
                 patch("sgtr_rl.training.sft_trainer.run_benchmark_evals"), \
                 patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                n_epochs = sft_config.num_epochs
                assert mock_val_eval.call_count == n_epochs + 1


class TestSFTFlipTargets:
    """flip_targets configuration test."""

    def test_flip_targets_applied(self, tmp_path, tiny_train_val_files):
        from sgtr_rl.training.train_config import TrainingConfig

        train_file, val_file = tiny_train_val_files
        config = TrainingConfig(
            algorithm="sft",
            backend="tinker",
            experiment_name="test_sft_flip",
            model_name="test-model",
            num_epochs=1,
            per_device_train_batch_size=2,
            train_file=train_file,
            val_file=val_file,
            run_dir=str(tmp_path / "run"),
            flip_targets=True,
        )

        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(config)
                # Call _load_prompts directly to check targets
                prompts = trainer._load_prompts()
                # Original targets alternate 1, 2, 1, 2 → flipped to 2, 1, 2, 1
                targets = [p["target"] for p in prompts]
                assert targets == ["2", "1", "2", "1"]


class TestSFTMetrics:
    """Metrics logging tests."""

    def test_metrics_logged_at_correct_steps(self, sft_config):
        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                # Collect step numbers from log_metrics calls with train/ keys
                steps = []
                for call in mocks["ml_logger"].log_metrics.call_args_list:
                    metrics = call[0][0]
                    if "train/nll" in metrics:
                        step = call[1].get("step") or call[0][1]
                        steps.append(step)

                # Steps should be 1, 2, 3, 4 (2 batches x 2 epochs)
                assert steps == [1, 2, 3, 4]

    def test_ml_logger_closed(self, sft_config):
        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                mocks["ml_logger"].close.assert_called_once()


class TestSFTCheckpoint:
    """Checkpoint saving tests."""

    def test_checkpoint_saved_at_end(self, sft_config):
        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                mocks["checkpoint_utils"].save_checkpoint.assert_called_once()
                call_kwargs = mocks["checkpoint_utils"].save_checkpoint.call_args[1]
                assert call_kwargs["name"] == "final"


class TestSFTBatching:
    """Batch size and drop-remainder tests."""

    def test_batches_drop_remainder(self, tmp_path):
        """6 prompts, batch_size=4 → exactly 1 batch per epoch (2 dropped)."""
        from sgtr_rl.training.train_config import TrainingConfig
        from tests.integration.conftest import _pw_record, _write_jsonl

        # 6 train records (3 UUIDs x 2, valid PW, not divisible by batch_size=4)
        train_records = [
            _pw_record("uuid-1", "1"),
            _pw_record("uuid-1", "2"),
            _pw_record("uuid-2", "1"),
            _pw_record("uuid-2", "2"),
            _pw_record("uuid-3", "1"),
            _pw_record("uuid-3", "2"),
        ]
        val_records = [
            _pw_record("val-uuid-1", "1"),
            _pw_record("val-uuid-1", "2"),
        ]
        train_path = tmp_path / "train6.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, train_records)
        _write_jsonl(val_path, val_records)

        config = TrainingConfig(
            algorithm="sft",
            backend="tinker",
            experiment_name="test_sft_batch",
            model_name="test-model",
            num_epochs=1,
            per_device_train_batch_size=4,
            train_file=str(train_path),
            val_file=str(val_path),
            run_dir=str(tmp_path / "run"),
        )

        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.training.sft_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.sft_trainer.run_benchmark_evals"), \
                 patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(config)
                trainer.train()

                # 6 // 4 = 1 batch per epoch, 1 epoch → 1 forward_backward call
                assert mocks["training_client"].forward_backward.call_count == 1
