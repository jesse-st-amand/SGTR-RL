"""High-level loop structure tests for SFT and GRPO trainers.

Mocks at a higher level (run_val_eval, run_benchmark_evals, generate_summary_plot)
to test loop structure independently of eval implementation.
"""

from unittest.mock import patch, MagicMock, call

from tests.integration.conftest import patch_tinker_modules


# ---------------------------------------------------------------------------
# SFT high-level tests
# ---------------------------------------------------------------------------


class TestSFTEvalSchedule:
    """SFT evaluation schedule tests."""

    def test_sft_eval_schedule(self, sft_config):
        """run_val_eval called at epoch 0, 1, ..., n_epochs (n+1 times total)."""
        with patch_tinker_modules():
            with patch("sgtr_rl.training.sft_trainer.run_val_eval") as mock_val, \
                 patch("sgtr_rl.training.sft_trainer.run_benchmark_evals"), \
                 patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                n_epochs = sft_config.num_epochs
                assert mock_val.call_count == n_epochs + 1

                # Verify epoch values: 0, 1, 2
                epochs = [c.kwargs["epoch"] for c in mock_val.call_args_list]
                assert epochs == list(range(n_epochs + 1))

    def test_sft_benchmark_schedule(self, sft_config):
        """run_benchmark_evals called at epoch 0, 1, ..., n_epochs with correct total_epochs."""
        with patch_tinker_modules():
            with patch("sgtr_rl.training.sft_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.sft_trainer.run_benchmark_evals") as mock_bench, \
                 patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                n_epochs = sft_config.num_epochs
                assert mock_bench.call_count == n_epochs + 1

                for c in mock_bench.call_args_list:
                    assert c.kwargs["total_epochs"] == n_epochs


class TestSFTTrainingSteps:
    """SFT training step count tests."""

    def test_sft_total_training_steps(self, sft_config):
        """forward_backward called exactly n_batches * n_epochs times."""
        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.training.sft_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.sft_trainer.run_benchmark_evals"), \
                 patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                n_epochs = sft_config.num_epochs
                n_prompts = 4  # tiny_train_val_files has 4 train records
                batch_size = sft_config.per_device_train_batch_size
                n_batches = n_prompts // batch_size
                expected = n_batches * n_epochs

                assert mocks["training_client"].forward_backward.call_count == expected


class TestSFTDataValidation:
    """Data validation tests."""

    def test_sft_data_validation_called(self, sft_config):
        """validate_training_data called when val_file exists."""
        with patch_tinker_modules():
            with patch("sgtr_rl.training.sft_trainer.validate_training_data") as mock_validate, \
                 patch("sgtr_rl.training.sft_trainer.generate_summary_plot"):
                from sgtr_rl.training.sft_trainer import TinkerSFTTrainer

                trainer = TinkerSFTTrainer(sft_config)
                trainer.train()

                mock_validate.assert_called_once_with(
                    sft_config.train_file, sft_config.val_file
                )


# ---------------------------------------------------------------------------
# GRPO high-level tests
# ---------------------------------------------------------------------------


class TestGRPOEvalSchedule:
    """GRPO evaluation schedule tests."""

    def test_grpo_eval_schedule(self, grpo_config):
        """run_val_eval called at epoch 0, 1, ..., n_epochs."""
        with patch_tinker_modules(num_sequences=2):
            with patch("sgtr_rl.training.grpo_trainer.run_val_eval") as mock_val, \
                 patch("sgtr_rl.training.grpo_trainer.run_benchmark_evals"):
                from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

                trainer = TinkerRLTrainer(grpo_config)
                trainer.train()

                n_epochs = grpo_config.num_epochs
                assert mock_val.call_count == n_epochs + 1

                epochs = [c.kwargs["epoch"] for c in mock_val.call_args_list]
                assert epochs == list(range(n_epochs + 1))

    def test_grpo_benchmark_schedule(self, grpo_config):
        """run_benchmark_evals called at epoch 0, 1, ..., n_epochs."""
        with patch_tinker_modules(num_sequences=2):
            with patch("sgtr_rl.training.grpo_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.grpo_trainer.run_benchmark_evals") as mock_bench:
                from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

                trainer = TinkerRLTrainer(grpo_config)
                trainer.train()

                n_epochs = grpo_config.num_epochs
                assert mock_bench.call_count == n_epochs + 1

                for c in mock_bench.call_args_list:
                    assert c.kwargs["total_epochs"] == n_epochs


class TestGRPOTrainingSteps:
    """GRPO training step count tests."""

    def test_grpo_total_training_steps(self, grpo_config):
        """forward_backward called at most n_batches * n_epochs times.

        Can be less due to skipped zero-signal groups.
        """
        with patch_tinker_modules(num_sequences=2) as mocks:
            from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

            trainer = TinkerRLTrainer(grpo_config)
            trainer.train()

            n_epochs = grpo_config.num_epochs
            n_prompts = 4
            batch_size = grpo_config.per_device_train_batch_size
            n_batches = n_prompts // batch_size
            max_steps = n_batches * n_epochs

            assert mocks["training_client"].forward_backward.call_count <= max_steps


class TestGRPOSamplingOrder:
    """GRPO sampling/training order tests."""

    def test_grpo_sampling_before_training(self, grpo_config):
        """save_weights_and_get_sampling_client called before forward_backward in each batch."""
        call_order = []

        with patch_tinker_modules(num_sequences=2) as mocks:
            orig_save = mocks["training_client"].save_weights_and_get_sampling_client
            orig_fwd = mocks["training_client"].forward_backward

            def track_save(*args, **kwargs):
                call_order.append("save_weights")
                return orig_save(*args, **kwargs)

            def track_fwd(*args, **kwargs):
                call_order.append("forward_backward")
                return orig_fwd(*args, **kwargs)

            mocks["training_client"].save_weights_and_get_sampling_client = MagicMock(
                side_effect=track_save
            )
            mocks["training_client"].forward_backward = MagicMock(
                side_effect=track_fwd
            )

            from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

            trainer = TinkerRLTrainer(grpo_config)
            trainer.train()

            # Every forward_backward should be preceded by a save_weights
            fwd_indices = [i for i, x in enumerate(call_order) if x == "forward_backward"]
            save_indices = [i for i, x in enumerate(call_order) if x == "save_weights"]

            for fwd_idx in fwd_indices:
                # There should be a save_weights call before this forward_backward
                preceding_saves = [s for s in save_indices if s < fwd_idx]
                assert len(preceding_saves) > 0
