"""High-level loop structure tests for SFT and GRPO training functions.

Mocks at a higher level (run_val_eval, run_benchmark_evals, generate_summary_plot)
to test loop structure independently of eval implementation.
"""

from unittest.mock import MagicMock, patch

from tests.integration.conftest import _build_ctx, patch_tinker_modules

# ---------------------------------------------------------------------------
# SFT high-level tests
# ---------------------------------------------------------------------------


class TestSFTEvalSchedule:
    """SFT evaluation schedule tests."""

    def test_sft_eval_schedule(self, sft_config, tiny_prompts):
        """run_val_eval called at each epoch boundary (n_epochs times)."""
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval") as mock_val, \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)

                n_epochs = sft_config.num_epochs
                assert mock_val.call_count == n_epochs

                # Verify epoch values: 1, 2
                epochs = [c.kwargs["epoch"] for c in mock_val.call_args_list]
                assert epochs == list(range(1, n_epochs + 1))

    def test_sft_benchmark_schedule(self, sft_config, tiny_prompts):
        """run_benchmark_evals called at each epoch boundary with correct total_epochs."""
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals") as mock_bench:
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)

                n_epochs = sft_config.num_epochs
                assert mock_bench.call_count == n_epochs

                for c in mock_bench.call_args_list:
                    assert c.kwargs["total_epochs"] == n_epochs


class TestSFTTrainingSteps:
    """SFT training step count tests."""

    def test_sft_total_training_steps(self, sft_config, tiny_prompts):
        """forward_backward called exactly n_batches * n_epochs times."""
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)

                n_epochs = sft_config.num_epochs
                n_prompts = 4  # tiny_train_val_files has 4 train records
                batch_size = sft_config.per_device_train_batch_size
                n_batches = n_prompts // batch_size
                expected = n_batches * n_epochs

                assert mocks["training_client"].forward_backward.call_count == expected


class TestSFTDataValidation:
    """Data validation tests (now in pipeline.py, not in train_sft)."""

    def test_pipeline_calls_validation(self, sft_config, tiny_prompts):
        """validate_training_data is called by run_training (pipeline)."""
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            with patch("sgtr_rl.pipeline.validate_training_data") as mock_validate, \
                 patch("sgtr_rl.pipeline.setup_tinker", return_value=_build_ctx(mocks)), \
                 patch("sgtr_rl.pipeline.run_val_eval"), \
                 patch("sgtr_rl.pipeline.run_benchmark_evals"), \
                 patch("sgtr_rl.pipeline.save_checkpoint"), \
                 patch("sgtr_rl.pipeline.generate_summary_plot"), \
                 patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.pipeline import run_training

                run_training(sft_config)

                mock_validate.assert_called_once_with(
                    sft_config.train_file, sft_config.val_file,
                    id_field=sft_config.id_field,
                )


# ---------------------------------------------------------------------------
# GRPO high-level tests
# ---------------------------------------------------------------------------


class TestGRPOEvalSchedule:
    """GRPO evaluation schedule tests."""

    def test_grpo_eval_schedule(self, grpo_config, tiny_prompts):
        """run_val_eval called at each epoch boundary (n_epochs times)."""
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules(num_sequences=2) as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.grpo.run_val_eval") as mock_val, \
                 patch("sgtr_rl.grpo.run_benchmark_evals"):
                from sgtr_rl.grpo import train_grpo

                train_grpo(grpo_config, ctx, prompts, val_prompts)

                n_epochs = grpo_config.num_epochs
                assert mock_val.call_count == n_epochs

                epochs = [c.kwargs["epoch"] for c in mock_val.call_args_list]
                assert epochs == list(range(1, n_epochs + 1))

    def test_grpo_benchmark_schedule(self, grpo_config, tiny_prompts):
        """run_benchmark_evals called at each epoch boundary."""
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules(num_sequences=2) as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.grpo.run_val_eval"), \
                 patch("sgtr_rl.grpo.run_benchmark_evals") as mock_bench:
                from sgtr_rl.grpo import train_grpo

                train_grpo(grpo_config, ctx, prompts, val_prompts)

                n_epochs = grpo_config.num_epochs
                assert mock_bench.call_count == n_epochs

                for c in mock_bench.call_args_list:
                    assert c.kwargs["total_epochs"] == n_epochs


class TestGRPOTrainingSteps:
    """GRPO training step count tests."""

    def test_grpo_total_training_steps(self, grpo_config, tiny_prompts):
        """forward_backward called at most n_batches * n_epochs times.

        Can be less due to skipped zero-signal groups.
        """
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules(num_sequences=2) as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.grpo.run_val_eval"), \
                 patch("sgtr_rl.grpo.run_benchmark_evals"):
                from sgtr_rl.grpo import train_grpo

                train_grpo(grpo_config, ctx, prompts, val_prompts)

                n_epochs = grpo_config.num_epochs
                n_prompts = 4
                batch_size = grpo_config.per_device_train_batch_size
                n_batches = n_prompts // batch_size
                max_steps = n_batches * n_epochs

                assert mocks["training_client"].forward_backward.call_count <= max_steps


class TestGRPOSamplingOrder:
    """GRPO sampling/training order tests."""

    def test_grpo_sampling_before_training(self, grpo_config, tiny_prompts):
        """save_weights_and_get_sampling_client called before forward_backward in each batch."""
        prompts, val_prompts = tiny_prompts
        call_order = []

        with patch_tinker_modules(num_sequences=2) as mocks:
            ctx = _build_ctx(mocks)
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
            # Update ctx to use patched training_client
            ctx.training_client = mocks["training_client"]

            with patch("sgtr_rl.grpo.run_val_eval"), \
                 patch("sgtr_rl.grpo.run_benchmark_evals"):
                from sgtr_rl.grpo import train_grpo

                train_grpo(grpo_config, ctx, prompts, val_prompts)

            # Every forward_backward should be preceded by a save_weights
            fwd_indices = [i for i, x in enumerate(call_order) if x == "forward_backward"]
            save_indices = [i for i, x in enumerate(call_order) if x == "save_weights"]

            for fwd_idx in fwd_indices:
                preceding_saves = [s for s in save_indices if s < fwd_idx]
                assert len(preceding_saves) > 0
