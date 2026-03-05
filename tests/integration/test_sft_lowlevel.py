"""Low-level integration tests for train_sft().

Uses the tinker_mocks fixture to inject fake tinker modules, then runs
the real training loop and asserts on pipeline math and orchestration.
"""

from unittest.mock import patch

from tests.integration.conftest import _build_ctx, _pw_record, _write_jsonl, patch_tinker_modules


class TestSFTSmoke:
    """Basic wiring and import tests."""

    def test_smoke_runs_to_completion(self, sft_config, tiny_prompts):
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)


class TestSFTBaseline:
    """Epoch 0 baseline evaluation (now handled by pipeline.py)."""

    def test_pipeline_runs_baseline(self, sft_config, tiny_prompts):
        """pipeline.run_training calls run_val_eval at epoch=0 before training."""
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.pipeline.setup_tinker", return_value=ctx), \
                 patch("sgtr_rl.pipeline.validate_training_data"), \
                 patch("sgtr_rl.pipeline.run_val_eval") as mock_pipeline_val, \
                 patch("sgtr_rl.pipeline.run_benchmark_evals") as mock_pipeline_bench, \
                 patch("sgtr_rl.pipeline.save_checkpoint"), \
                 patch("sgtr_rl.pipeline.generate_summary_plot"), \
                 patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.pipeline import run_training

                run_training(sft_config)

                # Pipeline should call baseline eval at epoch=0
                assert mock_pipeline_val.call_count == 1
                assert mock_pipeline_val.call_args.kwargs["epoch"] == 0
                assert mock_pipeline_bench.call_count == 1
                assert mock_pipeline_bench.call_args.kwargs["epoch"] == 0


class TestSFTForwardBackward:
    """Forward-backward and loss function tests."""

    def test_forward_backward_uses_cross_entropy(self, sft_config, tiny_prompts):
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)

                # Every forward_backward call should use cross_entropy
                fwd_bwd = mocks["training_client"].forward_backward
                for call in fwd_bwd.call_args_list:
                    assert call[1].get("loss_fn") == "cross_entropy"


class TestSFTAccuracy:
    """Train accuracy computation tests."""

    def test_train_accuracy_correct_when_logprob_above_threshold(self, sft_config, tiny_prompts):
        """logprob > log(0.5) → accuracy = 1.0."""
        prompts, val_prompts = tiny_prompts
        logprob_value = -0.1  # > log(0.5) ≈ -0.693
        with patch_tinker_modules(logprob_value=logprob_value) as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)

                # Check logged accuracy metrics — all should be 1.0
                for call in mocks["ml_logger"].log_metrics.call_args_list:
                    metrics = call[0][0]
                    if "train/accuracy" in metrics:
                        assert metrics["train/accuracy"] == 1.0

    def test_train_accuracy_zero_when_logprob_below_threshold(self, sft_config, tiny_prompts):
        """logprob < log(0.5) → accuracy = 0.0."""
        prompts, val_prompts = tiny_prompts
        logprob_value = -2.0  # < log(0.5) ≈ -0.693
        with patch_tinker_modules(logprob_value=logprob_value) as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)

                for call in mocks["ml_logger"].log_metrics.call_args_list:
                    metrics = call[0][0]
                    if "train/accuracy" in metrics:
                        assert metrics["train/accuracy"] == 0.0


class TestSFTEvalSchedule:
    """Validation and eval schedule tests."""

    def test_val_eval_at_each_epoch(self, sft_config, tiny_prompts):
        """run_val_eval called n_epochs times (at each epoch boundary)."""
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval") as mock_val_eval, \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)

                n_epochs = sft_config.num_epochs
                assert mock_val_eval.call_count == n_epochs


class TestSFTMetrics:
    """Metrics logging tests."""

    def test_metrics_logged_at_correct_steps(self, sft_config, tiny_prompts):
        prompts, val_prompts = tiny_prompts
        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(sft_config, ctx, prompts, val_prompts)

                # Collect step numbers from log_metrics calls with train/ keys
                steps = []
                for call in mocks["ml_logger"].log_metrics.call_args_list:
                    metrics = call[0][0]
                    if "train/nll" in metrics:
                        step = call[1].get("step") or call[0][1]
                        steps.append(step)

                # Steps should be 1, 2, 3, 4 (2 batches x 2 epochs)
                assert steps == [1, 2, 3, 4]


class TestSFTBatching:
    """Batch size and drop-remainder tests."""

    def test_batches_drop_remainder(self, tmp_path):
        """6 prompts, batch_size=4 → exactly 1 batch per epoch (2 dropped)."""
        from sgtr_rl.config import TrainingConfig
        from sgtr_rl.data import load_jsonl

        # 6 train records (3 IDs x 2, valid PW, not divisible by batch_size=4)
        train_records = [
            _pw_record("id-1", "1"),
            _pw_record("id-1", "2"),
            _pw_record("id-2", "1"),
            _pw_record("id-2", "2"),
            _pw_record("id-3", "1"),
            _pw_record("id-3", "2"),
        ]
        val_records = [
            _pw_record("val-id-1", "1"),
            _pw_record("val-id-1", "2"),
        ]
        train_path = tmp_path / "train6.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, train_records)
        _write_jsonl(val_path, val_records)

        config = TrainingConfig(
            algorithm="sft",
            experiment_name="test_sft_batch",
            model_name="test-model",
            num_epochs=1,
            batch_size=4,
            train_file=str(train_path),
            val_file=str(val_path),
            run_dir=str(tmp_path / "run"),
        )

        prompts = load_jsonl(str(train_path))
        val_prompts = load_jsonl(str(val_path))

        with patch_tinker_modules() as mocks:
            ctx = _build_ctx(mocks)
            with patch("sgtr_rl.sft.run_val_eval"), \
                 patch("sgtr_rl.sft.run_benchmark_evals"):
                from sgtr_rl.sft import train_sft

                train_sft(config, ctx, prompts, val_prompts)

                # 6 // 4 = 1 batch per epoch, 1 epoch → 1 forward_backward call
                assert mocks["training_client"].forward_backward.call_count == 1
