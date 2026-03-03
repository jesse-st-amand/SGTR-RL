"""Low-level integration tests for TinkerRLTrainer (GRPO).

Uses the tinker_mocks fixture to inject fake tinker modules, then runs
the real training loop and asserts on pipeline math and orchestration.
"""

from unittest.mock import patch, MagicMock

from tests.integration.conftest import patch_tinker_modules


def _make_varying_answer():
    """Return a side_effect function that alternates "1"/"2" answers."""
    state = {"n": 0}

    def fn(*args, **kwargs):
        state["n"] += 1
        return "1" if state["n"] % 2 == 1 else "2"

    return fn


class TestGRPOSmoke:
    """Basic wiring and import tests."""

    def test_smoke_runs_to_completion(self, grpo_config):
        with patch_tinker_modules(num_sequences=2):
            from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

            trainer = TinkerRLTrainer(grpo_config)
            trainer.train()


class TestGRPOLossFunction:
    """Loss function configuration tests."""

    def test_forward_backward_uses_importance_sampling(self, grpo_config):
        """forward_backward called with loss_fn='importance_sampling'."""
        with patch_tinker_modules(num_sequences=2) as mocks:
            # Use varying answers so some groups have signal
            mocks["renderers_mod"].get_text_content = MagicMock(
                side_effect=_make_varying_answer()
            )
            with patch("sgtr_rl.training.grpo_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.grpo_trainer.run_benchmark_evals"):
                from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

                trainer = TinkerRLTrainer(grpo_config)
                trainer.train()

                assert mocks["training_client"].forward_backward.call_count > 0
                for call in mocks["training_client"].forward_backward.call_args_list:
                    _, kwargs = call
                    assert kwargs.get("loss_fn") == "importance_sampling"


class TestGRPOAdvantages:
    """GRPO advantage centering tests."""

    def test_advantage_centering(self, grpo_config):
        """Groups with mixed rewards produce non-zero advantages and training signal."""
        with patch_tinker_modules(num_sequences=2) as mocks:
            mocks["renderers_mod"].get_text_content = MagicMock(
                side_effect=_make_varying_answer()
            )
            with patch("sgtr_rl.training.grpo_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.grpo_trainer.run_benchmark_evals"):
                from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

                trainer = TinkerRLTrainer(grpo_config)
                trainer.train()

                # With mixed rewards, forward_backward should be called
                assert mocks["training_client"].forward_backward.call_count > 0


class TestGRPOZeroSignal:
    """Zero-signal group skipping tests."""

    def test_zero_signal_groups_skipped(self, grpo_config):
        """When all completions get same reward → no datums → no forward_backward."""
        # All answers are "1": groups with target "1" get [1.0, 1.0],
        # groups with target "2" get [0.0, 0.0]. Both uniform → skipped.
        with patch_tinker_modules(answer_text="1", num_sequences=2) as mocks:
            with patch("sgtr_rl.training.grpo_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.grpo_trainer.run_benchmark_evals"):
                from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

                trainer = TinkerRLTrainer(grpo_config)
                trainer.train()

                # All groups had zero signal, so forward_backward should not be called
                assert mocks["training_client"].forward_backward.call_count == 0


class TestGRPOSampling:
    """Sampling configuration tests."""

    def test_sampling_uses_group_size(self, grpo_config):
        with patch_tinker_modules(num_sequences=2) as mocks:
            with patch("sgtr_rl.training.grpo_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.grpo_trainer.run_benchmark_evals"):
                from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

                trainer = TinkerRLTrainer(grpo_config)
                trainer.train()

                for call in mocks["sampling_client"].sample.call_args_list:
                    _, kwargs = call
                    assert kwargs["num_samples"] == grpo_config.num_rollouts_per_prompt


class TestGRPORunningAccuracy:
    """Running accuracy tracking tests."""

    def test_running_accuracy_tracked(self, grpo_config):
        """Cumulative correct/total tracked across batches."""
        with patch_tinker_modules(num_sequences=2) as mocks:
            mocks["renderers_mod"].get_text_content = MagicMock(
                side_effect=_make_varying_answer()
            )
            with patch("sgtr_rl.training.grpo_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.grpo_trainer.run_benchmark_evals"):
                from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

                trainer = TinkerRLTrainer(grpo_config)
                trainer.train()

                has_running_acc = False
                for call in mocks["ml_logger"].log_metrics.call_args_list:
                    metrics = call[0][0]
                    if "train/running_accuracy" in metrics:
                        has_running_acc = True
                        assert 0.0 <= metrics["train/running_accuracy"] <= 1.0
                assert has_running_acc


class TestGRPOReward:
    """Reward function wiring tests."""

    def test_reward_delegates_correctly(self):
        """_get_reward("1", "1") → 1.0, _get_reward("2", "1") → 0.0."""
        from sgtr_rl.training.grpo_trainer import TinkerRLTrainer
        from sgtr_rl.training.train_config import TrainingConfig

        config = TrainingConfig(algorithm="grpo")
        trainer = TinkerRLTrainer(config)
        assert trainer._get_reward("1", "1") == 1.0
        assert trainer._get_reward("2", "1") == 0.0
        assert trainer._get_reward("1", "2") == 0.0
        assert trainer._get_reward("2", "2") == 1.0


class TestGRPODatumConstruction:
    """Datum construction and padding tests."""

    def test_datum_construction(self, grpo_config):
        """Datums are constructed and passed to forward_backward."""
        with patch_tinker_modules(num_sequences=2) as mocks:
            mocks["renderers_mod"].get_text_content = MagicMock(
                side_effect=_make_varying_answer()
            )
            with patch("sgtr_rl.training.grpo_trainer.run_val_eval"), \
                 patch("sgtr_rl.training.grpo_trainer.run_benchmark_evals"):
                from sgtr_rl.training.grpo_trainer import TinkerRLTrainer

                trainer = TinkerRLTrainer(grpo_config)
                trainer.train()

                assert mocks["training_client"].forward_backward.call_count > 0
                for call in mocks["training_client"].forward_backward.call_args_list:
                    datums = call[0][0]
                    assert len(datums) > 0
