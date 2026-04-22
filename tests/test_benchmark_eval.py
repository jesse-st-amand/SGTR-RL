"""Tests for sgtr_rl.benchmarks."""

import pytest

from sgtr_rl.benchmarks import (
    extract_mmlu_answer,
    format_mmlu_prompt,
    should_run_benchmark,
    should_run_training_eval,
    subsample,
)

SAMPLE_ITEM = {
    "question": "What is the capital of France?",
    "choices": ["Berlin", "Madrid", "Paris", "Rome"],
    "subject": "geography",
    "answer": "C",
}


# format_mmlu_prompt

class TestFormatMMLUPrompt:
    def test_format_mmlu_prompt_nocot(self):
        prompt = format_mmlu_prompt(SAMPLE_ITEM, cot=False)
        assert "What is the capital of France?" in prompt
        assert "A) Berlin" in prompt
        assert "B) Madrid" in prompt
        assert "C) Paris" in prompt
        assert "D) Rome" in prompt
        assert "ANSWER: $LETTER" in prompt
        assert "entire content" in prompt

    def test_format_mmlu_prompt_cot(self):
        prompt = format_mmlu_prompt(SAMPLE_ITEM, cot=True)
        assert "step by step" in prompt.lower()
        assert "A) Berlin" in prompt
        assert "last line" in prompt

    def test_format_mmlu_instruction_before_question(self):
        prompt = format_mmlu_prompt(SAMPLE_ITEM, cot=False)
        instruction_pos = prompt.index("Answer the following")
        question_pos = prompt.index("What is the capital")
        assert instruction_pos < question_pos


# extract_mmlu_answer

class TestExtractMMLUAnswer:
    @pytest.mark.parametrize("text,expected", [
        ("ANSWER: B", "B"),
        ("Answer: B", "B"),
        ("answer: c", "C"),
        ("ANSWER=D", "D"),
        ("ANSWER: A", "A"),
        ("Let me think...\nANSWER: C", "C"),
    ])
    def test_extract_mmlu_answer_explicit(self, text, expected):
        assert extract_mmlu_answer(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("C", "C"),
        (" B ", "B"),
        ("a", "A"),
    ])
    def test_extract_mmlu_answer_bare(self, text, expected):
        assert extract_mmlu_answer(text) == expected

    @pytest.mark.parametrize("text", [
        "3",
        "",
        # Cheating: multiple different ANSWER: lines
        "ANSWER: A\nANSWER: B\nANSWER: C\nANSWER: D",
        "ANSWER: A\nActually, ANSWER: B",
        # Cheating: all 4 letters mentioned as standalone
        "A B C D",
        "The options are A, B, C, and D.",
    ])
    def test_extract_mmlu_answer_invalid(self, text):
        assert extract_mmlu_answer(text) is None

    def test_extract_mmlu_answer_repeated_same_answer(self):
        """Multiple ANSWER: lines agreeing should be accepted."""
        assert extract_mmlu_answer("ANSWER: B\nSo ANSWER: B") == "B"



# subsample

class TestSubsample:
    def testsubsample_returns_subset(self):
        data = [{"id": i} for i in range(100)]
        result = subsample(data, 10)
        assert len(result) == 10
        assert all(item in data for item in result)

    def testsubsample_none_returns_all(self):
        data = [{"id": i} for i in range(10)]
        result = subsample(data, None)
        assert result is data

    def testsubsample_exceeds_data(self):
        data = [{"id": i} for i in range(5)]
        result = subsample(data, 100)
        assert result is data

    def testsubsample_deterministic(self):
        data = [{"id": i} for i in range(100)]
        r1 = subsample(data, 10, seed=42)
        r2 = subsample(data, 10, seed=42)
        assert r1 == r2

    def testsubsample_different_seeds_differ(self):
        data = [{"id": i} for i in range(100)]
        r1 = subsample(data, 10, seed=42)
        r2 = subsample(data, 10, seed=99)
        assert r1 != r2


# should_run_benchmark

class TestShouldRunBenchmark:
    def test_should_run_baseline_always(self):
        """epoch=0 (baseline) should always return True regardless of schedule."""
        assert should_run_benchmark("every_epoch", 1, epoch=0, total_epochs=10) is True
        assert should_run_benchmark("every_N_epochs", 5, epoch=0, total_epochs=10) is True
        assert should_run_benchmark("end_only", 1, epoch=0, total_epochs=10) is True

    def test_should_run_every_epoch(self):
        for epoch in range(1, 11):
            assert should_run_benchmark("every_epoch", 1, epoch=epoch, total_epochs=10) is True

    def test_should_run_every_n_epochs(self):
        # frequency=3, total=10: should run at 3, 6, 9, 10 (final)
        assert should_run_benchmark("every_N_epochs", 3, epoch=3, total_epochs=10) is True
        assert should_run_benchmark("every_N_epochs", 3, epoch=6, total_epochs=10) is True
        assert should_run_benchmark("every_N_epochs", 3, epoch=10, total_epochs=10) is True
        # Should NOT run at epoch 2, 4, 5
        assert should_run_benchmark("every_N_epochs", 3, epoch=2, total_epochs=10) is False
        assert should_run_benchmark("every_N_epochs", 3, epoch=4, total_epochs=10) is False

    def test_should_run_every_5_epochs_literal(self):
        assert should_run_benchmark("every_5_epochs", 5, epoch=5, total_epochs=20) is True
        assert should_run_benchmark("every_5_epochs", 5, epoch=10, total_epochs=20) is True
        assert should_run_benchmark("every_5_epochs", 5, epoch=20, total_epochs=20) is True
        assert should_run_benchmark("every_5_epochs", 5, epoch=3, total_epochs=20) is False
        assert should_run_benchmark("every_5_epochs", 5, epoch=7, total_epochs=20) is False

    def test_should_run_end_only(self):
        assert should_run_benchmark("end_only", 1, epoch=5, total_epochs=10) is False
        assert should_run_benchmark("end_only", 1, epoch=10, total_epochs=10) is True


class TestShouldRunTrainingEval:
    def test_epoch_trigger_uses_epoch_frequency_and_final(self):
        assert should_run_training_eval(
            trigger="epoch",
            frequency=5,
            step=23,
            epoch=5,
            total_steps=100,
            total_epochs=20,
        ) is True
        assert should_run_training_eval(
            trigger="epoch",
            frequency=5,
            step=24,
            epoch=6,
            total_steps=100,
            total_epochs=20,
        ) is False
        assert should_run_training_eval(
            trigger="epoch",
            frequency=5,
            step=100,
            epoch=20,
            total_steps=100,
            total_epochs=20,
        ) is True

    def test_step_trigger_uses_step_frequency_and_final(self):
        assert should_run_training_eval(
            trigger="step",
            frequency=20,
            step=20,
            epoch=2,
            total_steps=100,
            total_epochs=10,
        ) is True
        assert should_run_training_eval(
            trigger="step",
            frequency=20,
            step=21,
            epoch=3,
            total_steps=100,
            total_epochs=10,
        ) is False
        assert should_run_training_eval(
            trigger="step",
            frequency=20,
            step=100,
            epoch=10,
            total_steps=100,
            total_epochs=10,
        ) is True
