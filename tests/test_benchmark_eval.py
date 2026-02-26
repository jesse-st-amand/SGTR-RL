"""Tests for sgtr_rl.training.benchmark_eval."""

import pytest

from sgtr_rl.training.benchmark_eval import (
    extract_mmlu_answer,
    format_mmlu_prompt,
    should_run_benchmark,
)

SAMPLE_ITEM = {
    "question": "What is the capital of France?",
    "choices": ["Berlin", "Madrid", "Paris", "Rome"],
    "subject": "geography",
    "answer": "C",
}


# ---------------------------------------------------------------------------
# format_mmlu_prompt
# ---------------------------------------------------------------------------

class TestFormatMMLUPrompt:
    def test_format_mmlu_prompt_nocot(self):
        prompt = format_mmlu_prompt(SAMPLE_ITEM, cot=False)
        assert "What is the capital of France?" in prompt
        assert "A. Berlin" in prompt
        assert "B. Madrid" in prompt
        assert "C. Paris" in prompt
        assert "D. Rome" in prompt
        assert "single letter" in prompt

    def test_format_mmlu_prompt_cot(self):
        prompt = format_mmlu_prompt(SAMPLE_ITEM, cot=True)
        assert "step by step" in prompt.lower()
        assert "A. Berlin" in prompt


# ---------------------------------------------------------------------------
# extract_mmlu_answer
# ---------------------------------------------------------------------------

class TestExtractMMLUAnswer:
    @pytest.mark.parametrize("text,expected", [
        ("Answer: B", "B"),
        ("answer: c", "C"),
        ("ANSWER=D", "D"),
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
    ])
    def test_extract_mmlu_answer_invalid(self, text):
        assert extract_mmlu_answer(text) is None


# ---------------------------------------------------------------------------
# should_run_benchmark
# ---------------------------------------------------------------------------

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

    def test_should_run_end_only(self):
        assert should_run_benchmark("end_only", 1, epoch=5, total_epochs=10) is False
        assert should_run_benchmark("end_only", 1, epoch=10, total_epochs=10) is True
