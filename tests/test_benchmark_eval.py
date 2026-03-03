"""Tests for sgtr_rl.training.benchmark_eval."""

import pytest

from sgtr_rl.training.benchmark_eval import (
    _filter_by_model,
    _subsample,
    extract_mmlu_answer,
    format_mmlu_prompt,
    should_run_benchmark,
)
from sgtr_rl.training.utils import flip_target

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


# ---------------------------------------------------------------------------
# extract_mmlu_answer
# ---------------------------------------------------------------------------

class TestExtractMMLUAnswer:
    @pytest.mark.parametrize("text,expected", [
        ("ANSWER: B", "B"),
        ("Answer: B", "B"),
        ("answer: c", "C"),
        ("ANSWER=D", "D"),
        ("ANSWER: A", "A"),
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

# ---------------------------------------------------------------------------
# flip_target (from utils)
# ---------------------------------------------------------------------------

class TestFlipTarget:
    def test_flip_target_1_to_2(self):
        assert flip_target("1") == "2"

    def test_flip_target_2_to_1(self):
        assert flip_target("2") == "1"

    def test_flip_target_other_unchanged(self):
        assert flip_target("A") == "A"
        assert flip_target("") == ""
        assert flip_target("3") == "3"


# ---------------------------------------------------------------------------
# _subsample
# ---------------------------------------------------------------------------

class TestSubsample:
    def test_subsample_returns_subset(self):
        data = [{"id": i} for i in range(100)]
        result = _subsample(data, 10)
        assert len(result) == 10
        assert all(item in data for item in result)

    def test_subsample_none_returns_all(self):
        data = [{"id": i} for i in range(10)]
        result = _subsample(data, None)
        assert result is data

    def test_subsample_exceeds_data(self):
        data = [{"id": i} for i in range(5)]
        result = _subsample(data, 100)
        assert result is data

    def test_subsample_deterministic(self):
        data = [{"id": i} for i in range(100)]
        r1 = _subsample(data, 10, seed=42)
        r2 = _subsample(data, 10, seed=42)
        assert r1 == r2

    def test_subsample_different_seeds_differ(self):
        data = [{"id": i} for i in range(100)]
        r1 = _subsample(data, 10, seed=42)
        r2 = _subsample(data, 10, seed=99)
        assert r1 != r2


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

    def test_should_run_every_5_epochs_literal(self):
        # Config files use "every_5_epochs" not "every_N_epochs"
        assert should_run_benchmark("every_5_epochs", 5, epoch=5, total_epochs=20) is True
        assert should_run_benchmark("every_5_epochs", 5, epoch=10, total_epochs=20) is True
        assert should_run_benchmark("every_5_epochs", 5, epoch=20, total_epochs=20) is True
        assert should_run_benchmark("every_5_epochs", 5, epoch=3, total_epochs=20) is False
        assert should_run_benchmark("every_5_epochs", 5, epoch=7, total_epochs=20) is False

    def test_should_run_end_only(self):
        assert should_run_benchmark("end_only", 1, epoch=5, total_epochs=10) is False
        assert should_run_benchmark("end_only", 1, epoch=10, total_epochs=10) is True


# ---------------------------------------------------------------------------
# _filter_by_model
# ---------------------------------------------------------------------------

class TestFilterByModel:
    def test_filter_pw_by_treatment_name_2(self):
        data = [
            {"prompt": "a", "metadata": {"treatment_name_1": "ll-3.1-8b", "treatment_name_2": "qwen-2.5-7b"}},
            {"prompt": "b", "metadata": {"treatment_name_1": "ll-3.1-8b", "treatment_name_2": "haiku-3.5"}},
            {"prompt": "c", "metadata": {"treatment_name_1": "ll-3.1-8b", "treatment_name_2": "qwen-2.5-7b"}},
        ]
        result = _filter_by_model(data, "qwen-2.5-7b")
        assert len(result) == 2
        assert result[0]["prompt"] == "a"
        assert result[1]["prompt"] == "c"

    def test_filter_ind_keeps_control_and_matching_treatment(self):
        data = [
            {"prompt": "ctrl", "metadata": {"treatment_name": "ll-3.1-8b", "is_control": True}},
            {"prompt": "qwen", "metadata": {"treatment_name": "qwen-2.5-7b", "is_control": False}},
            {"prompt": "haiku", "metadata": {"treatment_name": "haiku-3.5", "is_control": False}},
        ]
        result = _filter_by_model(data, "qwen-2.5-7b")
        assert len(result) == 2
        assert result[0]["prompt"] == "ctrl"
        assert result[1]["prompt"] == "qwen"

    def test_filter_returns_empty_for_no_match(self):
        data = [
            {"prompt": "a", "metadata": {"treatment_name_1": "ll-3.1-8b", "treatment_name_2": "haiku-3.5"}},
        ]
        result = _filter_by_model(data, "gpt-4o")
        assert len(result) == 0

    def test_filter_no_metadata(self):
        data = [{"prompt": "a"}]
        result = _filter_by_model(data, "qwen-2.5-7b")
        assert len(result) == 0
