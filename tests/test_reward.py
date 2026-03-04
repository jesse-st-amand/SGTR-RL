"""Tests for sgtr_rl.reward and sgtr_rl.answer."""

import pytest

from sgtr_rl.answer import extract_answer
from sgtr_rl.reward import sgtr_binary_reward

# ---------------------------------------------------------------------------
# extract_answer
# ---------------------------------------------------------------------------

class TestExtractAnswer:
    @pytest.mark.parametrize("text,expected", [
        ("Answer: 1", "1"),
        ("ANSWER=2", "2"),
        ("answer: 1", "1"),
        ("The answer: 2", "2"),
    ])
    def test_extract_explicit_answer_pattern(self, text, expected):
        assert extract_answer(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("I think 1 but then 2", "2"),
        ("Option 1 is wrong, so 2", "2"),
        ("Maybe 2 or maybe 1", "1"),
    ])
    def test_extract_last_standalone_digit(self, text, expected):
        assert extract_answer(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("1", "1"),
        (" 2 ", "2"),
    ])
    def test_extract_bare_digit(self, text, expected):
        assert extract_answer(text) == expected

    @pytest.mark.parametrize("text", [
        "",
        "hello",
        "Answer: 3",
        "no digits here",
    ])
    def test_extract_no_answer(self, text):
        assert extract_answer(text) is None

    def test_extract_cot_then_answer(self):
        text = "Let me think step by step... The writing style matches. Answer: 1"
        assert extract_answer(text) == "1"


# ---------------------------------------------------------------------------
# sgtr_binary_reward
# ---------------------------------------------------------------------------

class TestBinaryReward:
    def test_binary_reward_correct(self):
        rewards = sgtr_binary_reward(["Answer: 1", "2"], ["1", "2"])
        assert rewards == [1.0, 1.0]

    def test_binary_reward_incorrect(self):
        rewards = sgtr_binary_reward(["Answer: 1", "Answer: 1"], ["2", "2"])
        assert rewards == [0.0, 0.0]

    def test_binary_reward_no_answer(self):
        rewards = sgtr_binary_reward(["hello", ""], ["1", "2"])
        assert rewards == [0.0, 0.0]
