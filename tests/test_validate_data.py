"""Tests for sgtr_rl.data (validate_training_data, build_conversation)."""

import pytest
from conftest import _ind_record, _pw_record

from sgtr_rl.data import build_conversation, load_jsonl, validate_training_data


class TestBuildConversation:
    def test_string_prompt(self):
        item = {"prompt": "Which is yours?", "target": "1", "id": "u1"}
        convo = build_conversation(item, use_system_prompt=False)
        assert convo == [{"role": "user", "content": "Which is yours?"}]

    def test_multiturn_prompt(self):
        """Chat-format (AT/COLM) prompts are passed through as-is."""
        messages = [
            {"role": "user", "content": "Write code"},
            {"role": "assistant", "content": "def foo(): ..."},
            {"role": "user", "content": "Which is yours?"},
        ]
        item = {"prompt": messages, "target": "1", "id": "u1"}
        convo = build_conversation(item)
        assert convo == messages
        assert convo is not messages

    def test_with_system_prompt_string(self):
        item = {
            "prompt": "Which is yours?",
            "target": "1",
            "id": "u1",
            "system_prompt": "Be helpful.",
        }
        convo = build_conversation(item, use_system_prompt=True)
        assert len(convo) == 2
        assert convo[0] == {"role": "system", "content": "Be helpful."}
        assert convo[1] == {"role": "user", "content": "Which is yours?"}

    def test_with_system_prompt_multiturn(self):
        """System prompt is prepended to multi-turn conversations."""
        messages = [
            {"role": "user", "content": "Write code"},
            {"role": "assistant", "content": "def foo(): ..."},
            {"role": "user", "content": "Which is yours?"},
        ]
        item = {"prompt": messages, "target": "1", "id": "u1", "system_prompt": "Be helpful."}
        convo = build_conversation(item, use_system_prompt=True)
        assert len(convo) == 4
        assert convo[0] == {"role": "system", "content": "Be helpful."}
        assert convo[1:] == messages

    def test_system_prompt_enabled_but_missing_from_record(self):
        item = {"prompt": "Which is yours?", "target": "1", "id": "u1"}
        convo = build_conversation(item, use_system_prompt=True)
        assert convo == [{"role": "user", "content": "Which is yours?"}]

    def test_system_prompt_present_but_disabled(self):
        item = {
            "prompt": "Which is yours?",
            "target": "1",
            "id": "u1",
            "system_prompt": "Be helpful.",
        }
        convo = build_conversation(item, use_system_prompt=False)
        assert convo == [{"role": "user", "content": "Which is yours?"}]


class TestValidData:
    def test_valid_pw_data_passes(self, pw_train_val_files):
        train_path, val_path = pw_train_val_files
        train = load_jsonl(train_path)
        val = load_jsonl(val_path)
        result = validate_training_data(train, val)
        assert result["train_records"] == 4
        assert result["val_records"] == 4
        assert result["format"] == "pw"

    def test_valid_ind_data_passes(self):
        train = [_ind_record("t1", "1"), _ind_record("t2", "2")]
        val = [_ind_record("v1", "1"), _ind_record("v2", "2")]
        result = validate_training_data(train, val)
        assert result["train_records"] == 2
        assert result["val_records"] == 2
        assert result["format"] == "ind"


class TestIDLeakage:
    def test_id_overlap_raises(self):
        shared_id = "leaked-id"
        train = [
            _pw_record(shared_id, "1"),
            _pw_record(shared_id, "2"),
        ]
        val = [
            _pw_record(shared_id, "1"),
            _pw_record(shared_id, "2"),
        ]
        with pytest.raises(ValueError, match="ID leak"):
            validate_training_data(train, val)


class TestPWOrdering:
    def test_pw_missing_ordering_raises(self):
        train = [_pw_record("u1", "1")]
        val = [
            _pw_record("v1", "1"),
            _pw_record("v1", "2"),
        ]
        with pytest.raises(ValueError, match="don't have exactly 2 records"):
            validate_training_data(train, val)


class TestTargetValidation:
    def test_target_must_be_string(self):
        train = [
            {"prompt": "p", "target": 1, "id": "u1", "format": "ind"},
            {"prompt": "p", "target": "2", "id": "u2", "format": "ind"},
        ]
        val = [{"prompt": "p", "target": "1", "id": "v1", "format": "ind"}]
        with pytest.raises(ValueError, match="invalid target"):
            validate_training_data(train, val)

    def test_target_invalid_value_raises(self):
        train = [_ind_record("u1", "3")]
        val = [_ind_record("v1", "1")]
        with pytest.raises(ValueError, match="invalid target"):
            validate_training_data(train, val)


class TestSchemaValidation:
    def test_missing_prompt_raises(self):
        train = [{"target": "1", "id": "u1"}]
        val = [_ind_record("v1", "1")]
        with pytest.raises(ValueError, match="missing required field 'prompt'"):
            validate_training_data(train, val)

    def test_missing_id_raises(self):
        train = [{"prompt": "p", "target": "1"}]
        val = [_ind_record("v1", "1")]
        with pytest.raises(ValueError, match="missing required field 'id'"):
            validate_training_data(train, val)

    def test_empty_train(self):
        result = validate_training_data([], [_ind_record("v1", "1")])
        assert result["train_records"] == 0


class TestSummary:
    def test_returns_correct_summary(self, pw_train_val_files):
        train_path, val_path = pw_train_val_files
        train = load_jsonl(train_path)
        val = load_jsonl(val_path)
        result = validate_training_data(train, val)
        assert result["train_records"] == 4
        assert result["val_records"] == 4
        assert result["train_ids"] == 2
        assert result["val_ids"] == 2
        assert result["format"] == "pw"
        assert "1" in result["train_target_dist"]
        assert "2" in result["train_target_dist"]
