"""Tests for sgtr_rl.data (validate_training_data, build_conversation)."""

import json

import pytest
from conftest import _ind_record, _pw_record

from sgtr_rl.data import (
    build_conversation,
    load_jsonl,
    load_jsonl_many,
    randomize_binary_targets,
    subset_records_by_id,
    validate_training_data,
)


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

    def test_load_jsonl_many_namespaces_ids_for_multiple_sources(self, tmp_path):
        train_a = tmp_path / "dataset_a" / "train.jsonl"
        train_b = tmp_path / "dataset_b" / "train.jsonl"
        train_a.parent.mkdir()
        train_b.parent.mkdir()

        shared_records = [
            _pw_record("shared-id", "1"),
            _pw_record("shared-id", "2"),
        ]
        for path in (train_a, train_b):
            with open(path, "w") as f:
                for record in shared_records:
                    f.write(json.dumps(record) + "\n")

        combined = load_jsonl_many([str(train_a), str(train_b)])

        assert len(combined) == 4
        assert len({record["id"] for record in combined}) == 2
        assert {record["source_dataset"] for record in combined} == {"dataset_a", "dataset_b"}
        assert {record["source_id"] for record in combined} == {"shared-id"}

        summary = validate_training_data(combined, [])
        assert summary["train_ids"] == 2
        assert summary["format"] == "pw"

    def test_load_jsonl_many_per_id_one_source_keeps_one_source(self, tmp_path):
        train_a = tmp_path / "dataset_a" / "train.jsonl"
        train_b = tmp_path / "dataset_b" / "train.jsonl"
        train_a.parent.mkdir()
        train_b.parent.mkdir()

        records_a = [
            _pw_record("id-1", "1"),
            _pw_record("id-1", "2"),
            _pw_record("id-2", "1"),
            _pw_record("id-2", "2"),
        ]
        records_b = [
            _pw_record("id-1", "1"),
            _pw_record("id-1", "2"),
            _pw_record("id-2", "1"),
            _pw_record("id-2", "2"),
        ]
        for path, records in ((train_a, records_a), (train_b, records_b)):
            with open(path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        selected = load_jsonl_many(
            [str(train_a), str(train_b)],
            strategy="per_id_one_source",
            seed=7,
        )

        assert len(selected) == 4
        assert len({record["source_id"] for record in selected}) == 2
        assert len({record["id"] for record in selected}) == 2
        summary = validate_training_data(selected, [])
        assert summary["train_records"] == 4
        assert summary["train_ids"] == 2
        assert summary["format"] == "pw"

    def test_subset_records_by_id_keeps_full_pairwise_examples(self):
        records = [
            _pw_record("id-1", "1"),
            _pw_record("id-1", "2"),
            _pw_record("id-2", "1"),
            _pw_record("id-2", "2"),
            _pw_record("id-3", "1"),
            _pw_record("id-3", "2"),
        ]

        subset = subset_records_by_id(records, 1, seed=7)

        assert len(subset) == 2
        assert len({record["id"] for record in subset}) == 1
        summary = validate_training_data(subset, [])
        assert summary["train_records"] == 2
        assert summary["train_ids"] == 1
        assert summary["format"] == "pw"

    def test_randomize_binary_targets_preserves_pairwise_structure(self):
        records = [
            _pw_record("id-1", "1"),
            _pw_record("id-1", "2"),
            _pw_record("id-2", "1"),
            _pw_record("id-2", "2"),
        ]

        randomized = randomize_binary_targets(records, seed=3)

        assert len(randomized) == len(records)
        assert [record["id"] for record in randomized] == [record["id"] for record in records]
        summary = validate_training_data(randomized, [])
        assert summary["train_records"] == 4
        assert summary["train_ids"] == 2
        assert summary["format"] == "pw"
        for record_id in {"id-1", "id-2"}:
            original_targets = [record["target"] for record in records if record["id"] == record_id]
            randomized_targets = [
                record["target"] for record in randomized if record["id"] == record_id
            ]
            assert randomized_targets in [original_targets, list(reversed(original_targets))]


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
