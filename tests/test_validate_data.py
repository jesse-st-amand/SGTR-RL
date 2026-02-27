"""Tests for sgtr_rl.data_processing.validate_data."""

import pytest

from conftest import _ind_record, _pw_record, write_jsonl
from sgtr_rl.data_processing.validate_data import validate_training_data


# ---------------------------------------------------------------------------
# Baselines — valid data passes
# ---------------------------------------------------------------------------

class TestValidData:
    def test_valid_pw_data_passes(self, pw_train_val_files):
        train_path, val_path = pw_train_val_files
        result = validate_training_data(train_path, val_path)
        assert result["train_records"] == 4
        assert result["val_records"] == 4
        assert result["format"] == "pw"

    def test_valid_ind_data_passes(self, tmp_path):
        train = [_ind_record("t1", "1"), _ind_record("t2", "2")]
        val = [_ind_record("v1", "1"), _ind_record("v2", "2")]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        result = validate_training_data(str(train_path), str(val_path))
        assert result["train_records"] == 2
        assert result["val_records"] == 2
        assert result["format"] == "ind"


# ---------------------------------------------------------------------------
# UUID leakage
# ---------------------------------------------------------------------------

class TestUUIDLeakage:
    def test_uuid_overlap_raises(self, tmp_path):
        """Same UUID in train and val must raise ValueError."""
        shared_uuid = "leaked-uuid"
        train = [
            _pw_record(shared_uuid, "1"),
            _pw_record(shared_uuid, "2"),
        ]
        val = [
            _pw_record(shared_uuid, "1"),
            _pw_record(shared_uuid, "2"),
        ]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        with pytest.raises(ValueError, match="UUID leak"):
            validate_training_data(str(train_path), str(val_path))


# ---------------------------------------------------------------------------
# PW format: ordering check
# ---------------------------------------------------------------------------

class TestPWOrdering:
    def test_pw_missing_ordering_raises(self, tmp_path):
        """PW UUID with only 1 record (missing flip) must raise."""
        train = [_pw_record("u1", "1")]  # only 1 record instead of 2
        val = [
            _pw_record("v1", "1"),
            _pw_record("v1", "2"),
        ]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        with pytest.raises(ValueError, match="don't have exactly 2 records"):
            validate_training_data(str(train_path), str(val_path))


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

class TestTargetValidation:
    def test_target_must_be_string(self, tmp_path):
        """Target=1 (int) instead of '1' (string) must fail target validation."""
        # Build record manually to bypass builder's string typing
        train = [
            {"prompt": "p", "target": 1, "metadata": {"uuid": "u1", "format": "ind"}},
            {"prompt": "p", "target": "2", "metadata": {"uuid": "u2", "format": "ind"}},
        ]
        val = [{"prompt": "p", "target": "1", "metadata": {"uuid": "v1", "format": "ind"}}]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        with pytest.raises(ValueError, match="invalid target"):
            validate_training_data(str(train_path), str(val_path))

    def test_target_invalid_value_raises(self, tmp_path):
        """Target='3' or target='yes' must raise."""
        train = [_ind_record("u1", "3")]
        val = [_ind_record("v1", "1")]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        with pytest.raises(ValueError, match="invalid target"):
            validate_training_data(str(train_path), str(val_path))


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_missing_prompt_raises(self, tmp_path):
        train = [{"target": "1", "metadata": {"uuid": "u1"}}]
        val = [_ind_record("v1", "1")]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        with pytest.raises(ValueError, match="missing required field 'prompt'"):
            validate_training_data(str(train_path), str(val_path))

    def test_missing_metadata_raises(self, tmp_path):
        train = [{"prompt": "p", "target": "1"}]
        val = [_ind_record("v1", "1")]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        with pytest.raises(ValueError, match="missing required field 'metadata'"):
            validate_training_data(str(train_path), str(val_path))

    def test_missing_uuid_raises(self, tmp_path):
        train = [{"prompt": "p", "target": "1", "metadata": {"format": "ind"}}]
        val = [_ind_record("v1", "1")]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        with pytest.raises(ValueError, match="metadata missing 'uuid'"):
            validate_training_data(str(train_path), str(val_path))

    def test_empty_file_raises(self, tmp_path):
        """Empty JSONL file should lead to validation failure or empty summary."""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        train_path.write_text("")
        write_jsonl(val_path, [_ind_record("v1", "1")])
        # Empty train file means 0 records — this should still return a summary
        result = validate_training_data(str(train_path), str(val_path))
        assert result["train_records"] == 0


# ---------------------------------------------------------------------------
# Summary correctness
# ---------------------------------------------------------------------------

class TestSummary:
    def test_returns_correct_summary(self, pw_train_val_files):
        train_path, val_path = pw_train_val_files
        result = validate_training_data(train_path, val_path)
        assert result["train_records"] == 4
        assert result["val_records"] == 4
        assert result["train_uuids"] == 2
        assert result["val_uuids"] == 2
        assert result["format"] == "pw"
        assert "1" in result["train_target_dist"]
        assert "2" in result["train_target_dist"]
