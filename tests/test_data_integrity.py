"""Data integrity tests on actual data files.

All tests are marked @pytest.mark.datasci and skip if data files don't exist
(so CI without data still passes).

Note: These tests support both the old nested metadata format and the new flat
format. Real data files may be in either format depending on whether they've
been re-extracted with the updated prepare_data.py script.
"""

import json
from pathlib import Path

import pytest

from sgtr_rl.data import validate_training_data

PW_TRAIN = Path("data/training_data/llama8b_pw/train.jsonl")
PW_VAL = Path("data/training_data/llama8b_pw/val.jsonl")
MMLU_20 = Path("data/benchmarks/mmlu_20.jsonl")
MMLU_500 = Path("data/benchmarks/mmlu_500.jsonl")

pw_data_exists = pytest.mark.skipif(
    not (PW_TRAIN.exists() and PW_VAL.exists()),
    reason="PW training data not available",
)
benchmark_data_exists = pytest.mark.skipif(
    not (MMLU_20.exists() and MMLU_500.exists()),
    reason="Benchmark data files not available",
)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _get_record_id(rec: dict) -> str:
    """Get the record ID, supporting both flat and nested formats."""
    if "id" in rec:
        return rec["id"]
    return rec.get("metadata", {}).get("uuid", "")


def _is_flat_format(records: list[dict]) -> bool:
    """Check if records use the flat schema (has 'id' at top level)."""
    return len(records) > 0 and "id" in records[0]


# ---------------------------------------------------------------------------
# PW data validation
# ---------------------------------------------------------------------------

@pytest.mark.datasci
@pw_data_exists
class TestPWDataIntegrity:
    def test_pw_data_valid(self):
        """Full validation passes on real PW data (flat format only)."""
        train = _load_jsonl(PW_TRAIN)
        if not _is_flat_format(train):
            pytest.skip("Data uses old nested format — re-extract with prepare_data.py")
        result = validate_training_data(str(PW_TRAIN), str(PW_VAL))
        assert result["format"] == "pw"

    def test_pw_no_id_leakage(self):
        """Zero ID overlap between actual train/val."""
        train = _load_jsonl(PW_TRAIN)
        val = _load_jsonl(PW_VAL)
        train_ids = {_get_record_id(r) for r in train}
        val_ids = {_get_record_id(r) for r in val}
        assert len(train_ids & val_ids) == 0

    def test_pw_target_balance(self):
        """Both targets '1' and '2' present in train and val."""
        train = _load_jsonl(PW_TRAIN)
        val = _load_jsonl(PW_VAL)
        train_targets = {r["target"] for r in train}
        val_targets = {r["target"] for r in val}
        assert train_targets == {"1", "2"}
        assert val_targets == {"1", "2"}

    def test_pw_record_counts(self):
        """Expect 160 train, 40 val records."""
        train = _load_jsonl(PW_TRAIN)
        val = _load_jsonl(PW_VAL)
        assert len(train) == 160
        assert len(val) == 40


# ---------------------------------------------------------------------------
# Benchmark data validation
# ---------------------------------------------------------------------------

@pytest.mark.datasci
@benchmark_data_exists
class TestBenchmarkDataIntegrity:
    def test_benchmark_files_exist(self):
        """mmlu_20 has 20 items, mmlu_500 has 500."""
        mmlu_20 = _load_jsonl(MMLU_20)
        mmlu_500 = _load_jsonl(MMLU_500)
        assert len(mmlu_20) == 20
        assert len(mmlu_500) == 500

    def test_benchmark_schema(self):
        """Each item has question, choices (4), subject, answer (A-D)."""
        for path in [MMLU_20, MMLU_500]:
            items = _load_jsonl(path)
            for item in items:
                assert "question" in item
                assert "choices" in item
                assert len(item["choices"]) == 4
                assert "subject" in item
                assert "answer" in item
                assert item["answer"] in ("A", "B", "C", "D")
