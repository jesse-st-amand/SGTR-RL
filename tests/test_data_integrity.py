"""Data integrity tests on actual data files.

All tests are marked @pytest.mark.datasci and skip if data files don't exist
(so CI without data still passes).
"""

import json
from pathlib import Path

import pytest

from sgtr_rl.data_processing.validate_data import validate_training_data

PW_TRAIN = Path("data/training_data/sharegpt_pw/train.jsonl")
PW_VAL = Path("data/training_data/sharegpt_pw/val.jsonl")
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


# ---------------------------------------------------------------------------
# PW data validation
# ---------------------------------------------------------------------------

@pytest.mark.datasci
@pw_data_exists
class TestPWDataIntegrity:
    def test_pw_data_valid(self):
        """Full validation passes on real PW data."""
        result = validate_training_data(str(PW_TRAIN), str(PW_VAL))
        assert result["format"] == "pw"

    def test_pw_no_uuid_leakage(self):
        """Zero UUID overlap between actual train/val."""
        train = _load_jsonl(PW_TRAIN)
        val = _load_jsonl(PW_VAL)
        train_uuids = {r["metadata"]["uuid"] for r in train}
        val_uuids = {r["metadata"]["uuid"] for r in val}
        assert len(train_uuids & val_uuids) == 0

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
