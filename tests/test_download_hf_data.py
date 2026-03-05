"""Tests for scripts.prepare_data (download/extract utilities)."""


from scripts.prepare_data import (
    detect_format_from_experiment,
    filter_files,
    parse_eval_filename,
)

# Sample filenames from real data
PW_FILENAME = (
    "2026-01-21T14-26-18+00-00_ll-3.1-8b-eval-on-ll-3.1-8b-vs-qwen-2.5-7b"
    "_VR62kDdqZdxTWwRA2qPq7z.eval"
)
IND_CONTROL_FILENAME = (
    "2026-01-22T01-47-29+00-00_ll-3.1-8b-eval-on-ll-3.1-8b-control"
    "_jtq8oXbypQK8gNk7y5woFh.eval"
)
IND_TREATMENT_FILENAME = (
    "2026-01-22T01-47-29+00-00_ll-3.1-8b-eval-on-qwen-2.5-7b-treatment"
    "_oFtE6xvuPt8jidLNeVXoL8.eval"
)

# Sample HF repo file paths
SAMPLE_FILES = [
    "sharegpt/english_26/ICML_01_UT_PW-Q_Rec_NPr_FA_Inst/" + PW_FILENAME,
    "sharegpt/english_26/ICML_02_UT_IND-Q_Rec_NPr_FA_Inst/" + IND_CONTROL_FILENAME,
    "sharegpt/english_26/ICML_02_UT_IND-Q_Rec_NPr_FA_Inst/" + IND_TREATMENT_FILENAME,
    "sharegpt/english2_74/ICML_01_UT_PW-Q_Rec_NPr_FA_Inst/" + PW_FILENAME,
    "wikisum/split_a/ICML_01_UT_PW-Q_Rec_NPr_FA_Inst/" + PW_FILENAME,
    "README.md",
]


# ---------------------------------------------------------------------------
# parse_eval_filename
# ---------------------------------------------------------------------------


class TestParseEvalFilename:
    def test_pw_filename(self):
        result = parse_eval_filename(PW_FILENAME)
        assert result["evaluator"] == "ll-3.1-8b"
        assert result["self_model"] == "ll-3.1-8b"
        assert result["opponent"] == "qwen-2.5-7b"

    def test_ind_control_filename(self):
        result = parse_eval_filename(IND_CONTROL_FILENAME)
        assert result["evaluator"] == "ll-3.1-8b"
        assert result["self_model"] == "ll-3.1-8b-control"
        assert result["opponent"] is None

    def test_ind_treatment_filename(self):
        result = parse_eval_filename(IND_TREATMENT_FILENAME)
        assert result["evaluator"] == "ll-3.1-8b"
        assert result["self_model"] == "qwen-2.5-7b-treatment"
        assert result["opponent"] is None

    def test_invalid_filename(self):
        result = parse_eval_filename("random_file.eval")
        assert result["evaluator"] is None
        assert result["self_model"] is None


# ---------------------------------------------------------------------------
# detect_format_from_experiment
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_pw(self):
        assert detect_format_from_experiment("ICML_01_UT_PW-Q_Rec_NPr_FA_Inst") == "pw"

    def test_ind(self):
        assert detect_format_from_experiment("ICML_02_UT_IND-Q_Rec_NPr_FA_Inst") == "ind"

    def test_unknown(self):
        assert detect_format_from_experiment("ICML_03_something_else") is None


# ---------------------------------------------------------------------------
# filter_files
# ---------------------------------------------------------------------------


class TestFilterFiles:
    def test_filter_by_evaluator(self):
        result = filter_files(SAMPLE_FILES, evaluator="ll-3.1-8b")
        assert len(result) == 5  # all .eval files match
        assert "README.md" not in result

    def test_filter_by_nonexistent_evaluator(self):
        result = filter_files(SAMPLE_FILES, evaluator="gpt-4")
        assert len(result) == 0

    def test_filter_by_dataset(self):
        result = filter_files(SAMPLE_FILES, dataset="sharegpt")
        assert len(result) == 4
        assert all("sharegpt/" in f for f in result)

    def test_filter_by_dataset_wikisum(self):
        result = filter_files(SAMPLE_FILES, dataset="wikisum")
        assert len(result) == 1

    def test_filter_by_experiment(self):
        result = filter_files(
            SAMPLE_FILES, experiments=["ICML_01_UT_PW-Q_Rec_NPr_FA_Inst"]
        )
        assert len(result) == 3
        assert all("PW" in f for f in result)

    def test_filter_by_split(self):
        result = filter_files(SAMPLE_FILES, splits=["english_26"])
        assert len(result) == 3

    def test_filter_by_generator(self):
        result = filter_files(SAMPLE_FILES, generators=["qwen-2.5-7b"])
        # 3 PW files (-vs-qwen) + 1 IND treatment (-eval-on-qwen), not the control
        assert len(result) == 4

    def test_filter_combined(self):
        result = filter_files(
            SAMPLE_FILES,
            dataset="sharegpt",
            experiments=["ICML_02_UT_IND-Q_Rec_NPr_FA_Inst"],
        )
        assert len(result) == 2  # only IND experiments in sharegpt/english_26

    def test_no_filters_returns_all_eval(self):
        result = filter_files(SAMPLE_FILES)
        assert len(result) == 5  # all .eval files, not README.md

    def test_eval_extension_required(self):
        result = filter_files(["sharegpt/split/exp/file.txt"])
        assert len(result) == 0
