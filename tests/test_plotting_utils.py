"""Tests for scripts.plotting_utils."""

import json

import numpy as np
import pytest
import yaml

from sgtr_rl.scripts.plotting_utils import _build_title, _smooth, generate_summary_plot

# _build_title

class TestBuildTitle:
    def test_build_title_default_params(self):
        config = {
            "experiment_name": "14_sft_pw",
            "algorithm": "sft",
            "model": {"name": "Qwen/Qwen2-1.5B"},
            "hyperparameters": {
                "learning_rate": 5e-5,
                "batch_size": 16,
            },
            "data": {"dataset": "pw"},
        }
        title = _build_title(config)
        assert "Exp 14" in title
        assert "SFT" in title
        assert "pairwise" in title
        # No non-default params → no lr= in title
        assert "lr=" not in title

    def test_build_title_with_overrides(self):
        config = {
            "experiment_name": "14_sft_pw",
            "algorithm": "sft",
            "model": {"name": "Qwen/Qwen2-1.5B"},
            "hyperparameters": {"learning_rate": 1e-4},
            "data": {"dataset": "pw"},
        }
        title = _build_title(config)
        assert "lr=" in title

    def test_build_title_experiment_number(self):
        config = {
            "experiment_name": "14_sft_pw_uuid_split",
            "algorithm": "sft",
            "model": {},
            "data": {},
        }
        title = _build_title(config)
        assert "Exp 14" in title

    def test_build_title_individual_format(self):
        config = {
            "experiment_name": "02_sft_ind_vs_qwen",
            "algorithm": "sft",
            "model": {"name": "Qwen/Qwen2-1.5B"},
            "data": {"dataset": "sharegpt"},
        }
        title = _build_title(config)
        assert "individual" in title
        assert "pairwise" not in title

    def test_build_title_infers_format_from_train_files(self):
        config = {
            "experiment_name": "03_sft_multi",
            "algorithm": "sft",
            "model": {"name": "Qwen/Qwen2-1.5B"},
            "data": {
                "dataset": "sharegpt",
                "train_files": [
                    "data/training_data/ll-3.1-8b_ICML_01_UT_PW-Q_Rec_NPr_FA_Inst_vs_qwen-2.5-7b_sharegpt/train.jsonl"
                ],
            },
        }
        title = _build_title(config)
        assert "pairwise" in title


# _smooth

class TestSmooth:
    def test_smooth_basic(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _smooth(values, window=3)
        # With padding: [1, 1, 1, 2, 3, 4, 5] convolved with [1/3, 1/3, 1/3]
        # result[0] = (1+1+1)/3 = 1.0, result[1] = (1+1+2)/3 ≈ 1.33, etc.
        assert result[0] == pytest.approx(1.0)
        assert result[-1] == pytest.approx(4.0)

    def test_smooth_single_value(self):
        values = np.array([5.0])
        result = _smooth(values, window=10)
        assert len(result) == 1
        assert result[0] == pytest.approx(5.0)

    def test_smooth_output_length(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = _smooth(values, window=5)
        assert len(result) == len(values)


# generate_summary_plot

class TestGenerateSummaryPlot:
    def test_generate_summary_plot_creates_png(self, tmp_path):
        """With synthetic metrics.jsonl + config.yaml, creates summary.png."""
        metrics_dir = tmp_path / "metrics"
        metrics_dir.mkdir()

        # Write synthetic metrics
        records = []
        for step in range(20):
            records.append({
                "step": step,
                "train/nll": 0.7 - step * 0.01,
                "train/accuracy": 0.5 + step * 0.01,
            })
        # Add val metrics at epoch boundaries (every 10 steps)
        for step in [0, 10]:
            records.append({
                "step": step,
                "val/nll": 0.65 - step * 0.005,
                "val/accuracy": 0.55 + step * 0.005,
            })
        # Add a benchmark metric
        records.append({
            "step": 0,
            "benchmark/mmlu_20/accuracy": 0.25,
        })
        records.append({
            "step": 10,
            "benchmark/mmlu_20/accuracy": 0.30,
        })

        with open(metrics_dir / "metrics.jsonl", "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        # Write config
        config = {
            "experiment_name": "14_sft_pw",
            "algorithm": "sft",
            "model": {"name": "Qwen/Qwen2-1.5B"},
            "hyperparameters": {"learning_rate": 5e-5},
            "data": {"dataset": "pw"},
        }
        with open(tmp_path / "config.yaml", "w") as f:
            yaml.dump(config, f)

        out_path = generate_summary_plot(str(tmp_path))
        assert out_path.exists()
        assert out_path.name == "summary.png"
