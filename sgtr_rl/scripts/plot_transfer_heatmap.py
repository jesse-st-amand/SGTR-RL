#!/usr/bin/env python3
"""Plot no-averaging transfer heatmaps from completed SGTR runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sgtr_rl.transfer_result_sources import TransferRunResolver

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "transfer_plots"

DATASET_KEYS = {
    "ShareGPT": "sharegpt",
    "WikiSum": "wikisum",
    "BigCode": "bigcodebench",
    "PKU": "pku",
}

TASK_EVAL_ORDER = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)"]
PREF_EVAL_ORDER = ["PW Pref", "IND Pref"]
DATASET_EVAL_ORDER = ["ShareGPT", "WikiSum", "BigCode", "PKU"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot transfer heatmaps from explicit run sources."
    )
    parser.add_argument(
        "--source",
        choices=["old", "clean", "compare"],
        default="old",
        help="Which run source to plot.",
    )
    parser.add_argument(
        "--plot-set",
        choices=["standard", "adversarial", "all"],
        default="standard",
        help="Which experiment family to render.",
    )
    parser.add_argument(
        "--clean-manifest",
        action="append",
        default=[],
        help="Explicit clean batch manifest(s). Defaults to the latest standard manifests.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip rows whose requested runs are not completed yet instead of failing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base directory for rendered plots.",
    )
    return parser.parse_args()


def standard_specs() -> list[dict]:
    return [
        {
            "group": "Llama 3.1 8B vs Qwen-2.5-7B",
            "rows": [
                {
                    "label": "Llama 3.1 8B | vs Qwen-2.5-7B | UT PW ShareGPT",
                    "glob": "11_archived_ll8b_ut_pw_sharegpt_vs_qwen25__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen-2.5-7B | UT PW WikiSum",
                    "glob": "12_archived_ll8b_ut_pw_wikisum_vs_qwen25__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "WikiSum",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen-2.5-7B | UT PW BigCode",
                    "glob": "13_archived_ll8b_ut_pw_bigcodebench_vs_qwen25__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "BigCode",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen-2.5-7B | UT PW PKU",
                    "glob": "14_archived_ll8b_ut_pw_pku_vs_qwen25__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "PKU",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen-2.5-7B | UT IND ShareGPT",
                    "glob": "15_archived_ll8b_ut_ind_sharegpt_vs_qwen25__*",
                    "trained_task": "IND (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen-2.5-7B | AT PW ShareGPT",
                    "glob": "16_archived_ll8b_at_pw_sharegpt_vs_qwen25__*",
                    "trained_task": "PW (AT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen-2.5-7B | AT IND ShareGPT",
                    "glob": "17_archived_ll8b_at_ind_sharegpt_vs_qwen25__*",
                    "trained_task": "IND (AT)",
                    "trained_dataset": "ShareGPT",
                },
            ],
        },
        {
            "group": "Llama 3.1 8B vs Qwen3-30B",
            "rows": [
                {
                    "label": "Llama 3.1 8B | vs Qwen3-30B | UT PW ShareGPT",
                    "glob": "01_sft_pw_vs_qwen3_30b_tinker_small__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen3-30B | UT PW WikiSum",
                    "glob": "11_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_wikisum__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "WikiSum",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen3-30B | UT PW BigCode",
                    "glob": "12_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_bigcodebench__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "BigCode",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen3-30B | UT PW PKU",
                    "glob": "13_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_pku__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "PKU",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen3-30B | UT IND ShareGPT",
                    "glob": "14_sft_ut_ind_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "trained_task": "IND (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen3-30B | AT PW ShareGPT",
                    "glob": "15_sft_at_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "trained_task": "PW (AT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Llama 3.1 8B | vs Qwen3-30B | AT IND ShareGPT",
                    "glob": "16_sft_at_ind_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "trained_task": "IND (AT)",
                    "trained_dataset": "ShareGPT",
                },
            ],
        },
        {
            "group": "Qwen 3.0 30B vs Llama 3.1 8B",
            "rows": [
                {
                    "label": "Qwen 3.0 30B | vs Llama 3.1 8B | UT PW ShareGPT",
                    "glob": "01_sft_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Qwen 3.0 30B | vs Llama 3.1 8B | UT PW WikiSum",
                    "glob": "11_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_wikisum__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "WikiSum",
                },
                {
                    "label": "Qwen 3.0 30B | vs Llama 3.1 8B | UT PW BigCode",
                    "glob": "12_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_bigcodebench__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "BigCode",
                },
                {
                    "label": "Qwen 3.0 30B | vs Llama 3.1 8B | UT PW PKU",
                    "glob": "13_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_pku__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "PKU",
                },
                {
                    "label": "Qwen 3.0 30B | vs Llama 3.1 8B | UT IND ShareGPT",
                    "glob": "14_sft_ut_ind_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "trained_task": "IND (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Qwen 3.0 30B | vs Llama 3.1 8B | AT PW ShareGPT",
                    "glob": "15_sft_at_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "trained_task": "PW (AT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Qwen 3.0 30B | vs Llama 3.1 8B | AT IND ShareGPT",
                    "glob": "16_sft_at_ind_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "trained_task": "IND (AT)",
                    "trained_dataset": "ShareGPT",
                },
            ],
        },
        {
            "group": "Qwen 3.0 30B vs GPT-OSS-120B-Thinking",
            "rows": [
                {
                    "label": "Qwen 3.0 30B | vs GPT-OSS-120B-Th | UT PW ShareGPT",
                    "glob": "21_archived_qwen30_ut_pw_sharegpt_vs_oss120__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Qwen 3.0 30B | vs GPT-OSS-120B-Th | UT PW WikiSum",
                    "glob": "31_archived_qwen30_ut_pw_wikisum_vs_oss120__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "WikiSum",
                },
                {
                    "label": "Qwen 3.0 30B | vs GPT-OSS-120B-Th | UT PW BigCode",
                    "glob": "32_archived_qwen30_ut_pw_bigcodebench_vs_oss120__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "BigCode",
                },
                {
                    "label": "Qwen 3.0 30B | vs GPT-OSS-120B-Th | UT PW PKU",
                    "glob": "33_archived_qwen30_ut_pw_pku_vs_oss120__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "PKU",
                },
                {
                    "label": "Qwen 3.0 30B | vs GPT-OSS-120B-Th | UT IND ShareGPT",
                    "glob": "34_archived_qwen30_ut_ind_sharegpt_vs_oss120__*",
                    "trained_task": "IND (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Qwen 3.0 30B | vs GPT-OSS-120B-Th | AT PW ShareGPT",
                    "glob": "35_archived_qwen30_at_pw_sharegpt_vs_oss120__*",
                    "trained_task": "PW (AT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Qwen 3.0 30B | vs GPT-OSS-120B-Th | AT IND ShareGPT",
                    "glob": "36_archived_qwen30_at_ind_sharegpt_vs_oss120__*",
                    "trained_task": "IND (AT)",
                    "trained_dataset": "ShareGPT",
                },
            ],
        },
        {
            "group": "GPT-OSS 20B vs Llama 3.1 8B",
            "rows": [
                {
                    "label": "GPT-OSS 20B | vs Llama 3.1 8B | UT PW ShareGPT",
                    "glob": "01_sft_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "GPT-OSS 20B | vs Llama 3.1 8B | UT PW WikiSum",
                    "glob": "11_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_wikisum__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "WikiSum",
                },
                {
                    "label": "GPT-OSS 20B | vs Llama 3.1 8B | UT PW BigCode",
                    "glob": "12_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_bigcodebench__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "BigCode",
                },
                {
                    "label": "GPT-OSS 20B | vs Llama 3.1 8B | UT PW PKU",
                    "glob": "13_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_pku__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "PKU",
                },
                {
                    "label": "GPT-OSS 20B | vs Llama 3.1 8B | UT IND ShareGPT",
                    "glob": "14_sft_ut_ind_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "trained_task": "IND (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "GPT-OSS 20B | vs Llama 3.1 8B | AT PW ShareGPT",
                    "glob": "15_sft_at_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "trained_task": "PW (AT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "GPT-OSS 20B | vs Llama 3.1 8B | AT IND ShareGPT",
                    "glob": "16_sft_at_ind_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "trained_task": "IND (AT)",
                    "trained_dataset": "ShareGPT",
                },
            ],
        },
        {
            "group": "GPT-OSS 20B vs Qwen3-30B",
            "rows": [
                {
                    "label": "GPT-OSS 20B | vs Qwen3-30B | UT PW ShareGPT",
                    "glob": "01_sft_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "GPT-OSS 20B | vs Qwen3-30B | UT PW WikiSum",
                    "glob": "11_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_wikisum__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "WikiSum",
                },
                {
                    "label": "GPT-OSS 20B | vs Qwen3-30B | UT PW BigCode",
                    "glob": "12_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_bigcodebench__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "BigCode",
                },
                {
                    "label": "GPT-OSS 20B | vs Qwen3-30B | UT PW PKU",
                    "glob": "13_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_pku__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "PKU",
                },
                {
                    "label": "GPT-OSS 20B | vs Qwen3-30B | UT IND ShareGPT",
                    "glob": "14_sft_ut_ind_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "trained_task": "IND (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "GPT-OSS 20B | vs Qwen3-30B | AT PW ShareGPT",
                    "glob": "15_sft_at_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "trained_task": "PW (AT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "GPT-OSS 20B | vs Qwen3-30B | AT IND ShareGPT",
                    "glob": "16_sft_at_ind_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "trained_task": "IND (AT)",
                    "trained_dataset": "ShareGPT",
                },
            ],
        },
    ]


def adversarial_specs() -> list[dict]:
    return [
        {
            "group": "Adversarial / Train-As-Other",
            "rows": [
                {
                    "label": "Qwen 3.0 30B | train-as GPT-OSS-120B-Th | UT PW ShareGPT",
                    "glob": "22_archived_qwen30_ut_pw_sharegpt_train_as_oss120_vs_qwen30__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "Qwen 3.0 30B | train-as GPT-OSS-120B-Th | UT IND ShareGPT",
                    "glob": "23_archived_qwen30_ut_ind_sharegpt_train_as_oss120_vs_qwen30__*",
                    "trained_task": "IND (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "GPT-OSS 20B | train-as Qwen3-30B | UT PW ShareGPT",
                    "glob": "24_tinker_oss20_ut_pw_sharegpt_train_as_qwen30__*",
                    "trained_task": "PW (UT)",
                    "trained_dataset": "ShareGPT",
                },
                {
                    "label": "GPT-OSS 20B | train-as Qwen3-30B | UT IND ShareGPT",
                    "glob": "25_tinker_oss20_ut_ind_sharegpt_train_as_qwen30__*",
                    "trained_task": "IND (UT)",
                    "trained_dataset": "ShareGPT",
                },
            ],
        }
    ]
def load_metrics_by_step(run_dir: Path) -> dict[int, dict]:
    merged: dict[int, dict] = defaultdict(dict)
    with (run_dir / "metrics" / "metrics.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = int(row.pop("step"))
            merged[step].update(row)
    return dict(merged)


def get_first_last(run_dir: Path) -> tuple[dict, dict]:
    records = load_metrics_by_step(run_dir)
    return records[min(records)], records[max(records)]


def pick_metric(row: dict, candidates: list[str]) -> float | None:
    for key in candidates:
        if key in row:
            return float(row[key])
    return None


def task_metric_candidates(trained_task: str, eval_task: str) -> list[str]:
    base = {
        "PW (UT)": {
            "PW (UT)": ["val/accuracy"],
            "IND (UT)": [
                "benchmark/xeval_format_ind_full/accuracy",
                "benchmark/xeval_task_ut_ind_full/accuracy",
                "benchmark/xeval_format_ind/accuracy",
                "benchmark/xeval_task_ut_ind/accuracy",
            ],
            "PW (AT)": [
                "benchmark/xeval_tag_at_pw_full/accuracy",
                "benchmark/xeval_task_at_pw_full/accuracy",
                "benchmark/xeval_tag_at_pw/accuracy",
                "benchmark/xeval_task_at_pw/accuracy",
            ],
            "IND (AT)": [
                "benchmark/xeval_tag_at_ind_full/accuracy",
                "benchmark/xeval_task_at_ind_full/accuracy",
                "benchmark/xeval_tag_at_ind/accuracy",
                "benchmark/xeval_task_at_ind/accuracy",
            ],
            "PW Pref": [
                "benchmark/xeval_task_pref_pw_full/accuracy",
                "benchmark/xeval_task_pref_pw/accuracy",
            ],
            "IND Pref": [
                "benchmark/xeval_task_pref_ind_full/accuracy",
                "benchmark/xeval_task_pref_ind/accuracy",
            ],
        },
        "IND (UT)": {
            "PW (UT)": [
                "benchmark/xeval_format_pw_full/accuracy",
                "benchmark/xeval_task_ut_pw_full/accuracy",
                "benchmark/xeval_format_pw/accuracy",
                "benchmark/xeval_task_ut_pw/accuracy",
            ],
            "IND (UT)": ["val/accuracy"],
            "PW (AT)": [
                "benchmark/xeval_tag_at_pw_full/accuracy",
                "benchmark/xeval_task_at_pw_full/accuracy",
                "benchmark/xeval_tag_at_pw/accuracy",
                "benchmark/xeval_task_at_pw/accuracy",
            ],
            "IND (AT)": [
                "benchmark/xeval_tag_at_ind_full/accuracy",
                "benchmark/xeval_task_at_ind_full/accuracy",
                "benchmark/xeval_tag_at_ind/accuracy",
                "benchmark/xeval_task_at_ind/accuracy",
            ],
            "PW Pref": [
                "benchmark/xeval_task_pref_pw_full/accuracy",
                "benchmark/xeval_task_pref_pw/accuracy",
            ],
            "IND Pref": [
                "benchmark/xeval_task_pref_ind_full/accuracy",
                "benchmark/xeval_task_pref_ind/accuracy",
            ],
        },
        "PW (AT)": {
            "PW (UT)": [
                "benchmark/xeval_tag_ut_pw_full/accuracy",
                "benchmark/xeval_task_ut_pw_full/accuracy",
                "benchmark/xeval_tag_ut_pw/accuracy",
                "benchmark/xeval_task_ut_pw/accuracy",
            ],
            "IND (UT)": [
                "benchmark/xeval_tag_ut_ind_full/accuracy",
                "benchmark/xeval_task_ut_ind_full/accuracy",
                "benchmark/xeval_tag_ut_ind/accuracy",
                "benchmark/xeval_task_ut_ind/accuracy",
            ],
            "PW (AT)": ["val/accuracy"],
            "IND (AT)": [
                "benchmark/xeval_format_ind_full/accuracy",
                "benchmark/xeval_task_at_ind_full/accuracy",
                "benchmark/xeval_format_ind/accuracy",
                "benchmark/xeval_task_at_ind/accuracy",
            ],
            "PW Pref": [
                "benchmark/xeval_task_pref_pw_full/accuracy",
                "benchmark/xeval_task_pref_pw/accuracy",
            ],
            "IND Pref": [
                "benchmark/xeval_task_pref_ind_full/accuracy",
                "benchmark/xeval_task_pref_ind/accuracy",
            ],
        },
        "IND (AT)": {
            "PW (UT)": [
                "benchmark/xeval_tag_ut_pw_full/accuracy",
                "benchmark/xeval_task_ut_pw_full/accuracy",
                "benchmark/xeval_tag_ut_pw/accuracy",
                "benchmark/xeval_task_ut_pw/accuracy",
            ],
            "IND (UT)": [
                "benchmark/xeval_tag_ut_ind_full/accuracy",
                "benchmark/xeval_task_ut_ind_full/accuracy",
                "benchmark/xeval_tag_ut_ind/accuracy",
                "benchmark/xeval_task_ut_ind/accuracy",
            ],
            "PW (AT)": [
                "benchmark/xeval_format_pw_full/accuracy",
                "benchmark/xeval_task_at_pw_full/accuracy",
                "benchmark/xeval_format_pw/accuracy",
                "benchmark/xeval_task_at_pw/accuracy",
            ],
            "IND (AT)": ["val/accuracy"],
            "PW Pref": [
                "benchmark/xeval_task_pref_pw_full/accuracy",
                "benchmark/xeval_task_pref_pw/accuracy",
            ],
            "IND Pref": [
                "benchmark/xeval_task_pref_ind_full/accuracy",
                "benchmark/xeval_task_pref_ind/accuracy",
            ],
        },
    }
    return base[trained_task][eval_task]


def dataset_metric_candidates(eval_dataset: str) -> list[str]:
    key = DATASET_KEYS[eval_dataset]
    return [
        f"benchmark/xeval_dataset_{key}_full/accuracy",
        f"benchmark/xeval_dataset_{key}/accuracy",
    ]


def delta_or_nan(pre_row: dict, post_row: dict, candidates: list[str]) -> float:
    pre = pick_metric(pre_row, candidates)
    post = pick_metric(post_row, candidates)
    if pre is None or post is None:
        return np.nan
    return post - pre


def build_row_values(spec: dict, run_dir: Path) -> list[float]:
    pre_row, post_row = get_first_last(run_dir)
    row: list[float] = []
    for eval_task in TASK_EVAL_ORDER:
        row.append(
            delta_or_nan(pre_row, post_row, task_metric_candidates(spec["trained_task"], eval_task))
        )
    for pref_task in PREF_EVAL_ORDER:
        row.append(
            delta_or_nan(pre_row, post_row, task_metric_candidates(spec["trained_task"], pref_task))
        )
    dataset_vals: list[float] = []
    for eval_dataset in DATASET_EVAL_ORDER:
        if eval_dataset == spec["trained_dataset"]:
            value = np.nan
        else:
            value = delta_or_nan(pre_row, post_row, dataset_metric_candidates(eval_dataset))
        dataset_vals.append(value)
        row.append(value)

    rec_vals = row[:4]
    mean_rec = float(np.nanmean(rec_vals)) if not np.all(np.isnan(rec_vals)) else np.nan
    mean_dataset = float(np.nanmean(dataset_vals)) if not np.all(np.isnan(dataset_vals)) else np.nan
    row.extend([mean_rec, mean_dataset])
    return row


def build_rows(
    group_specs: list[dict],
    resolver: TransferRunResolver,
    *,
    allow_missing: bool,
) -> list[dict]:
    rows: list[dict] = []
    for group in group_specs:
        for spec in group["rows"]:
            resolved = resolver.resolve(spec["glob"], required=not allow_missing)
            if resolved is None:
                continue
            rows.append(
                {
                    "group": group["group"],
                    "label": spec["label"],
                    "glob": spec["glob"],
                    "run_dir": resolved.run_dir,
                    "values": build_row_values(spec, resolved.run_dir),
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"No rows available for source={resolver.source_name}. "
            "If runs are still in progress, retry later or use --allow-missing."
        )
    return rows


def rows_to_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str], list[float]]:
    separators: list[float] = []
    row_labels: list[str] = []
    matrix_rows: list[list[float]] = []
    prev_group: str | None = None
    for row in rows:
        if prev_group is not None and row["group"] != prev_group:
            separators.append(len(row_labels) - 0.5)
        row_labels.append(row["label"])
        matrix_rows.append(row["values"])
        prev_group = row["group"]
    return np.array(matrix_rows, dtype=float), row_labels, separators


COLUMN_LABELS = [
    "PW (UT)\nRec",
    "IND (UT)\nRec",
    "PW (AT)\nRec",
    "IND (AT)\nRec",
    "PW\nPref",
    "IND\nPref",
    "ShareGPT",
    "WikiSum",
    "BigCode",
    "PKU",
    "Mean\nRec",
    "Mean\nDataset",
]


def _draw_heatmap(
    ax,
    matrix: np.ndarray,
    row_labels: list[str],
    separators: list[float],
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
):
    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(COLUMN_LABELS)))
    ax.set_xticklabels(COLUMN_LABELS, rotation=0, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for sep in separators:
        ax.axhline(sep, color="white", linewidth=2.0)

    ax.axvline(3.5, color="white", linewidth=2.0)
    ax.axvline(5.5, color="white", linewidth=2.0)
    ax.axvline(9.5, color="white", linewidth=2.0)

    group_centers = {
        "Recognition": 1.5,
        "Preference": 4.5,
        "Dataset": 7.5,
        "Mean": 10.5,
    }
    for label, center in group_centers.items():
        ax.text(center, -1.15, label, ha="center", va="bottom", fontsize=12, fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            text_color = "white" if abs(value) >= 0.32 else "black"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8, color=text_color)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=30)
    return im, colorbar_label


def _write_run_manifest(output_path: Path, payload: dict) -> None:
    manifest_path = output_path.with_suffix(".sources.json")
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")


def plot_source_heatmap(
    group_specs: list[dict],
    resolver: TransferRunResolver,
    *,
    allow_missing: bool,
    output_path: Path,
    title: str,
) -> Path:
    rows = build_rows(group_specs, resolver, allow_missing=allow_missing)
    matrix, row_labels, separators = rows_to_matrix(rows)
    fig_h = max(8, 0.38 * len(row_labels) + 2.8)
    fig, ax = plt.subplots(figsize=(13.5, fig_h), constrained_layout=True)
    im, colorbar_label = _draw_heatmap(
        ax,
        matrix,
        row_labels,
        separators,
        title=title,
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        colorbar_label="Accuracy Δ (post - pre)",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label(colorbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _write_run_manifest(
        output_path,
        {
            "resolver": resolver.describe(),
            "rows": [
                {
                    "group": row["group"],
                    "label": row["label"],
                    "glob": row["glob"],
                    "run_dir": str(row["run_dir"]),
                }
                for row in rows
            ],
        },
    )
    return output_path


def plot_compare_heatmap(
    group_specs: list[dict],
    old_resolver: TransferRunResolver,
    clean_resolver: TransferRunResolver,
    *,
    allow_missing: bool,
    output_path: Path,
    title: str,
) -> Path:
    old_rows = build_rows(group_specs, old_resolver, allow_missing=allow_missing)
    clean_rows = build_rows(group_specs, clean_resolver, allow_missing=allow_missing)
    clean_by_label = {row["label"]: row for row in clean_rows}

    paired_rows: list[tuple[dict, dict]] = []
    for old_row in old_rows:
        clean_row = clean_by_label.get(old_row["label"])
        if clean_row is None:
            if allow_missing:
                continue
            raise FileNotFoundError(f"Missing clean row for label: {old_row['label']}")
        paired_rows.append((old_row, clean_row))
    if not paired_rows:
        raise FileNotFoundError("No overlapping old/clean rows available for comparison.")

    compare_rows = [
        {"group": old_row["group"], "label": old_row["label"], "values": old_row["values"]}
        for old_row, _ in paired_rows
    ]
    matrix_old, row_labels, separators = rows_to_matrix(compare_rows)
    matrix_clean = np.array([clean_row["values"] for _, clean_row in paired_rows], dtype=float)
    matrix_diff = matrix_clean - matrix_old

    fig_h = max(8, 0.38 * len(row_labels) + 2.8)
    fig, axes = plt.subplots(1, 3, figsize=(28, fig_h), constrained_layout=True)
    old_im, _ = _draw_heatmap(
        axes[0],
        matrix_old,
        row_labels,
        separators,
        title="Old (external)",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        colorbar_label="Accuracy Δ (post - pre)",
    )
    clean_im, _ = _draw_heatmap(
        axes[1],
        matrix_clean,
        row_labels,
        separators,
        title="Clean reruns (local manifests)",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        colorbar_label="Accuracy Δ (post - pre)",
    )
    diff_max = max(0.05, float(np.nanmax(np.abs(matrix_diff))))
    diff_im, _ = _draw_heatmap(
        axes[2],
        matrix_diff,
        row_labels,
        separators,
        title="Clean - old",
        cmap="RdBu_r",
        vmin=-diff_max,
        vmax=diff_max,
        colorbar_label="Δ difference",
    )
    fig.colorbar(old_im, ax=axes[:2], shrink=0.78, label="Accuracy Δ (post - pre)")
    fig.colorbar(diff_im, ax=axes[2], shrink=0.78, label="Δ difference (clean - old)")
    fig.suptitle(title, fontsize=18, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _write_run_manifest(
        output_path,
        {
            "old_resolver": old_resolver.describe(),
            "clean_resolver": clean_resolver.describe(),
            "rows": [
                {
                    "group": old_row["group"],
                    "label": old_row["label"],
                    "glob": old_row["glob"],
                    "old_run_dir": str(old_row["run_dir"]),
                    "clean_run_dir": str(clean_row["run_dir"]),
                }
                for old_row, clean_row in paired_rows
            ],
        },
    )
    return output_path


def _selected_specs(plot_set: str) -> list[tuple[str, list[dict], str]]:
    items: list[tuple[str, list[dict], str]] = []
    if plot_set in {"standard", "all"}:
        items.append(
            (
                "standard",
                standard_specs(),
                "Training Transfer Heatmap\n"
                "One row per training run; explicit source selection avoids mixing old, clean, "
                "and local scratch results",
            )
        )
    if plot_set in {"adversarial", "all"}:
        items.append(
            (
                "adversarial",
                adversarial_specs(),
                "Adversarial / Train-As-Other Heatmap\n"
                "One row per completed adversarial run; explicit source selection avoids "
                "mixing run families",
            )
        )
    return items


def _build_clean_resolver(manifest_args: list[str]) -> TransferRunResolver:
    manifest_paths = [Path(path) for path in manifest_args]
    return TransferRunResolver.clean(manifest_paths=manifest_paths or None)


def main() -> None:
    args = parse_args()
    selected = _selected_specs(args.plot_set)

    if args.source == "old":
        resolver = TransferRunResolver.old()
        for plot_key, specs, title in selected:
            output_path = args.output_dir / "old" / f"transfer_heatmap_{plot_key}.png"
            print(
                plot_source_heatmap(
                    specs,
                    resolver,
                    allow_missing=args.allow_missing,
                    output_path=output_path,
                    title=f"{title}\nSource: old external results",
                )
            )
        return

    if args.source == "clean":
        resolver = _build_clean_resolver(args.clean_manifest)
        for plot_key, specs, title in selected:
            output_path = args.output_dir / "clean" / f"transfer_heatmap_{plot_key}.png"
            print(
                plot_source_heatmap(
                    specs,
                    resolver,
                    allow_missing=args.allow_missing,
                    output_path=output_path,
                    title=f"{title}\nSource: clean local reruns",
                )
            )
        return

    old_resolver = TransferRunResolver.old()
    clean_resolver = _build_clean_resolver(args.clean_manifest)
    for plot_key, specs, title in selected:
        output_path = args.output_dir / "compare" / f"transfer_heatmap_{plot_key}_old_vs_clean.png"
        print(
            plot_compare_heatmap(
                specs,
                old_resolver,
                clean_resolver,
                allow_missing=args.allow_missing,
                output_path=output_path,
                title=f"{title}\nComparison: old external vs clean local reruns",
            )
        )


if __name__ == "__main__":
    main()
