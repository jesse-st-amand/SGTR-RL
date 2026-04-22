#!/usr/bin/env python3
"""Plot no-averaging transfer heatmaps from completed SGTR runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS_DIR = ROOT / "results"
EXTERNAL_RESULTS_DIR = ROOT.parent / "self-rec-research" / "_external" / "SGTR-RL" / "results"
RESULTS_DIR = (
    LOCAL_RESULTS_DIR
    if (LOCAL_RESULTS_DIR / "01_sft_pw_vs_qwen3_30b_tinker_small__20260324_130227").exists()
    else EXTERNAL_RESULTS_DIR
)
OUTPUT_DIR = ROOT / "results" / "transfer_plots"

STANDARD_OUTPUT = OUTPUT_DIR / "corrected_transfer_heatmap_standard.png"
ADVERSARIAL_OUTPUT = OUTPUT_DIR / "corrected_transfer_heatmap_adversarial.png"

DATASET_KEYS = {
    "ShareGPT": "sharegpt",
    "WikiSum": "wikisum",
    "BigCode": "bigcodebench",
    "PKU": "pku",
}

TASK_EVAL_ORDER = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)"]
PREF_EVAL_ORDER = ["PW Pref", "IND Pref"]
DATASET_EVAL_ORDER = ["ShareGPT", "WikiSum", "BigCode", "PKU"]


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


def resolve_completed(glob_pattern: str) -> Path:
    matches = sorted(RESULTS_DIR.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"No matches for {glob_pattern}")
    completed: list[Path] = []
    for match in matches:
        status_path = match / "status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text()).get("status")
        except Exception:
            status = None
        if status == "completed":
            completed.append(match)
    if not completed:
        raise FileNotFoundError(f"No completed runs for {glob_pattern}")
    return completed[-1]


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


def build_matrix(group_specs: list[dict]) -> tuple[np.ndarray, list[str], list[float]]:
    row_labels: list[str] = []
    rows: list[list[float]] = []
    separators: list[float] = []

    for group_index, group in enumerate(group_specs):
        if group_index > 0:
            separators.append(len(row_labels) - 0.5)
        for spec in group["rows"]:
            run_dir = resolve_completed(spec["glob"])
            pre_row, post_row = get_first_last(run_dir)

            row: list[float] = []
            for eval_task in TASK_EVAL_ORDER:
                row.append(
                    delta_or_nan(
                        pre_row, post_row, task_metric_candidates(spec["trained_task"], eval_task)
                    )
                )
            for pref_task in PREF_EVAL_ORDER:
                row.append(
                    delta_or_nan(
                        pre_row, post_row, task_metric_candidates(spec["trained_task"], pref_task)
                    )
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
            mean_dataset = (
                float(np.nanmean(dataset_vals)) if not np.all(np.isnan(dataset_vals)) else np.nan
            )
            row.extend([mean_rec, mean_dataset])

            row_labels.append(spec["label"])
            rows.append(row)

    return np.array(rows, dtype=float), row_labels, separators


def plot_heatmap(group_specs: list[dict], output_path: Path, title: str) -> Path:
    matrix, row_labels, separators = build_matrix(group_specs)
    col_labels = [
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

    fig_h = max(8, 0.38 * len(row_labels) + 2.8)
    fig, ax = plt.subplots(figsize=(13.5, fig_h), constrained_layout=True)

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=0, fontsize=10)
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

    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Accuracy Δ (post - pre)")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=30)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    standard_title = (
        "Training Transfer Heatmap (No Opponent Averaging, Standard Completed Runs)\n"
        "One row per training run; AE / rank columns omitted until we lock the exact "
        "definition"
    )
    adversarial_title = (
        "Adversarial / Train-As-Other Heatmap\n"
        "One row per completed adversarial run; AE / rank columns omitted until we lock "
        "the exact definition"
    )
    print(
        plot_heatmap(
            standard_specs(),
            STANDARD_OUTPUT,
            standard_title,
        )
    )
    print(
        plot_heatmap(
            adversarial_specs(),
            ADVERSARIAL_OUTPUT,
            adversarial_title,
        )
    )


if __name__ == "__main__":
    main()
