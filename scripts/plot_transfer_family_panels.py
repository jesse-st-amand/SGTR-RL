#!/usr/bin/env python3
"""Plot transfer panels without averaging across opponent families."""

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

DATASET_ORDER = ["WikiSum", "BigCode", "PKU", "ShareGPT"]
TASK_TRAIN_ORDER = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)"]
TASK_EVAL_ORDER = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)", "PW Pref", "IND Pref"]

DATASET_KEYS = {
    "ShareGPT": "sharegpt",
    "WikiSum": "wikisum",
    "BigCode": "bigcodebench",
    "PKU": "pku",
}

SELF_GROUPS = {
    "llama_self": {
        "title": "Llama 3.1 8B Transfer (No Opponent Averaging)",
        "subtitle": "Completed standard families only; each color is one opponent family",
        "dataset_output": OUTPUT_DIR / "corrected_dataset_transfer_llama_self_by_family.png",
        "task_output": OUTPUT_DIR / "corrected_task_transfer_llama_self_by_family.png",
        "families": [
            {
                "label": "vs qwen-2.5-7b",
                "color": "#2b6cb0",
                "dataset_runs": {
                    "ShareGPT": "11_archived_ll8b_ut_pw_sharegpt_vs_qwen25__*",
                    "WikiSum": "12_archived_ll8b_ut_pw_wikisum_vs_qwen25__*",
                    "BigCode": "13_archived_ll8b_ut_pw_bigcodebench_vs_qwen25__*",
                    "PKU": "14_archived_ll8b_ut_pw_pku_vs_qwen25__*",
                },
                "task_runs": {
                    "PW (UT)": "11_archived_ll8b_ut_pw_sharegpt_vs_qwen25__*",
                    "IND (UT)": "15_archived_ll8b_ut_ind_sharegpt_vs_qwen25__*",
                    "PW (AT)": "16_archived_ll8b_at_pw_sharegpt_vs_qwen25__*",
                    "IND (AT)": "17_archived_ll8b_at_ind_sharegpt_vs_qwen25__*",
                },
            },
            {
                "label": "vs qwen3-30b",
                "color": "#805ad5",
                "dataset_runs": {
                    "ShareGPT": "01_sft_pw_vs_qwen3_30b_tinker_small__*",
                    "WikiSum": "11_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_wikisum__*",
                    "BigCode": "12_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_bigcodebench__*",
                    "PKU": "13_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_pku__*",
                },
                "task_runs": {
                    "PW (UT)": "01_sft_pw_vs_qwen3_30b_tinker_small__*",
                    "IND (UT)": "14_sft_ut_ind_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "PW (AT)": "15_sft_at_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "IND (AT)": "16_sft_at_ind_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt__*",
                },
            },
        ],
    },
    "qwen_self": {
        "title": "Qwen 3.0 30B Transfer (No Opponent Averaging)",
        "subtitle": "Completed standard families only; each color is one opponent family",
        "dataset_output": OUTPUT_DIR / "corrected_dataset_transfer_qwen_self_by_family.png",
        "task_output": OUTPUT_DIR / "corrected_task_transfer_qwen_self_by_family.png",
        "families": [
            {
                "label": "vs llama-3.1-8b",
                "color": "#dd6b20",
                "dataset_runs": {
                    "ShareGPT": "01_sft_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small__*",
                    "WikiSum": "11_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_wikisum__*",
                    "BigCode": "12_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_bigcodebench__*",
                    "PKU": "13_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_pku__*",
                },
                "task_runs": {
                    "PW (UT)": "01_sft_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small__*",
                    "IND (UT)": "14_sft_ut_ind_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "PW (AT)": "15_sft_at_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "IND (AT)": "16_sft_at_ind_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                },
            },
            {
                "label": "vs gpt-oss-120b-thinking",
                "color": "#c53030",
                "dataset_runs": {
                    "ShareGPT": "21_archived_qwen30_ut_pw_sharegpt_vs_oss120__*",
                    "WikiSum": "31_archived_qwen30_ut_pw_wikisum_vs_oss120__*",
                    "BigCode": "32_archived_qwen30_ut_pw_bigcodebench_vs_oss120__*",
                    "PKU": "33_archived_qwen30_ut_pw_pku_vs_oss120__*",
                },
                "task_runs": {
                    "PW (UT)": "21_archived_qwen30_ut_pw_sharegpt_vs_oss120__*",
                    "IND (UT)": "34_archived_qwen30_ut_ind_sharegpt_vs_oss120__*",
                    "PW (AT)": "35_archived_qwen30_at_pw_sharegpt_vs_oss120__*",
                    "IND (AT)": "36_archived_qwen30_at_ind_sharegpt_vs_oss120__*",
                },
            },
        ],
    },
    "oss20_self": {
        "title": "GPT-OSS 20B Transfer (No Opponent Averaging)",
        "subtitle": "Completed standard families only; each color is one opponent family",
        "dataset_output": OUTPUT_DIR / "corrected_dataset_transfer_oss20_self_by_family.png",
        "task_output": OUTPUT_DIR / "corrected_task_transfer_oss20_self_by_family.png",
        "families": [
            {
                "label": "vs llama-3.1-8b",
                "color": "#1a936f",
                "dataset_runs": {
                    "ShareGPT": "01_sft_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small__*",
                    "WikiSum": "11_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_wikisum__*",
                    "BigCode": "12_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_bigcodebench__*",
                    "PKU": "13_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_pku__*",
                },
                "task_runs": {
                    "PW (UT)": "01_sft_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small__*",
                    "IND (UT)": "14_sft_ut_ind_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "PW (AT)": "15_sft_at_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                    "IND (AT)": "16_sft_at_ind_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt__*",
                },
            },
            {
                "label": "vs qwen3-30b",
                "color": "#7f8c1f",
                "dataset_runs": {
                    "ShareGPT": "01_sft_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small__*",
                    "WikiSum": "11_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_wikisum__*",
                    "BigCode": "12_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_bigcodebench__*",
                    "PKU": "13_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_pku__*",
                },
                "task_runs": {
                    "PW (UT)": "01_sft_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small__*",
                    "IND (UT)": "14_sft_ut_ind_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "PW (AT)": "15_sft_at_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt__*",
                    "IND (AT)": "16_sft_at_ind_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt__*",
                },
            },
        ],
    },
}


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


def pick_metric(row: dict, candidates: list[str]) -> float:
    for key in candidates:
        if key in row:
            return float(row[key])
    raise KeyError(f"None of {candidates} found")


def dataset_metric_candidates(dataset_label: str) -> list[str]:
    key = DATASET_KEYS[dataset_label]
    return [
        f"benchmark/xeval_dataset_{key}_full/accuracy",
        f"benchmark/xeval_dataset_{key}/accuracy",
    ]


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


def value_pair_for_dataset(family: dict, trained_on: str, eval_dataset: str) -> tuple[float, float]:
    run_dir = resolve_completed(family["dataset_runs"][trained_on])
    pre_row, post_row = get_first_last(run_dir)
    if eval_dataset == trained_on:
        return float(pre_row["val/accuracy"]), float(post_row["val/accuracy"])
    candidates = dataset_metric_candidates(eval_dataset)
    return pick_metric(pre_row, candidates), pick_metric(post_row, candidates)


def value_pair_for_task(family: dict, trained_on: str, eval_task: str) -> tuple[float, float]:
    run_dir = resolve_completed(family["task_runs"][trained_on])
    pre_row, post_row = get_first_last(run_dir)
    candidates = task_metric_candidates(trained_on, eval_task)
    return pick_metric(pre_row, candidates), pick_metric(post_row, candidates)


def add_common_legend(fig, families: list[dict]) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#8a8a8a",
            markersize=6,
            label="Pre-training",
        ),
    ]
    for family in families:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=family["color"],
                markerfacecolor=family["color"],
                lw=1.6,
                markersize=7,
                label=family["label"],
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=max(2, len(families) + 1),
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )


def draw_arrow(ax, x: float, pre: float, post: float, color: str) -> None:
    ax.plot(x, pre, "o", color="#8a8a8a", markersize=6, zorder=3)
    ax.plot(x, post, "o", color=color, markersize=7, zorder=4)
    delta = post - pre
    arrow_color = "#2f855a" if delta >= 0 else "#c53030"
    ax.annotate(
        "",
        xy=(x, post),
        xytext=(x, pre),
        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.6),
        zorder=2,
    )
    off = 0.02 if delta >= 0 else -0.04
    va = "bottom" if delta >= 0 else "top"
    delta_color = arrow_color
    ax.text(x + 0.02, post + off, f"{delta:+.2f}", fontsize=7, color=delta_color, va=va)


def style_dataset_background(ax, trained_on: str) -> None:
    trained_idx = DATASET_ORDER.index(trained_on)
    ax.axvspan(trained_idx - 0.48, trained_idx + 0.48, color="#f5e8b3", alpha=0.35, zorder=0)
    for idx in range(len(DATASET_ORDER)):
        if idx % 2 == 1:
            ax.axvspan(
                idx - 0.48,
                idx + 0.48,
                facecolor="none",
                hatch="//",
                edgecolor="#d0d0d0",
                linewidth=0.0,
                zorder=0,
            )


def style_task_background(ax, trained_on: str) -> None:
    for idx in range(len(TASK_EVAL_ORDER)):
        if idx % 2 == 1:
            ax.axvspan(
                idx - 0.48,
                idx + 0.48,
                facecolor="none",
                hatch="//",
                edgecolor="#d0d0d0",
                linewidth=0.0,
                zorder=0,
            )
    trained_idx = TASK_EVAL_ORDER.index(trained_on)
    ax.axvspan(trained_idx - 0.48, trained_idx + 0.48, color="#f5e8b3", alpha=0.35, zorder=0)


def plot_dataset_panels(group_key: str) -> Path:
    group = SELF_GROUPS[group_key]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True, constrained_layout=True)
    axes = axes.flatten()
    offsets = np.linspace(-0.09, 0.09, len(group["families"]))

    for ax, trained_on in zip(axes, DATASET_ORDER):
        style_dataset_background(ax, trained_on)
        xs = np.arange(len(DATASET_ORDER))
        for idx, eval_dataset in enumerate(DATASET_ORDER):
            for offset, family in zip(offsets, group["families"]):
                pre, post = value_pair_for_dataset(family, trained_on, eval_dataset)
                draw_arrow(ax, idx + offset, pre, post, family["color"])

        ax.set_title(f"Trained on: {trained_on}", fontsize=13, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(DATASET_ORDER, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Accuracy")
    axes[2].set_ylabel("Accuracy")
    add_common_legend(fig, group["families"])
    fig.suptitle(f"{group['title']}\n{group['subtitle']}", fontsize=16, fontweight="bold")
    group["dataset_output"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(group["dataset_output"], dpi=200, bbox_inches="tight")
    plt.close(fig)
    return group["dataset_output"]


def plot_task_panels(group_key: str) -> Path:
    group = SELF_GROUPS[group_key]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True, constrained_layout=True)
    axes = axes.flatten()
    offsets = np.linspace(-0.09, 0.09, len(group["families"]))

    for ax, trained_on in zip(axes, TASK_TRAIN_ORDER):
        style_task_background(ax, trained_on)
        xs = np.arange(len(TASK_EVAL_ORDER))
        for idx, eval_task in enumerate(TASK_EVAL_ORDER):
            for offset, family in zip(offsets, group["families"]):
                pre, post = value_pair_for_task(family, trained_on, eval_task)
                draw_arrow(ax, idx + offset, pre, post, family["color"])

        ax.set_title(f"Trained on: {trained_on}", fontsize=13, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(TASK_EVAL_ORDER, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Accuracy")
    axes[2].set_ylabel("Accuracy")
    add_common_legend(fig, group["families"])
    fig.suptitle(f"{group['title']}\n{group['subtitle']}", fontsize=16, fontweight="bold")
    group["task_output"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(group["task_output"], dpi=200, bbox_inches="tight")
    plt.close(fig)
    return group["task_output"]


def main() -> None:
    for group_key in SELF_GROUPS:
        print(plot_dataset_panels(group_key))
        print(plot_task_panels(group_key))


if __name__ == "__main__":
    main()
