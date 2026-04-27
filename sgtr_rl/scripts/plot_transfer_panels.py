#!/usr/bin/env python3
"""Plot corrected multi-model transfer panels using only completed standard families."""

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

DATASET_ORDER = ["WikiSum", "BigCode", "PKU", "ShareGPT"]
TASK_TRAIN_ORDER = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)"]
TASK_EVAL_ORDER = ["PW (UT)", "IND (UT)", "PW (AT)", "IND (AT)", "PW Pref", "IND Pref"]

DATASET_KEYS = {
    "ShareGPT": "sharegpt",
    "WikiSum": "wikisum",
    "BigCode": "bigcodebench",
    "PKU": "pku",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot transfer panels from explicit run sources.")
    parser.add_argument("--source", choices=["old", "clean"], default="old")
    parser.add_argument("--clean-manifest", action="append", default=[])
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()

MODEL_GROUPS = {
    "Llama 3.1 8B": {
        "color": "#2b6cb0",
        "legend": "Llama 3.1 8B (2 complete families)",
        "families": [
            {
                "name": "vs qwen-2.5-7b",
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
                "name": "vs qwen3-30b",
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
    "Qwen 3.0 30B": {
        "color": "#dd6b20",
        "legend": "Qwen 3.0 30B (2 complete families)",
        "families": [
            {
                "name": "vs llama-3.1-8b",
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
                "name": "vs gpt-oss-120b",
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
}
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


def mean_pair(values: list[tuple[float, float]]) -> tuple[float, float]:
    pre = float(np.mean([v[0] for v in values]))
    post = float(np.mean([v[1] for v in values]))
    return pre, post


def aggregate_dataset(
    resolver: TransferRunResolver,
    self_model: str,
    trained_on: str,
    eval_dataset: str,
    *,
    allow_missing: bool,
) -> tuple[float, float] | None:
    pairs: list[tuple[float, float]] = []
    for family in MODEL_GROUPS[self_model]["families"]:
        resolved = resolver.resolve(family["dataset_runs"][trained_on], required=not allow_missing)
        if resolved is None:
            continue
        run_dir = resolved.run_dir
        pre_row, post_row = get_first_last(run_dir)
        if eval_dataset == trained_on:
            pairs.append((float(pre_row["val/accuracy"]), float(post_row["val/accuracy"])))
        else:
            candidates = dataset_metric_candidates(eval_dataset)
            pairs.append((pick_metric(pre_row, candidates), pick_metric(post_row, candidates)))
    if not pairs:
        return None
    return mean_pair(pairs)


def aggregate_task(
    resolver: TransferRunResolver,
    self_model: str,
    trained_on: str,
    eval_task: str,
    *,
    allow_missing: bool,
) -> tuple[float, float] | None:
    pairs: list[tuple[float, float]] = []
    for family in MODEL_GROUPS[self_model]["families"]:
        resolved = resolver.resolve(family["task_runs"][trained_on], required=not allow_missing)
        if resolved is None:
            continue
        run_dir = resolved.run_dir
        pre_row, post_row = get_first_last(run_dir)
        candidates = task_metric_candidates(trained_on, eval_task)
        pairs.append((pick_metric(pre_row, candidates), pick_metric(post_row, candidates)))
    if not pairs:
        return None
    return mean_pair(pairs)


def add_model_legend(fig) -> None:
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
        plt.Line2D([0], [0], color="#2f855a", lw=1.8, label="Change"),
    ]
    for model_name, spec in MODEL_GROUPS.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=spec["color"],
                markersize=7,
                label=spec["legend"],
            )
        )
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08)
    )


def plot_dataset_panels(
    resolver: TransferRunResolver,
    *,
    allow_missing: bool,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True, constrained_layout=True)
    axes = axes.flatten()
    model_items = list(MODEL_GROUPS.items())
    offsets = np.linspace(-0.12, 0.12, len(model_items))

    for ax, trained_on in zip(axes, DATASET_ORDER):
        trained_idx = DATASET_ORDER.index(trained_on)
        ax.axvspan(trained_idx - 0.48, trained_idx + 0.48, color="#f5e8b3", alpha=0.35, zorder=0)

        xs = np.arange(len(DATASET_ORDER))
        for idx, eval_dataset in enumerate(DATASET_ORDER):
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

            for offset, (model_name, spec) in zip(offsets, model_items):
                pair = aggregate_dataset(
                    resolver,
                    model_name,
                    trained_on,
                    eval_dataset,
                    allow_missing=allow_missing,
                )
                if pair is None:
                    continue
                pre, post = pair
                x = idx + offset
                ax.plot(x, pre, "o", color="#8a8a8a", markersize=6, zorder=3)
                ax.plot(x, post, "o", color=spec["color"], markersize=7, zorder=4)
                ax.annotate(
                    "",
                    xy=(x, post),
                    xytext=(x, pre),
                    arrowprops=dict(arrowstyle="->", color="#2f855a", lw=1.6),
                    zorder=2,
                )
                delta = post - pre
                off = 0.02 if delta >= 0 else -0.04
                va = "bottom" if delta >= 0 else "top"
                ax.text(
                    x + 0.02,
                    post + off,
                    f"{delta:+.2f}",
                    fontsize=7,
                    color="#2f855a" if delta >= 0 else "#c53030",
                    va=va,
                )

        ax.set_title(f"Trained on: {trained_on}", fontsize=13, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(DATASET_ORDER, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Accuracy")
    axes[2].set_ylabel("Accuracy")
    add_model_legend(fig)
    fig.suptitle(
        f"Dataset Domain Transfer ({resolver.source_name} source)\n"
        "Llama 3.1 8B averaged over qwen-2.5-7b + qwen3-30b; "
        "Qwen 3.0 30B averaged over llama-3.1-8b + gpt-oss-120b; GPT-OSS 20B pending",
        fontsize=16,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def style_task_background(ax, trained_task: str) -> None:
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
    trained_idx = TASK_EVAL_ORDER.index("PW (UT)" if trained_task == "PW (UT)" else trained_task)
    ax.axvspan(trained_idx - 0.48, trained_idx + 0.48, color="#f5e8b3", alpha=0.35, zorder=0)


def plot_task_panels(
    resolver: TransferRunResolver,
    *,
    allow_missing: bool,
    output_path: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True, constrained_layout=True)
    axes = axes.flatten()
    model_items = list(MODEL_GROUPS.items())
    offsets = np.linspace(-0.12, 0.12, len(model_items))

    for ax, trained_on in zip(axes, TASK_TRAIN_ORDER):
        style_task_background(ax, trained_on)
        xs = np.arange(len(TASK_EVAL_ORDER))
        for idx, eval_task in enumerate(TASK_EVAL_ORDER):
            for offset, (model_name, spec) in zip(offsets, model_items):
                pair = aggregate_task(
                    resolver,
                    model_name,
                    trained_on,
                    eval_task,
                    allow_missing=allow_missing,
                )
                if pair is None:
                    continue
                pre, post = pair
                x = idx + offset
                ax.plot(x, pre, "o", color="#8a8a8a", markersize=6, zorder=3)
                ax.plot(x, post, "o", color=spec["color"], markersize=7, zorder=4)
                ax.annotate(
                    "",
                    xy=(x, post),
                    xytext=(x, pre),
                    arrowprops=dict(arrowstyle="->", color="#2f855a", lw=1.6),
                    zorder=2,
                )
                delta = post - pre
                off = 0.02 if delta >= 0 else -0.04
                va = "bottom" if delta >= 0 else "top"
                ax.text(
                    x + 0.02,
                    post + off,
                    f"{delta:+.2f}",
                    fontsize=7,
                    color="#2f855a" if delta >= 0 else "#c53030",
                    va=va,
                )

        ax.set_title(f"Trained on: {trained_on}", fontsize=13, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels(TASK_EVAL_ORDER, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Accuracy")
    axes[2].set_ylabel("Accuracy")
    add_model_legend(fig)
    fig.suptitle(
        f"Task Transfer ({resolver.source_name} source)\n"
        "Llama 3.1 8B averaged over qwen-2.5-7b + qwen3-30b; "
        "Qwen 3.0 30B averaged over llama-3.1-8b + gpt-oss-120b; GPT-OSS 20B pending",
        fontsize=16,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    resolver = (
        TransferRunResolver.old()
        if args.source == "old"
        else TransferRunResolver.clean(
            manifest_paths=[Path(path) for path in args.clean_manifest] or None
        )
    )
    print(
        plot_dataset_panels(
            resolver,
            allow_missing=args.allow_missing,
            output_path=args.output_dir / args.source / "transfer_dataset_panels.png",
        )
    )
    print(
        plot_task_panels(
            resolver,
            allow_missing=args.allow_missing,
            output_path=args.output_dir / args.source / "transfer_task_panels.png",
        )
    )


if __name__ == "__main__":
    main()
