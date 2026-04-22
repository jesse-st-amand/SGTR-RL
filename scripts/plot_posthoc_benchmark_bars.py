"""Plot a before/after bar chart from two posthoc benchmark eval directories.

Examples:
    python -m scripts.plot_posthoc_benchmark_bars \
        --pre-eval-dir results/.../posthoc_benchmarks/base_eval \
        --post-eval-dir results/.../posthoc_benchmarks/final_eval
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_BENCH_DISPLAY = {
    "mmlu_20": "MMLU (20-sample)",
    "mmlu_500": "MMLU (500-sample)",
    "mmlu_500_cot": "MMLU (500-sample, CoT)",
    "mmlu_500_nocot": "MMLU (500-sample, no CoT)",
    "mmlu_2000": "MMLU (2000-sample)",
    "xeval_dataset_bigcodebench": "BigCodeBench",
    "xeval_dataset_pku": "PKU SafeRLHF",
    "xeval_dataset_wikisum": "WikiSum",
    "xeval_format_ind": "Format: IND",
    "xeval_format_pw": "Format: PW",
    "xeval_tag_at_pw": "Tag: Assistant PW",
    "xeval_tag_at_ind": "Tag: Assistant IND",
    "xeval_task_pref_pw": "Task: Preference PW",
    "xeval_task_pref_ind": "Task: Preference IND",
    "xeval_source_numbered": "Source: Numbered",
    "xeval_vs_haiku_3_5_sharegpt": "Haiku-3.5",
    "xeval_vs_gpt_4o_sharegpt": "GPT-4o",
    "xeval_vs_ll_3_1_70b_sharegpt": "Llama-3.1-70B",
    "xeval_vs_opus_4_1_sharegpt": "Opus-4.1",
    "xeval_vs_qwen_2_5_7b_sharegpt": "Qwen-7B",
    "xeval_holdout_vs_ll_3_1_70b_sharegpt": "Holdout: Llama-3.1-70B",
    "xeval_pref_vs_haiku_3_5_sharegpt": "Haiku-3.5 (pref)",
    "xeval_pref_vs_gpt_4o_sharegpt": "GPT-4o (pref)",
    "xeval_pref_vs_ll_3_1_70b_sharegpt": "Llama-3.1-70B (pref)",
    "xeval_pref_vs_opus_4_1_sharegpt": "Opus-4.1 (pref)",
}


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_metric_record(eval_dir: Path) -> dict:
    metrics_path = eval_dir / "metrics" / "metrics.jsonl"
    records = []
    with open(metrics_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No metric records found in {metrics_path}")
    return records[-1]


def _extract_benchmark_accuracies(record: dict) -> dict[str, float]:
    result = {}
    for key, value in record.items():
        match = re.match(r"benchmark/(.+)/accuracy$", key)
        if match:
            result[match.group(1)] = value
    return result


def _label_for_benchmark(name: str) -> str:
    if name in _BENCH_DISPLAY:
        return _BENCH_DISPLAY[name]
    return name.replace("_", " ")


def _build_default_title(pre_eval_dir: Path, post_eval_dir: Path) -> str:
    pre_cfg = _load_json(pre_eval_dir / "eval_config.json")
    post_cfg = _load_json(post_eval_dir / "eval_config.json")
    source_run = Path(post_cfg["source_run_dir"]).name
    if pre_cfg.get("source_run_dir") != post_cfg.get("source_run_dir"):
        raise ValueError("Pre and post evals must come from the same source run")
    return f"{source_run}: base model vs final checkpoint"


def generate_plot(
    *,
    pre_eval_dir: str | Path,
    post_eval_dir: str | Path,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    pre_eval_dir = Path(pre_eval_dir)
    post_eval_dir = Path(post_eval_dir)

    pre_metrics = _extract_benchmark_accuracies(_load_metric_record(pre_eval_dir))
    post_metrics = _extract_benchmark_accuracies(_load_metric_record(post_eval_dir))

    ordered_names = []
    for name in list(pre_metrics) + list(post_metrics):
        if name not in ordered_names:
            ordered_names.append(name)
    if not ordered_names:
        raise ValueError("No benchmark accuracy metrics found in either eval dir")

    pre_values = [pre_metrics.get(name, np.nan) for name in ordered_names]
    post_values = [post_metrics.get(name, np.nan) for name in ordered_names]
    labels = [_label_for_benchmark(name) for name in ordered_names]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.5), 5.5))
    x = np.arange(len(labels))
    width = 0.36

    pre_bars = ax.bar(
        x - width / 2,
        pre_values,
        width,
        label="Base model",
        color="#94a3b8",
        edgecolor="white",
        linewidth=0.7,
    )
    post_bars = ax.bar(
        x + width / 2,
        post_values,
        width,
        label="Final checkpoint",
        color="#2563eb",
        edgecolor="white",
        linewidth=0.7,
    )

    for bars, color in ((pre_bars, "#475569"), (post_bars, "#1d4ed8")):
        for bar in bars:
            height = bar.get_height()
            if np.isnan(height):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )

    ax.axhline(0.5, color="#cbd5e1", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title or _build_default_title(pre_eval_dir, post_eval_dir))
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    if output_path is None:
        output_path = post_eval_dir / "pre_post_benchmarks.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot pre/post posthoc benchmark accuracies")
    parser.add_argument("--pre-eval-dir", required=True, help="Base-model posthoc eval directory")
    parser.add_argument(
        "--post-eval-dir",
        required=True,
        help="Final-checkpoint posthoc eval directory",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output PNG path; default is <post-eval-dir>/pre_post_benchmarks.png",
    )
    parser.add_argument("--title", default=None, help="Optional plot title override")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = generate_plot(
        pre_eval_dir=args.pre_eval_dir,
        post_eval_dir=args.post_eval_dir,
        output_path=args.output_path,
        title=args.title,
    )
    print(output_path)


if __name__ == "__main__":
    main()
