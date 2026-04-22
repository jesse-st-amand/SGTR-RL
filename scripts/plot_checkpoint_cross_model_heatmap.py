"""Plot cross-model checkpoint generalisation from posthoc benchmark eval dirs.

Examples:
    python -m scripts.plot_checkpoint_cross_model_heatmap \
        --eval-dirs results/.../posthoc_benchmarks/qwen_cross_model \
                   results/.../posthoc_benchmarks/haiku_cross_model
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_MODEL_DISPLAY = {
    "qwen-2.5-7b": "Qwen-7B",
    "qwen_2_5_7b": "Qwen-7B",
    "haiku-3.5": "Haiku-3.5",
    "haiku_3_5": "Haiku-3.5",
    "gpt-4o": "GPT-4o",
    "gpt_4o": "GPT-4o",
    "ll-3.1-70b": "Llama-70B",
    "ll_3_1_70b": "Llama-70B",
    "opus-4.1": "Opus-4.1",
    "opus_4_1": "Opus-4.1",
}

_MODEL_ORDER = [
    "qwen_2_5_7b",
    "haiku_3_5",
    "gpt_4o",
    "ll_3_1_70b",
    "opus_4_1",
]


def _canonical_slug(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


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


def _source_model_label(eval_dir: Path) -> str:
    eval_cfg = _load_json(eval_dir / "eval_config.json")
    source_run_dir = Path(eval_cfg["source_run_dir"])
    run_cfg = _load_yaml(source_run_dir / "config.yaml")
    generator_models = run_cfg.get("data", {}).get("generator_models", [])
    if not generator_models:
        return source_run_dir.name
    if len(generator_models) == 1:
        model = generator_models[0]
        return _MODEL_DISPLAY.get(model, model)
    return " + ".join(_MODEL_DISPLAY.get(model, model) for model in generator_models)


def _benchmark_slug(name: str) -> str | None:
    for pattern in (
        r"xeval_vs_(.+)_sharegpt$",
        r"xeval_holdout_vs_(.+)_sharegpt$",
    ):
        match = re.match(pattern, name)
        if match:
            return _canonical_slug(match.group(1))
    return None


def _extract_cross_model_accuracies(record: dict) -> dict[str, float]:
    result = {}
    for key, value in record.items():
        match = re.match(r"benchmark/(.+)/accuracy$", key)
        if not match:
            continue
        slug = _benchmark_slug(match.group(1))
        if slug is not None:
            result[slug] = value
    return result


def generate_plot(
    *,
    eval_dirs: list[str | Path],
    output_path: str | Path | None = None,
    title: str = "Checkpoint Cross-Model Generalisation",
) -> Path:
    eval_paths = [Path(path) for path in eval_dirs]
    if not eval_paths:
        raise ValueError("At least one eval directory is required")

    row_labels = []
    row_slug_order = []
    values_by_row = []
    column_slugs = []

    for eval_dir in eval_paths:
        label = _source_model_label(eval_dir)
        row_labels.append(label)
        eval_cfg = _load_json(eval_dir / "eval_config.json")
        run_cfg = _load_yaml(Path(eval_cfg["source_run_dir"]) / "config.yaml")
        generator_models = run_cfg.get("data", {}).get("generator_models", [])
        row_slug = _canonical_slug(generator_models[0]) if len(generator_models) == 1 else label
        row_slug_order.append(row_slug)
        metrics = _extract_cross_model_accuracies(_load_metric_record(eval_dir))
        for slug in metrics:
            if slug not in column_slugs:
                column_slugs.append(slug)
        values_by_row.append(metrics)

    def _sort_key(slug: str) -> tuple[int, str]:
        if slug in _MODEL_ORDER:
            return (_MODEL_ORDER.index(slug), slug)
        return (len(_MODEL_ORDER), slug)

    column_slugs = sorted(column_slugs, key=_sort_key)

    ordered_rows = sorted(
        zip(row_labels, row_slug_order, values_by_row),
        key=lambda item: _sort_key(item[1]),
    )
    row_labels = [label for label, _, _ in ordered_rows]
    matrix = np.array(
        [
            [row_metrics.get(slug, np.nan) for slug in column_slugs]
            for _, _, row_metrics in ordered_rows
        ]
    )
    column_labels = [_MODEL_DISPLAY.get(slug, slug) for slug in column_slugs]

    fig, ax = plt.subplots(
        figsize=(max(7, len(column_labels) * 1.35), max(4, len(row_labels) * 0.85))
    )
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    fig.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Benchmarked Against")
    ax.set_ylabel("Checkpoint Trained Against")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    fig.tight_layout()
    if output_path is None:
        output_path = Path("checkpoint_cross_model_heatmap.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot checkpoint cross-model heatmap")
    parser.add_argument(
        "--eval-dirs",
        nargs="+",
        required=True,
        help="Posthoc benchmark eval directories to include",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output PNG path; default is ./checkpoint_cross_model_heatmap.png",
    )
    parser.add_argument(
        "--title",
        default="Checkpoint Cross-Model Generalisation",
        help="Optional plot title override",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = generate_plot(
        eval_dirs=args.eval_dirs,
        output_path=args.output_path,
        title=args.title,
    )
    print(output_path)


if __name__ == "__main__":
    main()
