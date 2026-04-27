"""Compare sanity sweeps run with different optimizer step budgets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

METRICS = [
    ("val", "Validation", 0.50),
    ("mmlu_20", "MMLU-20", 0.25),
    ("xeval_dataset_wikisum", "WikiSum xeval", 0.50),
]

COMPARISON_COLORS = {
    "group_a": "#1f77b4",
    "group_b": "#d62728",
}


@dataclass
class RunData:
    label: str
    train_ids: int
    val_steps: list[int]
    val_acc: list[float]
    final_metrics: dict[str, float]


def _load_eval_payloads(dir_path: Path) -> list[dict]:
    payloads: list[dict] = []
    if not dir_path.exists():
        return payloads
    for path in sorted(
        dir_path.glob("step_*.json"),
        key=lambda item: int(item.stem.split("_")[1]),
    ):
        payloads.append(json.loads(path.read_text()))
    return payloads


def _label_and_ids(config: dict) -> tuple[str, int]:
    data_cfg = config.get("data", {})
    train_ids = int(data_cfg.get("max_train_ids", 0))
    rand_labels = bool(data_cfg.get("randomize_train_labels", False))
    label = f"{train_ids} IDs"
    if rand_labels:
        label += " + rand labels"
    return label, train_ids


def _load_run(run_dir: Path) -> RunData | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text()) or {}
    label, train_ids = _label_and_ids(config)
    if "rand labels" in label:
        return None

    val_payloads = _load_eval_payloads(run_dir / "val_predictions")
    if not val_payloads:
        return None
    val_steps = [int(payload["step"]) for payload in val_payloads]
    val_acc = [float(payload["accuracy"]) for payload in val_payloads]

    final_metrics: dict[str, float] = {"val": val_acc[-1]}
    for metric_name, _, _ in METRICS[1:]:
        payloads = _load_eval_payloads(run_dir / "benchmark_predictions" / metric_name)
        if not payloads:
            raise ValueError(f"Missing {metric_name} payloads in {run_dir}")
        final_metrics[metric_name] = float(payloads[-1]["accuracy"])

    return RunData(
        label=label,
        train_ids=train_ids,
        val_steps=val_steps,
        val_acc=val_acc,
        final_metrics=final_metrics,
    )


def _load_group(base_dir: Path) -> dict[int, RunData]:
    runs: dict[int, RunData] = {}
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run = _load_run(run_dir)
        if run is None:
            continue
        runs[run.train_ids] = run
    return runs


def plot_val_curves(
    runs_a: dict[int, RunData],
    runs_b: dict[int, RunData],
    *,
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    common_ids = sorted(set(runs_a) & set(runs_b))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    fig.suptitle("Validation Curves: 100-Step vs 400-Step Sanity Sweeps", fontsize=14)

    for axis, train_ids in zip(axes.flat, common_ids, strict=True):
        run_a = runs_a[train_ids]
        run_b = runs_b[train_ids]
        axis.plot(
            run_a.val_steps,
            run_a.val_acc,
            marker="o",
            linewidth=2,
            color=COMPARISON_COLORS["group_a"],
            label=label_a,
        )
        axis.plot(
            run_b.val_steps,
            run_b.val_acc,
            marker="o",
            linewidth=2,
            color=COMPARISON_COLORS["group_b"],
            label=label_b,
        )
        axis.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        axis.set_title(f"{train_ids} IDs")
        axis.set_xlabel("Optimizer Step")
        axis.set_ylabel("Validation accuracy")
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="lower right")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_bars(
    runs_a: dict[int, RunData],
    runs_b: dict[int, RunData],
    *,
    label_a: str,
    label_b: str,
    output_path: Path,
) -> None:
    common_ids = sorted(set(runs_a) & set(runs_b))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Sanity Sweep Endpoints: 100-Step vs 400-Step", fontsize=14)

    x = np.arange(len(common_ids))
    width = 0.36
    xlabels = [f"{train_ids} IDs" for train_ids in common_ids]

    for axis, (metric_name, title, chance) in zip(axes, METRICS, strict=True):
        vals_a = [runs_a[train_ids].final_metrics[metric_name] for train_ids in common_ids]
        vals_b = [runs_b[train_ids].final_metrics[metric_name] for train_ids in common_ids]
        bars_a = axis.bar(
            x - width / 2,
            vals_a,
            width,
            label=label_a,
            color=COMPARISON_COLORS["group_a"],
        )
        bars_b = axis.bar(
            x + width / 2,
            vals_b,
            width,
            label=label_b,
            color=COMPARISON_COLORS["group_b"],
        )
        axis.axhline(chance, color="gray", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(xlabels, rotation=20, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.grid(True, axis="y", alpha=0.3)
        for bars in (bars_a, bars_b):
            for bar in bars:
                height = bar.get_height()
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-a",
        default="results/sanity_overnight",
        help="Baseline sweep directory, typically the 100-step runs.",
    )
    parser.add_argument(
        "--group-b",
        default="results/sanity_step_budget_400",
        help="Comparison sweep directory, typically the 400-step runs.",
    )
    parser.add_argument(
        "--label-a",
        default="100 steps",
        help="Legend label for group A.",
    )
    parser.add_argument(
        "--label-b",
        default="400 steps",
        help="Legend label for group B.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sanity_step_budget_400/plots",
        help="Directory where comparison plots should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    group_a = Path(args.group_a)
    group_b = Path(args.group_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_a = _load_group(group_a)
    runs_b = _load_group(group_b)
    if not runs_a or not runs_b:
        raise ValueError("Both groups must contain completed sanity runs")

    plot_val_curves(
        runs_a,
        runs_b,
        label_a=args.label_a,
        label_b=args.label_b,
        output_path=output_dir / "sanity_val_curve_budget_comparison.png",
    )
    plot_endpoint_bars(
        runs_a,
        runs_b,
        label_a=args.label_a,
        label_b=args.label_b,
        output_path=output_dir / "sanity_endpoint_budget_comparison.png",
    )

    print(output_dir)


if __name__ == "__main__":
    main()
