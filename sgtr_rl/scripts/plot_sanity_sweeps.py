"""Plot cross-run summaries for sanity-check sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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

RUN_COLORS = {
    "1 IDs": "#1f77b4",
    "10 IDs": "#d62728",
    "40 IDs": "#2ca02c",
    "80 IDs": "#ff7f0e",
    "80 IDs + rand labels": "#9467bd",
}

BAR_COLORS = {
    "step_0": "#4c78a8",
    "step_final": "#e45756",
    "late_train": "#72b7b2",
    "val_final": "#f58518",
    "gap": "#b279a2",
    "late_nll": "#54a24b",
    "val_nll": "#eeca3b",
}


@dataclass
class RunSummary:
    run_dir: Path
    label: str
    order_key: tuple[int, int]
    train_steps: list[int]
    train_nll: list[float]
    train_acc: list[float]
    val_steps: list[int]
    val_nll: list[float]
    val_acc: list[float]
    initial_metrics: dict[str, float]
    final_metrics: dict[str, float]
    delta_metrics: dict[str, float]
    late_train_acc: float
    late_train_nll: float
    final_val_acc: float
    final_val_nll: float


def _moving_average(values: list[float], window: int = 5) -> np.ndarray:
    if not values:
        return np.array([])
    if len(values) <= 2:
        return np.array(values, dtype=float)
    window = min(window, len(values))
    kernel = np.ones(window) / window
    padded = np.pad(
        np.array(values, dtype=float),
        (window - 1, 0),
        mode="edge",
    )
    return np.convolve(padded, kernel, mode="valid")


def _label_from_config(config: dict) -> tuple[str, tuple[int, int]]:
    data_cfg = config.get("data", {})
    max_train_ids = int(data_cfg.get("max_train_ids", 0))
    rand_labels = bool(data_cfg.get("randomize_train_labels", False))
    if rand_labels:
        return f"{max_train_ids} IDs + rand labels", (max_train_ids, 1)
    return f"{max_train_ids} IDs", (max_train_ids, 0)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for record in records:
        if key in record:
            steps.append(int(record["step"]))
            values.append(float(record[key]))
    return steps, values


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


def _metric_payloads(run_dir: Path, metric_name: str) -> list[dict]:
    if metric_name == "val":
        return _load_eval_payloads(run_dir / "val_predictions")
    return _load_eval_payloads(run_dir / "benchmark_predictions" / metric_name)


def _load_run(run_dir: Path) -> RunSummary:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    label, order_key = _label_from_config(config)

    records = _load_jsonl(run_dir / "metrics" / "metrics.jsonl")
    train_steps, train_nll = _series(records, "train/nll")
    _, train_acc = _series(records, "train/accuracy")
    val_steps, val_nll = _series(records, "val/nll")
    _, val_acc = _series(records, "val/accuracy")

    if not train_acc or not train_nll or not val_acc or not val_nll:
        raise ValueError(f"Incomplete metrics in {run_dir}")

    initial_metrics: dict[str, float] = {}
    final_metrics: dict[str, float] = {}
    delta_metrics: dict[str, float] = {}
    for metric_name, _, _ in METRICS:
        payloads = _metric_payloads(run_dir, metric_name)
        if not payloads:
            raise ValueError(f"Missing payloads for {metric_name} in {run_dir}")
        initial = float(payloads[0]["accuracy"])
        final = float(payloads[-1]["accuracy"])
        initial_metrics[metric_name] = initial
        final_metrics[metric_name] = final
        delta_metrics[metric_name] = final - initial

    tail = min(10, len(train_acc))
    late_train_acc = statistics.mean(train_acc[-tail:])
    late_train_nll = statistics.mean(train_nll[-tail:])

    return RunSummary(
        run_dir=run_dir,
        label=label,
        order_key=order_key,
        train_steps=train_steps,
        train_nll=train_nll,
        train_acc=train_acc,
        val_steps=val_steps,
        val_nll=val_nll,
        val_acc=val_acc,
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        delta_metrics=delta_metrics,
        late_train_acc=late_train_acc,
        late_train_nll=late_train_nll,
        final_val_acc=val_acc[-1],
        final_val_nll=val_nll[-1],
    )


def _load_runs(base_dir: Path) -> list[RunSummary]:
    runs = []
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if not (run_dir / "config.yaml").exists():
            continue
        if not (run_dir / "metrics" / "metrics.jsonl").exists():
            continue
        runs.append(_load_run(run_dir))
    return sorted(runs, key=lambda run: run.order_key)


def _save_summary_csv(runs: list[RunSummary], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "run_dir",
                "late_train_acc",
                "late_train_nll",
                "final_val_acc",
                "final_val_nll",
                "val_initial",
                "val_final",
                "val_delta",
                "mmlu_20_initial",
                "mmlu_20_final",
                "mmlu_20_delta",
                "xeval_dataset_wikisum_initial",
                "xeval_dataset_wikisum_final",
                "xeval_dataset_wikisum_delta",
            ],
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "label": run.label,
                    "run_dir": str(run.run_dir),
                    "late_train_acc": run.late_train_acc,
                    "late_train_nll": run.late_train_nll,
                    "final_val_acc": run.final_val_acc,
                    "final_val_nll": run.final_val_nll,
                    "val_initial": run.initial_metrics["val"],
                    "val_final": run.final_metrics["val"],
                    "val_delta": run.delta_metrics["val"],
                    "mmlu_20_initial": run.initial_metrics["mmlu_20"],
                    "mmlu_20_final": run.final_metrics["mmlu_20"],
                    "mmlu_20_delta": run.delta_metrics["mmlu_20"],
                    "xeval_dataset_wikisum_initial": run.initial_metrics["xeval_dataset_wikisum"],
                    "xeval_dataset_wikisum_final": run.final_metrics["xeval_dataset_wikisum"],
                    "xeval_dataset_wikisum_delta": run.delta_metrics["xeval_dataset_wikisum"],
                }
            )


def _style_for_run(index: int, run: RunSummary, total: int) -> tuple[tuple[float, ...], str]:
    color = RUN_COLORS.get(run.label)
    if color is None:
        fallback = plt.get_cmap("tab10")
        color = fallback(index % 10)
    linestyle = "--" if "rand labels" in run.label.lower() else "-"
    return color, linestyle


def plot_training_dynamics(runs: list[RunSummary], output_path: Path) -> None:
    max_step = max(run.train_steps[-1] for run in runs if run.train_steps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex="col", constrained_layout=True)
    fig.suptitle(f"Sanity Sweep: Training Dynamics up to {max_step} Optimizer Steps", fontsize=14)

    ax_train_nll, ax_train_acc = axes[0]
    ax_val_nll, ax_val_acc = axes[1]

    for index, run in enumerate(runs):
        color, linestyle = _style_for_run(index, run, len(runs))
        ax_train_nll.plot(
            run.train_steps,
            _moving_average(run.train_nll),
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=run.label,
        )
        ax_train_acc.plot(
            run.train_steps,
            _moving_average(run.train_acc),
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=run.label,
        )
        ax_val_nll.plot(
            run.val_steps,
            run.val_nll,
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            label=run.label,
        )
        ax_val_acc.plot(
            run.val_steps,
            run.val_acc,
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2,
            markersize=5,
            label=run.label,
        )

    ax_train_nll.set_title("Train NLL (smoothed)")
    ax_train_nll.set_ylabel("NLL")
    ax_train_nll.grid(True, alpha=0.3)

    ax_train_acc.set_title("Train Accuracy (smoothed)")
    ax_train_acc.set_ylabel("Accuracy")
    ax_train_acc.set_ylim(0.0, 1.05)
    ax_train_acc.grid(True, alpha=0.3)

    ax_val_nll.set_title("Validation NLL")
    ax_val_nll.set_xlabel("Optimizer Step")
    ax_val_nll.set_ylabel("NLL")
    ax_val_nll.grid(True, alpha=0.3)

    ax_val_acc.set_title("Validation Accuracy")
    ax_val_acc.set_xlabel("Optimizer Step")
    ax_val_acc.set_ylabel("Accuracy")
    ax_val_acc.set_ylim(0.0, 1.05)
    ax_val_acc.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax_val_acc.grid(True, alpha=0.3)
    ax_val_acc.legend(loc="lower right", fontsize=9)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_accuracies(runs: list[RunSummary], output_path: Path) -> None:
    max_step = max(run.val_steps[-1] for run in runs if run.val_steps)
    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=(16, 5),
        sharey=False,
        constrained_layout=True,
    )
    fig.suptitle(f"Sanity Sweep: Step 0 vs Step {max_step} Accuracy", fontsize=14)

    x = np.arange(len(runs))
    width = 0.36
    labels = [run.label for run in runs]

    for axis, (metric_name, title, chance) in zip(axes, METRICS, strict=True):
        initial = [run.initial_metrics[metric_name] for run in runs]
        final = [run.final_metrics[metric_name] for run in runs]
        bars_a = axis.bar(
            x - width / 2,
            initial,
            width,
            label="Step 0",
            color=BAR_COLORS["step_0"],
        )
        bars_b = axis.bar(
            x + width / 2,
            final,
            width,
            label=f"Step {max_step}",
            color=BAR_COLORS["step_final"],
        )
        axis.axhline(chance, color="gray", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
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
    axes[0].legend(loc="upper left", fontsize=9)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_deltas(runs: list[RunSummary], output_path: Path) -> None:
    matrix = np.array(
        [
            [run.delta_metrics[metric_name] for metric_name, _, _ in METRICS]
            for run in runs
        ],
        dtype=float,
    )
    vmax = max(0.01, float(np.max(np.abs(matrix))))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Sanity Sweep: Accuracy Change from Step 0 to Step 100")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([title for _, title, _ in METRICS])
    ax.set_yticks(np.arange(len(runs)))
    ax.set_yticklabels([run.label for run in runs])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if abs(value) > vmax * 0.45 else "black"
            ax.text(
                col,
                row,
                f"{value:+.1%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Accuracy delta")
    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_gap(runs: list[RunSummary], output_path: Path) -> None:
    labels = [run.label for run in runs]
    x = np.arange(len(runs))
    width = 0.36

    late_train_acc = [run.late_train_acc for run in runs]
    final_val_acc = [run.final_val_acc for run in runs]
    gaps = [train - val for train, val in zip(late_train_acc, final_val_acc, strict=True)]
    late_train_nll = [run.late_train_nll for run in runs]
    final_val_nll = [run.final_val_nll for run in runs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sanity Sweep: Late-Train Batch Behavior vs Validation", fontsize=14)

    ax = axes[0]
    bars_a = ax.bar(
        x - width / 2,
        late_train_acc,
        width,
        label="Late-train batch accuracy",
        color=BAR_COLORS["late_train"],
    )
    bars_b = ax.bar(
        x + width / 2,
        final_val_acc,
        width,
        label="Final val accuracy",
        color=BAR_COLORS["val_final"],
    )
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Accuracy")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    for bars in (bars_a, bars_b):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax = axes[1]
    gap_bars = ax.bar(
        x,
        gaps,
        color=BAR_COLORS["gap"],
        label="Late-train batch acc - final val acc",
    )
    ax.plot(
        x,
        late_train_nll,
        color=BAR_COLORS["late_nll"],
        marker="o",
        linewidth=2,
        label="Late-train batch NLL",
    )
    ax.plot(
        x,
        final_val_nll,
        color=BAR_COLORS["val_nll"],
        marker="o",
        linewidth=2,
        label="Final val NLL",
    )
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Accuracy Gap and NLL")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    y_values = gaps + late_train_nll + final_val_nll
    y_min = min(y_values)
    y_max = max(y_values)
    pad = max(0.02, (y_max - y_min) * 0.15)
    ax.set_ylim(y_min - pad, y_max + pad)
    for bar in gap_bars:
        height = bar.get_height()
        offset = 0.02 if height >= 0 else -0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{height:+.1%}",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="results/sanity_overnight",
        help="Directory containing sanity sweep run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write plots. Defaults to <base_dir>/plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(base_dir)
    if not runs:
        raise ValueError(f"No run directories found in {base_dir}")

    plot_training_dynamics(runs, output_dir / "sanity_training_dynamics.png")
    plot_endpoint_accuracies(runs, output_dir / "sanity_endpoint_accuracies.png")
    plot_accuracy_deltas(runs, output_dir / "sanity_accuracy_deltas.png")
    plot_generalization_gap(runs, output_dir / "sanity_generalization_gap.png")
    _save_summary_csv(runs, output_dir / "sanity_summary.csv")

    print(output_dir)


if __name__ == "__main__":
    main()
