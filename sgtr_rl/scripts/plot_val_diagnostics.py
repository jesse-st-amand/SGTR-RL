"""Plot validation prediction-balance, loss, and margin diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

COMPARE_COLORS = {
    "a_acc": "#1f77b4",
    "b_acc": "#d62728",
    "a_1": "#1f77b4",
    "a_2": "#ff7f0e",
    "b_1": "#2ca02c",
    "b_2": "#d62728",
}

MARGIN_COLORS = {
    "1": "#1f77b4",
    "2": "#d62728",
}


def _step_key(path: Path) -> int:
    match = re.search(r"step_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Could not parse step from {path}")
    return int(match.group(1))


def _load_val_payloads(run_dir: Path) -> list[dict]:
    return _load_prediction_payloads(run_dir, split_name="val")


def _load_prediction_payloads(run_dir: Path, *, split_name: str) -> list[dict]:
    val_dir = run_dir / f"{split_name}_predictions"
    payloads = []
    for path in sorted(val_dir.glob("step_*.json"), key=_step_key):
        payloads.append(json.loads(path.read_text()))
    if not payloads:
        raise ValueError(f"No val prediction payloads found in {val_dir}")
    return payloads


def _load_margin_payloads(run_dir: Path) -> list[dict]:
    return _load_diagnostic_payloads(run_dir, split_name="val")


def _load_diagnostic_payloads(run_dir: Path, *, split_name: str) -> list[dict]:
    diag_dir = run_dir / f"{split_name}_diagnostics"
    payloads = []
    for path in sorted(diag_dir.glob("step_*.json"), key=_step_key):
        payloads.append(json.loads(path.read_text()))
    if not payloads:
        raise ValueError(f"No val diagnostics payloads found in {diag_dir}")
    return payloads


def _load_metric_records(run_dir: Path) -> list[dict]:
    metrics_path = run_dir / "metrics" / "metrics.jsonl"
    if not metrics_path.exists():
        raise ValueError(f"No metrics file found at {metrics_path}")
    return [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]


def _metric_series(records: list[dict], metric_name: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for record in records:
        if metric_name in record:
            steps.append(int(record["step"]))
            values.append(float(record[metric_name]))
    if not steps:
        raise ValueError(f"Metric {metric_name} not found in records")
    return steps, values


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start:index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def _summarize_margin_payloads(payloads: list[dict]) -> dict[str, list[float]]:
    steps: list[int] = []
    mean_abs_margin: list[float] = []
    low_margin_share: list[float] = []
    wrong_confident_count: list[float] = []
    diag_accuracy: list[float] = []
    predicted_1_pct: list[float] = []

    for payload in payloads:
        examples = payload["examples"]
        steps.append(int(payload["step"]))
        margins = [float(example["margin_1_minus_2"]) for example in examples]
        preds = [
            str(example.get("predicted_by_margin", "1" if margin >= 0 else "2"))
            for example, margin in zip(examples, margins, strict=False)
        ]
        targets = [str(example["target"]) for example in examples]
        mean_abs_margin.append(sum(abs(margin) for margin in margins) / len(margins))
        low_margin_share.append(
            sum(abs(margin) < 0.25 for margin in margins) / len(margins)
        )
        wrong_confident_count.append(
            float(
                sum(
                    pred != target and abs(margin) >= 2.0
                    for pred, target, margin in zip(preds, targets, margins, strict=False)
                )
            )
        )
        diag_accuracy.append(
            sum(pred == target for pred, target in zip(preds, targets, strict=False))
            / len(preds)
        )
        predicted_1_pct.append(sum(pred == "1" for pred in preds) / len(preds))

    return {
        "steps": steps,
        "mean_abs_margin": mean_abs_margin,
        "low_margin_share": low_margin_share,
        "wrong_confident_count": wrong_confident_count,
        "diag_accuracy": diag_accuracy,
        "predicted_1_pct": predicted_1_pct,
    }


def _logsumexp_two(a: float, b: float) -> float:
    max_val = max(a, b)
    return max_val + math.log(math.exp(a - max_val) + math.exp(b - max_val))


def _extract_full_split_diagnostic_stats(run_dir: Path, *, split_name: str) -> dict:
    margin_payloads = _load_diagnostic_payloads(run_dir, split_name=split_name)
    val_payloads = _load_prediction_payloads(run_dir, split_name=split_name)
    val_by_step = {int(payload["step"]): payload for payload in val_payloads}
    total_val = len(val_payloads[0]["predictions"])
    diagnostic_size = len(margin_payloads[0]["examples"])
    panel_keys = [
        f"{index}:{example['id']}:{example['target']}"
        for index, example in enumerate(margin_payloads[0]["examples"])
    ]

    steps: list[int] = []
    mean_target_nll: list[float] = []
    mean_binary_ce: list[float] = []
    mean_pair_mass: list[float] = []
    mean_other_mass: list[float] = []
    diag_accuracy: list[float] = []
    wrong_count: list[float] = []
    confident_wrong_count: list[float] = []
    prediction_mismatch_count: list[float] = []
    mean_abs_margin: list[float] = []
    low_margin_share: list[float] = []
    example_traces: dict[str, dict] = {}

    for payload in margin_payloads:
        step = int(payload["step"])
        examples = payload["examples"]
        step_target_nlls: list[float] = []
        step_binary_ce: list[float] = []
        step_pair_mass: list[float] = []
        step_other_mass: list[float] = []
        step_margins: list[float] = []
        step_correct: list[bool] = []
        step_mismatch = 0

        val_payload = val_by_step.get(step)
        if val_payload is None:
            raise ValueError(f"Missing val_predictions payload for step {step}")

        for index, example in enumerate(examples):
            panel_key = panel_keys[index]
            example_id = str(example["id"])
            target = str(example["target"])
            logprob_1 = float(example["logprob_1"])
            logprob_2 = float(example["logprob_2"])
            margin = float(example["margin_1_minus_2"])
            target_logprob = logprob_1 if target == "1" else logprob_2
            target_nll = -target_logprob
            binary_ce = _logsumexp_two(logprob_1, logprob_2) - target_logprob
            pair_mass = min(1.0, math.exp(logprob_1) + math.exp(logprob_2))
            other_mass = max(0.0, 1.0 - pair_mass)
            pred = str(example.get("predicted_by_margin", "1" if margin >= 0 else "2"))
            is_correct = pred == target

            step_target_nlls.append(target_nll)
            step_binary_ce.append(binary_ce)
            step_pair_mass.append(pair_mass)
            step_other_mass.append(other_mass)
            step_margins.append(margin)
            step_correct.append(is_correct)

            if diagnostic_size == total_val:
                generated_prediction = str(val_payload["predictions"][index]["prediction"])
                step_mismatch += int(generated_prediction != pred)

            trace = example_traces.setdefault(panel_key, {
                "panel_key": panel_key,
                "position": index + 1,
                "id": example_id,
                "target": target,
                "prompt_preview": str(example["prompt_preview"]),
                "steps": [],
                "target_nll": [],
                "binary_ce": [],
                "other_mass": [],
                "margin": [],
                "correct_by_margin": [],
            })
            trace["steps"].append(step)
            trace["target_nll"].append(target_nll)
            trace["binary_ce"].append(binary_ce)
            trace["other_mass"].append(other_mass)
            trace["margin"].append(margin)
            trace["correct_by_margin"].append(is_correct)

        steps.append(step)
        mean_target_nll.append(sum(step_target_nlls) / len(step_target_nlls))
        mean_binary_ce.append(sum(step_binary_ce) / len(step_binary_ce))
        mean_pair_mass.append(sum(step_pair_mass) / len(step_pair_mass))
        mean_other_mass.append(sum(step_other_mass) / len(step_other_mass))
        diag_accuracy.append(sum(step_correct) / len(step_correct))
        wrong_count.append(sum(not item["correct"] for item in val_payload["predictions"]))
        confident_wrong_count.append(
            float(
                sum(
                    (not correct) and abs(margin) >= 2.0
                    for correct, margin in zip(step_correct, step_margins, strict=False)
                )
            )
        )
        mean_abs_margin.append(sum(abs(margin) for margin in step_margins) / len(step_margins))
        low_margin_share.append(
            sum(abs(margin) < 0.25 for margin in step_margins) / len(step_margins)
        )
        prediction_mismatch_count.append(float(step_mismatch))

    return {
        "steps": steps,
        "coverage": diagnostic_size / total_val,
        "diagnostic_size": diagnostic_size,
        "total_val": total_val,
        "mean_target_nll": mean_target_nll,
        "mean_binary_ce": mean_binary_ce,
        "mean_pair_mass": mean_pair_mass,
        "mean_other_mass": mean_other_mass,
        "diag_accuracy": diag_accuracy,
        "wrong_count": wrong_count,
        "confident_wrong_count": confident_wrong_count,
        "prediction_mismatch_count": prediction_mismatch_count,
        "mean_abs_margin": mean_abs_margin,
        "low_margin_share": low_margin_share,
        "example_traces": example_traces,
    }


def _extract_full_val_diagnostic_stats(run_dir: Path) -> dict:
    return _extract_full_split_diagnostic_stats(run_dir, split_name="val")


def plot_prediction_balance_comparison(
    *,
    run_a: Path,
    label_a: str,
    run_b: Path,
    label_b: str,
    output_path: Path,
) -> None:
    payloads_a = _load_val_payloads(run_a)
    payloads_b = _load_val_payloads(run_b)

    def series(payloads: list[dict]) -> tuple[list[int], list[float], list[int], list[int]]:
        steps = [int(payload["step"]) for payload in payloads]
        acc = [float(payload["accuracy"]) for payload in payloads]
        pred_1 = []
        pred_2 = []
        for payload in payloads:
            predictions = payload["predictions"]
            pred_1.append(sum(1 for item in predictions if item["prediction"] == "1"))
            pred_2.append(sum(1 for item in predictions if item["prediction"] == "2"))
        return steps, acc, pred_1, pred_2

    steps_a, acc_a, pred_1_a, pred_2_a = series(payloads_a)
    steps_b, acc_b, pred_1_b, pred_2_b = series(payloads_b)
    total_a = len(payloads_a[0]["predictions"])
    total_b = len(payloads_b[0]["predictions"])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    fig.suptitle("Validation Dynamics and Prediction Balance", fontsize=14)

    axes[0].plot(
        steps_a,
        acc_a,
        color=COMPARE_COLORS["a_acc"],
        marker="o",
        linewidth=2,
        label=label_a,
    )
    axes[0].plot(
        steps_b,
        acc_b,
        color=COMPARE_COLORS["b_acc"],
        marker="o",
        linewidth=2,
        label=label_b,
    )
    axes[0].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[0].set_title("Validation accuracy")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(
        steps_a,
        pred_1_a,
        color=COMPARE_COLORS["a_1"],
        marker="o",
        linewidth=2,
        label='Predict "1"',
    )
    axes[1].plot(
        steps_a,
        pred_2_a,
        color=COMPARE_COLORS["a_2"],
        marker="o",
        linewidth=2,
        label='Predict "2"',
    )
    axes[1].axhline(total_a / 2, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title(f"{label_a}: prediction counts")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Count")
    axes[1].set_ylim(0, total_a + 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="center right")

    axes[2].plot(
        steps_b,
        pred_1_b,
        color=COMPARE_COLORS["b_1"],
        marker="o",
        linewidth=2,
        label='Predict "1"',
    )
    axes[2].plot(
        steps_b,
        pred_2_b,
        color=COMPARE_COLORS["b_2"],
        marker="o",
        linewidth=2,
        label='Predict "2"',
    )
    axes[2].axhline(total_b / 2, color="gray", linestyle=":", linewidth=1)
    axes[2].set_title(f"{label_b}: prediction counts")
    axes[2].set_xlabel("Optimizer step")
    axes[2].set_ylabel("Count")
    axes[2].set_ylim(0, total_b + 1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="center right")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve_comparison(
    *,
    run_a: Path,
    label_a: str,
    run_b: Path,
    label_b: str,
    output_path: Path,
) -> None:
    records_a = _load_metric_records(run_a)
    records_b = _load_metric_records(run_b)

    train_steps_a, train_nll_a = _metric_series(records_a, "train/nll")
    train_steps_b, train_nll_b = _metric_series(records_b, "train/nll")
    val_steps_a, val_nll_a = _metric_series(records_a, "val/nll")
    val_steps_b, val_nll_b = _metric_series(records_b, "val/nll")

    smooth_a = _moving_average(train_nll_a, window=10)
    smooth_b = _moving_average(train_nll_b, window=10)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle("Loss Curves", fontsize=14)

    axes[0].plot(
        train_steps_a,
        train_nll_a,
        color=COMPARE_COLORS["a_acc"],
        alpha=0.25,
        linewidth=1,
    )
    axes[0].plot(
        train_steps_a,
        smooth_a,
        color=COMPARE_COLORS["a_acc"],
        linewidth=2.5,
        label=f"{label_a} (10-step avg)",
    )
    axes[0].plot(
        train_steps_b,
        train_nll_b,
        color=COMPARE_COLORS["b_acc"],
        alpha=0.25,
        linewidth=1,
    )
    axes[0].plot(
        train_steps_b,
        smooth_b,
        color=COMPARE_COLORS["b_acc"],
        linewidth=2.5,
        label=f"{label_b} (10-step avg)",
    )
    axes[0].set_title("Train NLL")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("NLL")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        val_steps_a,
        val_nll_a,
        color=COMPARE_COLORS["a_acc"],
        marker="o",
        linewidth=2,
        label=label_a,
    )
    axes[1].plot(
        val_steps_b,
        val_nll_b,
        color=COMPARE_COLORS["b_acc"],
        marker="o",
        linewidth=2,
        label=label_b,
    )
    axes[1].set_title("Validation NLL")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("NLL")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostic_overview(
    *,
    run_dir: Path,
    output_path: Path,
) -> None:
    records = _load_metric_records(run_dir)
    margin_payloads = _load_margin_payloads(run_dir)
    val_payloads = _load_val_payloads(run_dir)

    train_steps, train_nll = _metric_series(records, "train/nll")
    val_steps, val_nll = _metric_series(records, "val/nll")
    val_acc_steps, val_acc = _metric_series(records, "val/accuracy")
    margin_summary = _summarize_margin_payloads(margin_payloads)
    train_smooth = _moving_average(train_nll, window=10)

    pred_steps = [int(payload["step"]) for payload in val_payloads]
    pred_1 = []
    pred_2 = []
    total = len(val_payloads[0]["predictions"])
    for payload in val_payloads:
        predictions = payload["predictions"]
        pred_1.append(sum(1 for item in predictions if item["prediction"] == "1"))
        pred_2.append(sum(1 for item in predictions if item["prediction"] == "2"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle("Loss and Margin Diagnostics", fontsize=14)

    axes[0, 0].plot(train_steps, train_nll, color="#1f77b4", alpha=0.2, linewidth=1)
    axes[0, 0].plot(
        train_steps,
        train_smooth,
        color="#1f77b4",
        linewidth=2.5,
        label="Train NLL (10-step avg)",
    )
    axes[0, 0].plot(
        val_steps,
        val_nll,
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Val NLL",
    )
    axes[0, 0].set_title("Loss curves")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("NLL")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right")

    axes[0, 1].plot(
        val_acc_steps,
        val_acc,
        color="#2ca02c",
        marker="o",
        linewidth=2,
        label="Val accuracy",
    )
    axes[0, 1].plot(
        margin_summary["steps"],
        margin_summary["diag_accuracy"],
        color="#9467bd",
        marker="o",
        linewidth=2,
        label="Diag-panel accuracy",
    )
    axes[0, 1].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Optimizer step")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="lower right")

    axes[1, 0].plot(
        pred_steps,
        pred_1,
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label='Predict "1"',
    )
    axes[1, 0].plot(
        pred_steps,
        pred_2,
        color="#ff7f0e",
        marker="o",
        linewidth=2,
        label='Predict "2"',
    )
    axes[1, 0].axhline(total / 2, color="gray", linestyle=":", linewidth=1)
    axes[1, 0].set_title("Validation prediction counts")
    axes[1, 0].set_xlabel("Optimizer step")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_ylim(0, total + 1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="center right")

    axes[1, 1].plot(
        margin_summary["steps"],
        margin_summary["mean_abs_margin"],
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Mean |margin|",
    )
    axes[1, 1].plot(
        margin_summary["steps"],
        margin_summary["low_margin_share"],
        color="#17becf",
        marker="o",
        linewidth=2,
        label="Low-margin share",
    )
    axes[1, 1].plot(
        margin_summary["steps"],
        margin_summary["wrong_confident_count"],
        color="#8c564b",
        marker="o",
        linewidth=2,
        label="Confident wrong count",
    )
    axes[1, 1].set_title("Margin summary")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].set_ylabel("Diagnostic summary")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper left")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_full_val_nll_breakdown(
    *,
    run_dir: Path,
    output_path: Path,
) -> None:
    records = _load_metric_records(run_dir)
    stats = _extract_full_val_diagnostic_stats(run_dir)
    val_steps, val_nll = _metric_series(records, "val/nll")
    val_acc_steps, val_acc = _metric_series(records, "val/accuracy")
    val_nll_by_step = dict(zip(val_steps, val_nll, strict=False))
    full_coverage = stats["diagnostic_size"] == stats["total_val"]
    estimated_eot_nll = (
        [
            max(0.0, 2.0 * val_nll_by_step[step] - answer_nll)
            for step, answer_nll in zip(
                stats["steps"], stats["mean_target_nll"], strict=False
            )
            if step in val_nll_by_step
        ]
        if full_coverage
        else []
    )

    coverage_label = (
        f"{stats['diagnostic_size']}/{stats['total_val']} val examples"
        f" ({stats['coverage']:.0%} coverage)"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle(f"Full-Val NLL Breakdown ({coverage_label})", fontsize=14)

    axes[0, 0].plot(
        val_steps,
        val_nll,
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Logged val NLL",
    )
    axes[0, 0].plot(
        stats["steps"],
        stats["mean_target_nll"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Mean answer-token NLL",
    )
    axes[0, 0].plot(
        stats["steps"],
        stats["mean_binary_ce"],
        color="#2ca02c",
        marker="o",
        linewidth=2,
        label="Binary CE over {1,2}",
    )
    if full_coverage and len(estimated_eot_nll) == len(stats["steps"]):
        axes[0, 0].plot(
            stats["steps"],
            estimated_eot_nll,
            color="#9467bd",
            marker="o",
            linewidth=2,
            linestyle="--",
            label="Estimated <|eot_id|> NLL",
        )
    axes[0, 0].set_title("Where val NLL comes from")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("NLL")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left")

    axes[0, 1].plot(
        stats["steps"],
        stats["mean_pair_mass"],
        color="#9467bd",
        marker="o",
        linewidth=2,
        label='Mean prob mass on {"1","2"}',
    )
    axes[0, 1].plot(
        stats["steps"],
        stats["mean_other_mass"],
        color="#ff7f0e",
        marker="o",
        linewidth=2,
        label="Mean prob mass on other tokens",
    )
    axes[0, 1].set_title("Probability mass allocation")
    axes[0, 1].set_xlabel("Optimizer step")
    axes[0, 1].set_ylabel("Probability mass")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="upper right")

    axes[1, 0].plot(
        val_acc_steps,
        val_acc,
        color="#2ca02c",
        marker="o",
        linewidth=2,
        label="Val accuracy",
    )
    axes[1, 0].plot(
        stats["steps"],
        stats["diag_accuracy"],
        color="#9467bd",
        marker="o",
        linewidth=2,
        label="Diag argmax accuracy",
    )
    axes[1, 0].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[1, 0].set_title("Accuracy stays flatter than NLL")
    axes[1, 0].set_xlabel("Optimizer step")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="lower right")

    axis_count = axes[1, 0].twinx()
    axis_count.plot(
        stats["steps"],
        stats["wrong_count"],
        color="#8c564b",
        marker="o",
        linewidth=1.8,
        linestyle="--",
        label="Wrong examples",
    )
    axis_count.plot(
        stats["steps"],
        stats["confident_wrong_count"],
        color="#17becf",
        marker="o",
        linewidth=1.8,
        linestyle="--",
        label="Confident wrong examples",
    )
    if full_coverage:
        axis_count.plot(
            stats["steps"],
            stats["prediction_mismatch_count"],
            color="#ff7f0e",
            marker="o",
            linewidth=1.8,
            linestyle="--",
            label="Diag/gen mismatch count",
        )
    axis_count.set_ylabel("Example count")
    lines_a, labels_a = axes[1, 0].get_legend_handles_labels()
    lines_b, labels_b = axis_count.get_legend_handles_labels()
    axes[1, 0].legend(lines_a + lines_b, labels_a + labels_b, loc="center right")

    axes[1, 1].plot(
        stats["steps"],
        stats["mean_abs_margin"],
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Mean |margin|",
    )
    axes[1, 1].plot(
        stats["steps"],
        stats["low_margin_share"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Low-margin share",
    )
    axes[1, 1].set_title("Confidence profile")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].set_ylabel("Diagnostic summary")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper left")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_train_panel_overview(
    *,
    run_dir: Path,
    output_path: Path,
) -> None:
    stats = _extract_full_split_diagnostic_stats(run_dir, split_name="train_panel")
    records = _load_metric_records(run_dir)

    train_steps, train_nll = _metric_series(records, "train/nll")
    train_acc_steps, train_acc = _metric_series(records, "train/accuracy")
    panel_steps, panel_nll = _metric_series(records, "train_panel/nll")
    panel_acc_steps, panel_acc = _metric_series(records, "train_panel/accuracy")
    smooth_train_nll = _moving_average(train_nll, window=10)

    payloads = _load_prediction_payloads(run_dir, split_name="train_panel")
    pred_steps = [int(payload["step"]) for payload in payloads]
    pred_1 = []
    pred_2 = []
    total = len(payloads[0]["predictions"])
    for payload in payloads:
        predictions = payload["predictions"]
        pred_1.append(sum(1 for item in predictions if item["prediction"] == "1"))
        pred_2.append(sum(1 for item in predictions if item["prediction"] == "2"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle("Train-Panel Diagnostics", fontsize=14)

    axes[0, 0].plot(
        train_steps,
        train_nll,
        color="#1f77b4",
        alpha=0.18,
        linewidth=1,
    )
    axes[0, 0].plot(
        train_steps,
        smooth_train_nll,
        color="#1f77b4",
        linewidth=2.5,
        label="Batch train NLL (10-step avg)",
    )
    axes[0, 0].plot(
        panel_steps,
        panel_nll,
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Train-panel NLL",
    )
    axes[0, 0].plot(
        stats["steps"],
        stats["mean_target_nll"],
        color="#2ca02c",
        marker="o",
        linewidth=2,
        label="Train-panel answer-token NLL",
    )
    axes[0, 0].plot(
        stats["steps"],
        stats["mean_binary_ce"],
        color="#9467bd",
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Train-panel binary CE over {1,2}",
    )
    axes[0, 0].set_title("Train loss vs fixed train panel")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("NLL")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].plot(
        train_acc_steps,
        train_acc,
        color="#1f77b4",
        alpha=0.22,
        linewidth=1,
        label="Batch train accuracy",
    )
    axes[0, 1].plot(
        panel_acc_steps,
        panel_acc,
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Train-panel accuracy",
    )
    axes[0, 1].plot(
        stats["steps"],
        stats["diag_accuracy"],
        color="#2ca02c",
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Train-panel diag argmax accuracy",
    )
    axes[0, 1].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Optimizer step")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="lower right", fontsize=8)

    axes[1, 0].plot(
        pred_steps,
        pred_1,
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label='Predict "1"',
    )
    axes[1, 0].plot(
        pred_steps,
        pred_2,
        color="#ff7f0e",
        marker="o",
        linewidth=2,
        label='Predict "2"',
    )
    axes[1, 0].axhline(total / 2, color="gray", linestyle=":", linewidth=1)
    axes[1, 0].set_title("Train-panel prediction counts")
    axes[1, 0].set_xlabel("Optimizer step")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_ylim(0, total + 1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="center right")

    axes[1, 1].plot(
        stats["steps"],
        stats["mean_abs_margin"],
        color="#d62728",
        marker="o",
        linewidth=2,
        label="Mean |margin|",
    )
    axes[1, 1].plot(
        stats["steps"],
        stats["low_margin_share"],
        color="#17becf",
        marker="o",
        linewidth=2,
        label="Low-margin share",
    )
    axes[1, 1].plot(
        stats["steps"],
        stats["confident_wrong_count"],
        color="#8c564b",
        marker="o",
        linewidth=2,
        label="Confident wrong count",
    )
    axes[1, 1].set_title("Train-panel confidence")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].set_ylabel("Diagnostic summary")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper left")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hard_example_trajectories(
    *,
    run_dir: Path,
    output_path: Path,
    top_k: int = 6,
) -> None:
    stats = _extract_full_val_diagnostic_stats(run_dir)
    traces = sorted(
        stats["example_traces"].values(),
        key=lambda trace: trace["target_nll"][-1],
        reverse=True,
    )[:top_k]
    if not traces:
        raise ValueError(f"No diagnostic traces found in {run_dir}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    fig.suptitle("Hardest Validation Examples by Final Target NLL", fontsize=14)

    for trace in traces:
        short_id = trace["id"][:8]
        label = f"#{trace['position']} {short_id} | t={trace['target']}"
        color = MARGIN_COLORS.get(trace["target"], None)
        axes[0].plot(
            trace["steps"],
            trace["target_nll"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        axes[1].plot(
            trace["steps"],
            trace["margin"],
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

    axes[0].set_title("Target-token NLL")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("NLL")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)

    axes[1].axhline(0.0, color="gray", linestyle=":", linewidth=1)
    axes[1].set_title("Corresponding log p(1) - log p(2) margins")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Margin")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", fontsize=8)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_full_val_diagnostic_comparison(
    *,
    run_a: Path,
    label_a: str,
    run_b: Path,
    label_b: str,
    output_path: Path,
) -> None:
    stats_a = _extract_full_val_diagnostic_stats(run_a)
    stats_b = _extract_full_val_diagnostic_stats(run_b)
    records_a = _load_metric_records(run_a)
    records_b = _load_metric_records(run_b)

    val_steps_a, val_nll_a = _metric_series(records_a, "val/nll")
    val_steps_b, val_nll_b = _metric_series(records_b, "val/nll")
    val_acc_steps_a, val_acc_a = _metric_series(records_a, "val/accuracy")
    val_acc_steps_b, val_acc_b = _metric_series(records_b, "val/accuracy")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    fig.suptitle("Full-Val Diagnostic Comparison", fontsize=14)

    axes[0, 0].plot(
        val_steps_a,
        val_nll_a,
        color=COMPARE_COLORS["a_acc"],
        marker="o",
        linewidth=2,
        label=f"{label_a}: logged val NLL",
    )
    axes[0, 0].plot(
        stats_a["steps"],
        stats_a["mean_target_nll"],
        color=COMPARE_COLORS["a_2"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label=f"{label_a}: answer-token NLL",
    )
    axes[0, 0].plot(
        val_steps_b,
        val_nll_b,
        color=COMPARE_COLORS["b_acc"],
        marker="o",
        linewidth=2,
        label=f"{label_b}: logged val NLL",
    )
    axes[0, 0].plot(
        stats_b["steps"],
        stats_b["mean_target_nll"],
        color=COMPARE_COLORS["b_1"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label=f"{label_b}: answer-token NLL",
    )
    axes[0, 0].set_title("Validation loss")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("NLL")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)

    axes[0, 1].plot(
        val_acc_steps_a,
        val_acc_a,
        color=COMPARE_COLORS["a_acc"],
        marker="o",
        linewidth=2,
        label=label_a,
    )
    axes[0, 1].plot(
        val_acc_steps_b,
        val_acc_b,
        color=COMPARE_COLORS["b_acc"],
        marker="o",
        linewidth=2,
        label=label_b,
    )
    axes[0, 1].axhline(0.5, color="gray", linestyle=":", linewidth=1)
    axes[0, 1].set_title("Validation accuracy")
    axes[0, 1].set_xlabel("Optimizer step")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc="lower right")

    axes[1, 0].plot(
        stats_a["steps"],
        stats_a["confident_wrong_count"],
        color=COMPARE_COLORS["a_acc"],
        marker="o",
        linewidth=2,
        label=f"{label_a}: confident wrong",
    )
    axes[1, 0].plot(
        stats_a["steps"],
        stats_a["mean_abs_margin"],
        color=COMPARE_COLORS["a_2"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label=f"{label_a}: mean |margin|",
    )
    axes[1, 0].plot(
        stats_b["steps"],
        stats_b["confident_wrong_count"],
        color=COMPARE_COLORS["b_acc"],
        marker="o",
        linewidth=2,
        label=f"{label_b}: confident wrong",
    )
    axes[1, 0].plot(
        stats_b["steps"],
        stats_b["mean_abs_margin"],
        color=COMPARE_COLORS["b_1"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label=f"{label_b}: mean |margin|",
    )
    axes[1, 0].set_title("Confidence and hard mistakes")
    axes[1, 0].set_xlabel("Optimizer step")
    axes[1, 0].set_ylabel("Diagnostic summary")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc="upper left", fontsize=8)

    axes[1, 1].plot(
        stats_a["steps"],
        stats_a["prediction_mismatch_count"],
        color=COMPARE_COLORS["a_acc"],
        marker="o",
        linewidth=2,
        label=f"{label_a}: diag/gen mismatch",
    )
    axes[1, 1].plot(
        stats_a["steps"],
        stats_a["low_margin_share"],
        color=COMPARE_COLORS["a_2"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label=f"{label_a}: low-margin share",
    )
    axes[1, 1].plot(
        stats_b["steps"],
        stats_b["prediction_mismatch_count"],
        color=COMPARE_COLORS["b_acc"],
        marker="o",
        linewidth=2,
        label=f"{label_b}: diag/gen mismatch",
    )
    axes[1, 1].plot(
        stats_b["steps"],
        stats_b["low_margin_share"],
        color=COMPARE_COLORS["b_1"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label=f"{label_b}: low-margin share",
    )
    axes[1, 1].set_title("Boundary behavior")
    axes[1, 1].set_xlabel("Optimizer step")
    axes[1, 1].set_ylabel("Diagnostic summary")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc="upper left", fontsize=8)

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_margin_panel(
    *,
    run_dir: Path,
    output_path: Path,
) -> None:
    payloads = _load_margin_payloads(run_dir)
    panel_keys = [
        f"{index}:{example['id']}:{example['target']}"
        for index, example in enumerate(payloads[0]["examples"])
    ]

    if not panel_keys:
        raise ValueError(f"No diagnostic examples found in {run_dir}")

    example_traces: dict[str, dict] = {}
    for index, example in enumerate(payloads[0]["examples"]):
        panel_key = panel_keys[index]
        example_traces[panel_key] = {
            "position": index + 1,
            "id": str(example["id"]),
            "target": str(example["target"]),
            "prompt_preview": str(example["prompt_preview"]),
            "steps": [],
            "margins": [],
        }

    for payload in payloads:
        for index, example in enumerate(payload["examples"]):
            panel_key = panel_keys[index]
            trace = example_traces[panel_key]
            trace["steps"].append(int(payload["step"]))
            trace["margins"].append(float(example["margin_1_minus_2"]))

    n_examples = len(example_traces)
    ncols = min(3, n_examples)
    nrows = math.ceil(n_examples / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle("Fixed-Panel log p(1) - log p(2) Margins", fontsize=14)

    for axis, panel_key in zip(axes.flat, panel_keys, strict=False):
        trace = example_traces[panel_key]
        color = MARGIN_COLORS.get(trace["target"], "#2ca02c")
        axis.plot(trace["steps"], trace["margins"], color=color, marker="o", linewidth=2)
        axis.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        short_id = trace["id"][:8]
        axis.set_title(f"#{trace['position']} | {short_id} | target={trace['target']}")
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("log p(1) - log p(2)")
        axis.grid(True, alpha=0.3)
        preview = textwrap.fill(trace["prompt_preview"], width=48)
        axis.text(
            0.02,
            0.04,
            preview,
            transform=axis.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    for axis in axes.flat[n_examples:]:
        axis.axis("off")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-run-a",
        default=None,
        help="First run directory for curve comparison",
    )
    parser.add_argument("--label-a", default="Run A", help="Label for --compare-run-a")
    parser.add_argument(
        "--compare-run-b",
        default=None,
        help="Second run directory for curve comparison",
    )
    parser.add_argument("--label-b", default="Run B", help="Label for --compare-run-b")
    parser.add_argument(
        "--margin-run",
        default=None,
        help="Run directory containing val_diagnostics",
    )
    parser.add_argument(
        "--compare-full-run-a",
        default=None,
        help="First full-val diagnostic run directory for comparison",
    )
    parser.add_argument(
        "--compare-full-run-b",
        default=None,
        help="Second full-val diagnostic run directory for comparison",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_run_a and args.compare_run_b:
        plot_prediction_balance_comparison(
            run_a=Path(args.compare_run_a),
            label_a=args.label_a,
            run_b=Path(args.compare_run_b),
            label_b=args.label_b,
            output_path=output_dir / "prediction_balance_comparison.png",
        )
        plot_loss_curve_comparison(
            run_a=Path(args.compare_run_a),
            label_a=args.label_a,
            run_b=Path(args.compare_run_b),
            label_b=args.label_b,
            output_path=output_dir / "loss_curve_comparison.png",
        )

    if args.margin_run:
        margin_run = Path(args.margin_run)
        plot_diagnostic_overview(
            run_dir=margin_run,
            output_path=output_dir / "diagnostic_overview.png",
        )
        plot_full_val_nll_breakdown(
            run_dir=margin_run,
            output_path=output_dir / "full_val_nll_breakdown.png",
        )
        plot_hard_example_trajectories(
            run_dir=margin_run,
            output_path=output_dir / "hard_example_trajectories.png",
        )
        plot_margin_panel(
            run_dir=margin_run,
            output_path=output_dir / "margin_panel.png",
        )
        if (margin_run / "train_panel_diagnostics").exists():
            plot_train_panel_overview(
                run_dir=margin_run,
                output_path=output_dir / "train_panel_overview.png",
            )

    if args.compare_full_run_a and args.compare_full_run_b:
        plot_full_val_diagnostic_comparison(
            run_a=Path(args.compare_full_run_a),
            label_a=args.label_a,
            run_b=Path(args.compare_full_run_b),
            label_b=args.label_b,
            output_path=output_dir / "full_val_diagnostic_comparison.png",
        )


if __name__ == "__main__":
    main()
