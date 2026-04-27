"""Generate per-model summary plots for SFT experiments.

Each model gets a figure with:
- Top subplot: train accuracy and val accuracy over epochs
- Bottom subplot: bar chart of before/after training accuracy for each eval

Usage:
    python -m scripts.plot_summary
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Map from result dir pattern to display name
RUNS = {
    "01_sft_pw_vs_qwen__20260305_194916": (
        "SFT",
        "PW",
        "ShareGPT",
        "Llama-3.1-8B",
        "Qwen-2.5-7B",
    ),
    "01_sft_pw_vs_haiku_3_5__20260305_214551": (
        "SFT",
        "PW",
        "ShareGPT",
        "Llama-3.1-8B",
        "Haiku-3.5",
    ),
    "01_sft_pw_vs_gpt_4o__20260305_222402": (
        "SFT",
        "PW",
        "ShareGPT",
        "Llama-3.1-8B",
        "GPT-4o",
    ),
    "01_sft_pw_vs_ll_3_1_70b__20260305_225808": (
        "SFT",
        "PW",
        "ShareGPT",
        "Llama-3.1-8B",
        "Llama-3.1-70B",
    ),
    "01_sft_pw_vs_opus_4_1__20260305_233155": (
        "SFT",
        "PW",
        "ShareGPT",
        "Llama-3.1-8B",
        "Opus-4.1",
    ),
    "02_sft_ind_vs_qwen__20260305_213803": (
        "SFT",
        "IND",
        "ShareGPT",
        "Llama-3.1-8B",
        "Qwen-2.5-7B",
    ),
}

# Ordered eval keys per training format
EVAL_ORDER_PW = [
    "xeval_dataset_bigcodebench",
    "xeval_dataset_pku",
    "xeval_dataset_wikisum",
    "xeval_task_pref_pw",
    "xeval_tag_at_pw",
    "xeval_format_ind",
    "mmlu_2000",
]

EVAL_ORDER_IND = [
    "xeval_dataset_bigcodebench",
    "xeval_dataset_pku",
    "xeval_dataset_wikisum",
    "xeval_task_pref_ind",
    "xeval_tag_at_ind",
    "xeval_format_pw",
    "mmlu_2000",
]

# Readable names for benchmark keys
EVAL_LABELS = {
    "xeval_dataset_bigcodebench": "BigCodeBench",
    "xeval_dataset_pku": "PKU SafeRLHF",
    "xeval_dataset_wikisum": "WikiSum",
    "xeval_format_ind": "Format:\nIndividual",
    "xeval_format_pw": "Format:\nPairwise",
    "xeval_task_pref_pw": "Task:\nPreference",
    "xeval_task_pref_ind": "Task:\nPreference",
    "xeval_tag_at_pw": "Tag:\nAssistant",
    "xeval_tag_at_ind": "Tag:\nAssistant",
    "mmlu_2000": "MMLU\n(2000-sample)",
}


def load_metrics(metrics_path: Path) -> list[dict]:
    with open(metrics_path) as f:
        return [json.loads(line) for line in f]


def extract_train_accuracy(metrics: list[dict]) -> tuple[list[int], list[float]]:
    """Extract per-step train accuracy, smoothed to epoch-level."""
    steps, accs = [], []
    for m in metrics:
        if "train/accuracy" in m:
            steps.append(m["step"])
            accs.append(m["train/accuracy"])
    if not steps:
        return [], []

    # Find epoch size from val accuracy steps
    val_steps = sorted(m["step"] for m in metrics if "val/accuracy" in m)
    if len(val_steps) >= 2:
        epoch_size = val_steps[1] - val_steps[0]
    else:
        epoch_size = 10

    # Average train accuracy per epoch
    epoch_steps, epoch_accs = [], []
    for epoch_start in range(steps[0], steps[-1] + 1, epoch_size):
        epoch_end = epoch_start + epoch_size
        epoch_vals = [a for s, a in zip(steps, accs) if epoch_start < s <= epoch_end]
        if epoch_vals:
            epoch_steps.append(epoch_end // epoch_size)
            epoch_accs.append(np.mean(epoch_vals))

    return epoch_steps, epoch_accs


def extract_val_accuracy(metrics: list[dict]) -> tuple[list[int], list[float]]:
    """Extract val accuracy at epoch boundaries."""
    val_steps = sorted(m["step"] for m in metrics if "val/accuracy" in m)
    if len(val_steps) >= 2:
        epoch_size = val_steps[1] - val_steps[0]
    else:
        epoch_size = 10

    epochs, accs = [], []
    for m in metrics:
        if "val/accuracy" in m:
            epochs.append(m["step"] // epoch_size)
            accs.append(m["val/accuracy"])
    return epochs, accs


def extract_benchmark_before_after(
    metrics: list[dict],
) -> dict[str, tuple[float, float]]:
    """Extract first and last benchmark accuracy for each eval."""
    # Collect all benchmark accuracy values by key
    by_key: dict[str, list[tuple[int, float]]] = {}
    for m in metrics:
        step = m["step"]
        for k, v in m.items():
            if k.startswith("benchmark/") and k.endswith("/accuracy"):
                short = k.split("/")[1]
                if short not in EVAL_LABELS:
                    continue
                by_key.setdefault(short, []).append((step, v))

    result = {}
    for key, vals in by_key.items():
        vals.sort(key=lambda x: x[0])
        before = vals[0][1]
        after = vals[-1][1]
        result[key] = (before, after)
    return result


def plot_model(
    title: tuple[str, str, str, str, str],
    metrics: list[dict],
    output_path: Path,
):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [1, 1.2]}
    )
    algo, fmt, dataset, model, opponent = title
    fig.suptitle(
        f"{model} vs {opponent} ({algo})\n"
        f"Dataset: {dataset} | Format: {'Pairwise' if fmt == 'PW' else 'Individual'}"
        f" | Task: Recognition | Tag: User",
        fontsize=13, fontweight="bold",
    )

    # --- Top: Train + Val accuracy ---
    train_epochs, train_accs = extract_train_accuracy(metrics)
    val_epochs, val_accs = extract_val_accuracy(metrics)

    if train_epochs:
        ax_top.plot(train_epochs, train_accs, "o-", label="Train Acc (epoch avg)",
                    color="#2196F3", markersize=4, alpha=0.8)
    if val_epochs:
        ax_top.plot(val_epochs, val_accs, "s-", label="Val Acc",
                    color="#FF5722", markersize=5)

    ax_top.set_xlabel("Epoch")
    ax_top.set_ylabel("Accuracy")
    ax_top.set_ylim(-0.05, 1.05)
    if val_epochs:
        ax_top.set_xticks(range(val_epochs[0], val_epochs[-1] + 1))
    ax_top.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Chance")
    ax_top.legend(loc="lower right")
    ax_top.set_title("Training Progress")
    ax_top.grid(True, alpha=0.3)

    # --- Bottom: Before/After bar chart ---
    benchmarks = extract_benchmark_before_after(metrics)

    eval_order = EVAL_ORDER_IND if fmt == "IND" else EVAL_ORDER_PW
    ordered_keys = [k for k in eval_order if k in benchmarks]

    if ordered_keys:
        labels = [EVAL_LABELS[k] for k in ordered_keys]
        befores = [benchmarks[k][0] for k in ordered_keys]
        afters = [benchmarks[k][1] for k in ordered_keys]

        x = np.arange(len(labels))
        width = 0.35

        bars_before = ax_bot.bar(x - width / 2, befores, width, label="Before Training",
                                  color="#90CAF9", edgecolor="#1565C0", linewidth=0.5)
        bars_after = ax_bot.bar(x + width / 2, afters, width, label="After Training",
                                 color="#FF8A65", edgecolor="#BF360C", linewidth=0.5)

        # Value labels on bars
        for bar in bars_before:
            h = bar.get_height()
            ax_bot.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7)
        for bar in bars_after:
            h = bar.get_height()
            ax_bot.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                        f"{h:.0%}", ha="center", va="bottom", fontsize=7)

        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
        ax_bot.set_ylabel("Accuracy")
        ax_bot.set_ylim(0, 1.15)
        ax_bot.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
        ax_bot.legend(loc="upper right")
        ax_bot.set_title("Cross-Eval Accuracy (Before vs After Training)")
        ax_bot.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    results_dir = Path("results")
    output_dir = results_dir / "summary_sft_experiments"
    output_dir.mkdir(exist_ok=True)

    for run_name, title in RUNS.items():
        metrics_path = results_dir / run_name / "metrics" / "metrics.jsonl"
        if not metrics_path.exists():
            print(f"Skipping {run_name}: no metrics file")
            continue

        metrics = load_metrics(metrics_path)
        safe_name = run_name.split("__")[0]
        output_path = output_dir / f"{safe_name}.png"
        plot_model(title, metrics, output_path)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
