"""Plot cross-evaluation results across experiments.

Generates several summary plots from completed experiment results:
1. Before/after bar charts for each experiment's cross-evals
2. Model generalisation heatmap (trained vs eval performance)
3. Cross-eval learning curves over training
4. Flipped vs normal comparison

Usage:
    python -m scripts.plot_cross_evals [--results-dir results] [--output-dir plots]
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sgtr_rl.logging_setup import setup_logging

matplotlib.rcParams.update({"font.size": 10})

logger = logging.getLogger(__name__)

# Display names for benchmark keys
_BENCH_DISPLAY = {
    "ind_rec_xeval": "IND Rec",
    "pw_pref_xeval": "PW Pref",
    "ind_pref_xeval": "IND Pref",
    "pw_rec_xeval": "PW Rec",
    "wikisum_pw_rec_xeval": "Wikisum",
    "pku_pw_rec_xeval": "PKU",
    "bigcode_pw_rec_xeval": "BigCode",
    "mmlu_20": "MMLU (n=20)",
    "mmlu_500": "MMLU (n=500)",
}

# Group benchmarks by type for plotting
_CROSS_OP_EVALS = ["ind_rec_xeval", "pw_rec_xeval", "pw_pref_xeval", "ind_pref_xeval"]
_CROSS_DATASET_EVALS = ["wikisum_pw_rec_xeval", "pku_pw_rec_xeval", "bigcode_pw_rec_xeval"]
_ALL_CROSS_EVALS = _CROSS_OP_EVALS + _CROSS_DATASET_EVALS

# Short names for "other" models
_MODEL_DISPLAY = {
    "qwen-2.5-7b": "Qwen-7B",
    "haiku-3.5": "Haiku-3.5",
    "gpt-4o": "GPT-4o",
    "ll-3.1-70b": "Llama-70B",
    "opus-4.1": "Opus-4.1",
}


def _config_train_files(config: dict) -> list[str]:
    data = config.get("data", {})
    train_files = data.get("train_files") or []
    if train_files:
        return list(train_files)
    train_file = data.get("train_file", "")
    return [train_file] if train_file else []


def load_experiment_metrics(results_dir: Path) -> dict:
    """Load metrics from all experiment result directories.

    Returns:
        Dict mapping experiment name -> list of metric records.
    """
    experiments = {}
    for d in sorted(results_dir.iterdir()):
        if not d.is_dir():
            continue
        metrics_path = d / "metrics" / "metrics.jsonl"
        config_path = d / "config.yaml"
        if not metrics_path.exists():
            continue

        # Extract experiment name (before the timestamp)
        name = d.name.rsplit("__", 1)[0]

        records = []
        with open(metrics_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        config = {}
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        experiments[name] = {"records": records, "config": config, "dir": d}

    return experiments


def extract_benchmark_series(records: list[dict]) -> dict:
    """Extract benchmark accuracy timeseries from metric records.

    Returns:
        Dict mapping benchmark_name -> {"steps": [...], "values": [...]}
    """
    series = defaultdict(lambda: {"steps": [], "values": []})
    for r in records:
        step = r["step"]
        for key, val in r.items():
            m = re.match(r"benchmark/(.+)/accuracy$", key)
            if m:
                name = m.group(1)
                series[name]["steps"].append(step)
                series[name]["values"].append(val)
    return dict(series)


def extract_val_series(records: list[dict]) -> dict:
    """Extract val accuracy and nll timeseries."""
    series = {"steps": [], "accuracy": [], "nll": []}
    for r in records:
        if "val/accuracy" in r:
            series["steps"].append(r["step"])
            series["accuracy"].append(r["val/accuracy"])
            if "val/nll" in r:
                series["nll"].append(r["val/nll"])
    return series


def extract_train_series(records: list[dict]) -> dict:
    """Extract training accuracy and nll timeseries."""
    series = {"steps": [], "accuracy": [], "nll": []}
    for r in records:
        if "train/nll" in r:
            series["steps"].append(r["step"])
            series["nll"].append(r["train/nll"])
            series["accuracy"].append(r.get("train/accuracy", 0))
    return series


def get_experiment_info(config: dict) -> dict:
    """Extract key info from experiment config."""
    data = config.get("data", {})
    generators = data.get("generator_models", [])
    if not generators:
        other_model = "unknown"
    elif len(generators) == 1:
        other_model = generators[0]
    else:
        other_model = "multiple models"
    # Detect format from train file path
    train_files = _config_train_files(config)
    train_file = train_files[0] if train_files else ""
    if "_pw/" in train_file or "_pw." in train_file:
        fmt = "PW"
    elif "_ind/" in train_file or "_ind." in train_file:
        fmt = "IND"
    else:
        fmt = "?"

    return {
        "other_model": other_model,
        "format": fmt,
        "description": config.get("description", ""),
    }


def plot_before_after(experiments: dict, output_dir: Path):
    """Plot before/after bar charts for each experiment's cross-evals."""
    for exp_name, exp_data in sorted(experiments.items()):
        bench = extract_benchmark_series(exp_data["records"])
        info = get_experiment_info(exp_data["config"])

        # Filter to cross-evals that exist for this experiment
        eval_names = [e for e in _ALL_CROSS_EVALS if e in bench]
        if not eval_names:
            continue

        baselines = []
        finals = []
        labels = []
        for e in eval_names:
            vals = bench[e]["values"]
            if len(vals) >= 2:
                baselines.append(vals[0])
                finals.append(vals[-1])
                labels.append(_BENCH_DISPLAY.get(e, e))

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, baselines, width, label="Before training (epoch 0)",
                       color="#94a3b8", edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(x + width / 2, finals, width, label="After training (epoch 20)",
                       color="#3b82f6", edgecolor="white", linewidth=0.5)

        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8, color="#64748b")
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8, color="#1e40af")

        ax.axhline(y=0.5, color="#cbd5e1", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_ylabel("Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper left", fontsize=9)

        model_str = _MODEL_DISPLAY.get(info['other_model'], info['other_model'])
        ax.set_title(f"{exp_name}\n{info['format']} Rec vs {model_str}")

        plt.tight_layout()
        fig.savefig(output_dir / f"before_after_{exp_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved before_after_{exp_name}.png")


def plot_model_heatmap(experiments: dict, output_dir: Path):
    """Plot heatmap of final cross-eval accuracy by training model.

    Only includes PW rec experiments.
    """
    pw_exps = {}
    for name, data in experiments.items():
        info = get_experiment_info(data["config"])
        if info["format"] == "PW":
            pw_exps[info["other_model"]] = data

    if len(pw_exps) < 2:
        logger.info("Not enough PW rec experiments for heatmap")
        return

    # Collect eval names present across all experiments
    all_evals = set()
    for data in pw_exps.values():
        bench = extract_benchmark_series(data["records"])
        all_evals.update(e for e in _ALL_CROSS_EVALS if e in bench)

    eval_names = [e for e in _ALL_CROSS_EVALS if e in all_evals]
    model_names = sorted(pw_exps.keys(), key=lambda m: _MODEL_DISPLAY.get(m, m))

    # Build matrices for baseline and final
    baseline_matrix = np.full((len(model_names), len(eval_names)), np.nan)
    final_matrix = np.full((len(model_names), len(eval_names)), np.nan)
    improvement_matrix = np.full((len(model_names), len(eval_names)), np.nan)

    for i, model in enumerate(model_names):
        bench = extract_benchmark_series(pw_exps[model]["records"])
        for j, eval_name in enumerate(eval_names):
            if eval_name in bench:
                vals = bench[eval_name]["values"]
                if len(vals) >= 2:
                    baseline_matrix[i, j] = vals[0]
                    final_matrix[i, j] = vals[-1]
                    improvement_matrix[i, j] = vals[-1] - vals[0]

    model_labels = [_MODEL_DISPLAY.get(m, m) for m in model_names]
    eval_labels = [_BENCH_DISPLAY.get(e, e) for e in eval_names]

    # Plot 1: Final accuracy heatmap
    fig, ax = plt.subplots(
        figsize=(max(8, len(eval_labels) * 1.2), max(4, len(model_labels) * 0.8))
    )
    im = ax.imshow(final_matrix, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(eval_labels)))
    ax.set_xticklabels(eval_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)

    # Annotate cells
    for i in range(len(model_labels)):
        for j in range(len(eval_labels)):
            val = final_matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=color)

    ax.set_title("Final Cross-Eval Accuracy\n(PW Rec, by training 'other' model)")
    ax.set_xlabel("Cross-Eval Benchmark")
    ax.set_ylabel("Trained against")
    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "heatmap_final_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap_final_accuracy.png")

    # Plot 2: Improvement heatmap
    fig, ax = plt.subplots(
        figsize=(max(8, len(eval_labels) * 1.2), max(4, len(model_labels) * 0.8))
    )
    max_imp = np.nanmax(np.abs(improvement_matrix))
    im = ax.imshow(improvement_matrix, cmap="RdBu", vmin=-max_imp, vmax=max_imp, aspect="auto")

    ax.set_xticks(range(len(eval_labels)))
    ax.set_xticklabels(eval_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels)

    for i in range(len(model_labels)):
        for j in range(len(eval_labels)):
            val = improvement_matrix[i, j]
            if not np.isnan(val):
                sign = "+" if val > 0 else ""
                ax.text(j, i, f"{sign}{val:.0%}", ha="center", va="center",
                        fontsize=10, fontweight="bold")

    ax.set_title(
        "Improvement from Training (final - baseline)\n"
        "(PW Rec, by training 'other' model)"
    )
    ax.set_xlabel("Cross-Eval Benchmark")
    ax.set_ylabel("Trained against")
    fig.colorbar(im, ax=ax, label="Accuracy change", shrink=0.8)

    plt.tight_layout()
    fig.savefig(output_dir / "heatmap_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap_improvement.png")


def plot_cross_eval_curves(experiments: dict, output_dir: Path):
    """Plot cross-eval accuracy over training epochs for each experiment."""
    for exp_name, exp_data in sorted(experiments.items()):
        bench = extract_benchmark_series(exp_data["records"])
        info = get_experiment_info(exp_data["config"])

        eval_names = [e for e in _ALL_CROSS_EVALS if e in bench and len(bench[e]["values"]) > 1]
        if not eval_names:
            continue

        # Compute steps per epoch from config
        config = exp_data["config"]
        hp = config.get("hyperparameters", {})
        bs = hp.get("batch_size", hp.get("per_device_train_batch_size", 16))
        train_files = [Path(path) for path in _config_train_files(config) if Path(path).exists()]
        if train_files:
            n_train = 0
            for train_file in train_files:
                with open(train_file) as f:
                    n_train += sum(1 for _ in f)
            bpe = n_train // bs
        else:
            bpe = 10

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for i, e in enumerate(eval_names):
            steps = np.array(bench[e]["steps"])
            epochs = steps / bpe
            vals = bench[e]["values"]
            label = _BENCH_DISPLAY.get(e, e)
            ax.plot(epochs, vals, "o-", color=colors[i], markersize=5, linewidth=1.5, label=label)

        ax.axhline(y=0.5, color="#cbd5e1", linestyle="--", linewidth=0.8, label="Chance")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        model_str = _MODEL_DISPLAY.get(info['other_model'], info['other_model'])
        ax.set_title(f"Cross-Eval Learning Curves: {exp_name}\n{info['format']} Rec vs {model_str}")

        plt.tight_layout()
        fig.savefig(output_dir / f"curves_{exp_name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved curves_{exp_name}.png")



def plot_combined_overview(experiments: dict, output_dir: Path):
    """Single overview figure with all PW rec experiments' cross-eval results."""
    pw_exps = {}
    for name, data in experiments.items():
        info = get_experiment_info(data["config"])
        if info["format"] == "PW":
            pw_exps[name] = (data, info)

    if len(pw_exps) < 2:
        return

    eval_names = _CROSS_DATASET_EVALS + [e for e in _CROSS_OP_EVALS if any(
        e in extract_benchmark_series(d["records"]) for d, _ in pw_exps.values()
    )] + ["mmlu_20"]
    # Deduplicate while preserving order
    seen = set()
    unique_evals = []
    for e in eval_names:
        if e not in seen:
            seen.add(e)
            unique_evals.append(e)
    eval_names = unique_evals

    n_evals = len(eval_names)
    n_models = len(pw_exps)

    ncols = 2
    nrows = (n_evals + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 5), sharey=True)
    axes = axes.flatten()

    sorted_exps = sorted(
        pw_exps.items(),
        key=lambda item: _MODEL_DISPLAY.get(item[1][1]["other_model"], ""),
    )
    model_labels = [
        _MODEL_DISPLAY.get(info["other_model"], info["other_model"])
        for _, (_, info) in sorted_exps
    ]

    for j, eval_name in enumerate(eval_names):
        ax = axes[j]
        x = np.arange(n_models)
        width = 0.35

        baselines = []
        finals = []
        for _, (data, info) in sorted_exps:
            bench = extract_benchmark_series(data["records"])
            if eval_name in bench and len(bench[eval_name]["values"]) >= 2:
                baselines.append(bench[eval_name]["values"][0])
                finals.append(bench[eval_name]["values"][-1])
            else:
                baselines.append(np.nan)
                finals.append(np.nan)

        ax.bar(x - width / 2, baselines, width, color="#94a3b8", edgecolor="white",
               linewidth=0.5, label="Before" if j == 0 else "")
        ax.bar(x + width / 2, finals, width, color="#3b82f6", edgecolor="white",
               linewidth=0.5, label="After" if j == 0 else "")

        for i, (b, f) in enumerate(zip(baselines, finals)):
            if not np.isnan(b):
                # Use dark text for short bars, white for tall ones
                b_color = "white" if b > 0.15 else "#333333"
                b_va = "top" if b > 0.15 else "bottom"
                b_offset = -0.03 if b > 0.15 else 0.01
                ax.text(i - width / 2, b + b_offset, f"{b:.0%}", ha="center",
                        va=b_va, fontsize=10, fontweight="bold", color=b_color)
            if not np.isnan(f):
                f_color = "white" if f > 0.15 else "#333333"
                f_va = "top" if f > 0.15 else "bottom"
                f_offset = -0.03 if f > 0.15 else 0.01
                ax.text(i + width / 2, f + f_offset, f"{f:.0%}", ha="center",
                        va=f_va, fontsize=10, fontweight="bold", color=f_color)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=13)
        ax.set_title(
            _BENCH_DISPLAY.get(eval_name, eval_name),
            fontsize=15,
            fontweight="bold",
        )
        if j % ncols == 0:
            ax.set_ylabel("Accuracy", fontsize=13)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    # Hide unused axes
    for k in range(n_evals, len(axes)):
        axes[k].set_visible(False)

    # Legend in top-right subplot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#94a3b8", edgecolor="white", label="Before training"),
        Patch(facecolor="#3b82f6", edgecolor="white", label="After training"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper left", fontsize=12)

    fig.suptitle(
        "Eval Generalisation: Llama-3.1-8B on ShareGPT (PW Rec)\n"
        "x-axis indicates 'other' model trained against",
        fontsize=18,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    fig.savefig(output_dir / "overview_by_model.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved overview_by_model.png")


def plot_format_comparison(experiments: dict, output_dir: Path):
    """2x2 heatmap: trained on PW/IND vs evaluated on PW/IND.

    Diagonal = val accuracy (in-distribution), off-diagonal = cross-eval.
    """
    by_model: dict[str, dict[str, tuple]] = {}
    for name, data in experiments.items():
        info = get_experiment_info(data["config"])
        model = info["other_model"]
        if model not in by_model:
            by_model[model] = {}
        by_model[model][info["format"]] = (name, data, info)

    for model, formats in by_model.items():
        if "PW" not in formats or "IND" not in formats:
            continue

        pw_name, pw_data, pw_info = formats["PW"]
        ind_name, ind_data, ind_info = formats["IND"]

        pw_bench = extract_benchmark_series(pw_data["records"])
        ind_bench = extract_benchmark_series(ind_data["records"])
        pw_val = extract_val_series(pw_data["records"])
        ind_val = extract_val_series(ind_data["records"])

        # Build 2x2 matrices: rows = trained on, cols = evaluated on
        # [PW→PW, PW→IND]
        # [IND→PW, IND→IND]
        matrix = np.full((2, 2), np.nan)
        baseline_matrix = np.full((2, 2), np.nan)

        # Diagonal: val accuracy (final + baseline)
        if pw_val["accuracy"]:
            matrix[0, 0] = pw_val["accuracy"][-1]
            baseline_matrix[0, 0] = pw_val["accuracy"][0]
        if ind_val["accuracy"]:
            matrix[1, 1] = ind_val["accuracy"][-1]
            baseline_matrix[1, 1] = ind_val["accuracy"][0]

        # Off-diagonal: cross-eval accuracy (final + baseline)
        if "ind_rec_xeval" in pw_bench and pw_bench["ind_rec_xeval"]["values"]:
            matrix[0, 1] = pw_bench["ind_rec_xeval"]["values"][-1]
            baseline_matrix[0, 1] = pw_bench["ind_rec_xeval"]["values"][0]
        if "pw_rec_xeval" in ind_bench and ind_bench["pw_rec_xeval"]["values"]:
            matrix[1, 0] = ind_bench["pw_rec_xeval"]["values"][-1]
            baseline_matrix[1, 0] = ind_bench["pw_rec_xeval"]["values"][0]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="equal")

        labels = ["PW Rec", "IND Rec"]
        ax.set_xticks(range(2))
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_yticks(range(2))
        ax.set_yticklabels(labels, fontsize=14)
        ax.set_xlabel("Evaluated on", fontsize=14, fontweight="bold")
        ax.set_ylabel("Trained on", fontsize=14, fontweight="bold")

        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                base = baseline_matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.55 else "black"
                    if not np.isnan(base):
                        diff = val - base
                        sign = "+" if diff >= 0 else ""
                        diff_str = f"\n({sign}{diff:.0%})"
                    else:
                        diff_str = ""
                    ax.text(j, i, f"{val:.0%}{diff_str}", ha="center", va="center",
                            fontsize=16, fontweight="bold", color=color)

        fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)

        model_str = _MODEL_DISPLAY.get(model, model)
        ax.set_title(f"PW vs IND Transfer: Llama-3.1-8B vs {model_str}\nShareGPT",
                     fontsize=16, fontweight="bold")

        plt.tight_layout()
        fig.savefig(output_dir / f"format_comparison_{model}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved format_comparison_{model}.png")


def plot_mmlu_summary(experiments: dict, output_dir: Path):
    """Plot MMLU accuracy over training for all experiments, plus before/after summary."""
    # Collect MMLU data
    exp_mmlu = {}
    for name, data in sorted(experiments.items()):
        bench = extract_benchmark_series(data["records"])
        info = get_experiment_info(data["config"])

        mmlu_data = {}
        for key in ["mmlu_20", "mmlu_500"]:
            if key in bench and len(bench[key]["values"]) >= 2:
                mmlu_data[key] = bench[key]
        if mmlu_data:
            exp_mmlu[name] = {"bench": mmlu_data, "info": info, "config": data["config"]}

    if not exp_mmlu:
        logger.info("No MMLU data found")
        return

    # Plot 1: MMLU-20 learning curves (all experiments on one plot)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, (name, ed) in enumerate(exp_mmlu.items()):
        if "mmlu_20" not in ed["bench"]:
            continue
        info = ed["info"]
        series = ed["bench"]["mmlu_20"]

        # Compute epochs
        config = ed["config"]
        hp = config.get("hyperparameters", {})
        bs = hp.get("batch_size", hp.get("per_device_train_batch_size", 16))
        train_files = [Path(path) for path in _config_train_files(config) if Path(path).exists()]
        if train_files:
            n_train = 0
            for train_file in train_files:
                with open(train_file) as f:
                    n_train += sum(1 for _ in f)
            bpe = max(n_train // bs, 1)
        else:
            bpe = 10

        epochs = np.array(series["steps"]) / bpe
        vals = series["values"]

        model_str = _MODEL_DISPLAY.get(info["other_model"], info["other_model"])
        label = f"{info['format']} vs {model_str}"

        ax.plot(epochs, vals, "o-", color=colors[i % 10], markersize=3,
                linewidth=1.2, alpha=0.8, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMLU-20 Accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title("MMLU-20 Accuracy During Training (all experiments)")

    plt.tight_layout()
    fig.savefig(output_dir / "mmlu_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mmlu_learning_curves.png")

    # Plot 2: MMLU before/after bar chart
    names = []
    baselines_20 = []
    finals_20 = []
    baselines_500 = []
    finals_500 = []

    for name, ed in exp_mmlu.items():
        info = ed["info"]
        model_str = _MODEL_DISPLAY.get(info["other_model"], info["other_model"])
        label = f"{info['format']} vs {model_str}"
        names.append(label)

        if "mmlu_20" in ed["bench"]:
            vals = ed["bench"]["mmlu_20"]["values"]
            baselines_20.append(vals[0])
            finals_20.append(vals[-1])
        else:
            baselines_20.append(np.nan)
            finals_20.append(np.nan)

        if "mmlu_500" in ed["bench"]:
            vals = ed["bench"]["mmlu_500"]["values"]
            baselines_500.append(vals[0])
            finals_500.append(vals[-1])
        else:
            baselines_500.append(np.nan)
            finals_500.append(np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, baselines, finals, title in [
        (axes[0], baselines_20, finals_20, "MMLU-20"),
        (axes[1], baselines_500, finals_500, "MMLU-500"),
    ]:
        width = 0.35
        valid = [i for i in range(len(baselines)) if not np.isnan(baselines[i])]
        if not valid:
            ax.set_visible(False)
            continue

        b = [baselines[i] for i in valid]
        f = [finals[i] for i in valid]
        n = [names[i] for i in valid]
        xv = np.arange(len(valid))

        bars1 = ax.bar(xv - width / 2, b, width, label="Before (epoch 0)",
                       color="#94a3b8", edgecolor="white", linewidth=0.5)
        bars2 = ax.bar(xv + width / 2, f, width, label="After (epoch 20)",
                       color="#3b82f6", edgecolor="white", linewidth=0.5)

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=7, color="#64748b")
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=7, color="#1e40af")

        ax.set_xticks(xv)
        ax.set_xticklabels(n, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 0.85)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("MMLU Impact: Does SGTR Training Affect General Capabilities?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "mmlu_before_after.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mmlu_before_after.png")


def main():
    parser = argparse.ArgumentParser(description="Plot cross-evaluation results")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output plots (default: results/batch_<exp_nums>/plots)",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Filter to specific experiment numbers (e.g. 15 16)",
    )
    args = parser.parse_args()

    setup_logging("plot_cross_evals")

    results_dir = Path(args.results_dir)

    experiments = load_experiment_metrics(results_dir)

    if args.experiments:
        filtered = {}
        for name, data in experiments.items():
            num = name.split("_")[0]
            if num in args.experiments:
                filtered[name] = data
        experiments = filtered

    # Default output dir based on experiment numbers
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        exp_nums = sorted(name.split("_")[0] for name in experiments)
        if exp_nums:
            batch_name = f"batch_{exp_nums[0]}-{exp_nums[-1]}"
        else:
            batch_name = "batch"
        output_dir = results_dir / f"{batch_name}_plots" / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loaded {len(experiments)} experiments: {sorted(experiments.keys())}")

    plot_before_after(experiments, output_dir)
    plot_model_heatmap(experiments, output_dir)
    plot_cross_eval_curves(experiments, output_dir)
    plot_combined_overview(experiments, output_dir)
    plot_format_comparison(experiments, output_dir)
    plot_mmlu_summary(experiments, output_dir)

    logger.info(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
