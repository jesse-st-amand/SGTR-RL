"""Generate a summary plot for a training run."""

import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Short names for well-known models
_MODEL_SHORT = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen2-1.5B": "Qwen2-1.5B",
}

_GENERATOR_SHORT = {
    "qwen-2.5-7b": "qwen-2.5-7b",
    "ll-3.1-8b": "ll-3.1-8b",
}

# Prettify benchmark metric keys for legend labels
_BENCH_DISPLAY = {
    "mmlu_20": "MMLU (20-sample)",
    "mmlu_500": "MMLU (500-sample)",
    "mmlu_500_cot": "MMLU (500-sample, CoT)",
    "mmlu_2000": "MMLU (2000-sample)",
    "mmlu_2000_cot": "MMLU (2000-sample, CoT)",
    # Legacy names from older runs
    "mmlu_canary": "MMLU (20-sample)",
    "mmlu_full_nocot": "MMLU (500-sample)",
    "mmlu_full_cot": "MMLU (500-sample, CoT)",
    # SGTR cross-eval benchmarks
    "cross_ind_val": "SGTR IND (cross-eval)",
    "cross_ind_val_cot": "SGTR IND CoT (cross-eval)",
    "cross_pw_val": "SGTR PW (cross-eval)",
}

# Default hyperparams (only show overrides in title)
_DEFAULTS = {
    "learning_rate": 5e-5,
    "batch_size": 16,
    "lora_rank": 32,
    "num_epochs": 20,
    "seed": 42,
}


def _build_title(config: dict) -> str:
    """Build a descriptive title from the frozen config.yaml."""
    exp_name = config.get("experiment_name", "")
    # Extract experiment number
    match = re.match(r"(\d+)", exp_name)
    exp_num = match.group(1) if match else exp_name

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    hp = config.get("hyperparameters", {})

    model_name = model_cfg.get("name", "")
    model_short = _MODEL_SHORT.get(model_name, model_name.split("/")[-1])

    generators = data_cfg.get("generator_models", [])
    gen_short = ", ".join(_GENERATOR_SHORT.get(g, g) for g in generators) if generators else "?"

    algorithm = config.get("algorithm", "?").upper()
    # Non-default params
    param_parts = []
    lr = hp.get("learning_rate", _DEFAULTS["learning_rate"])
    if lr != _DEFAULTS["learning_rate"]:
        param_parts.append(f"lr={lr}")
    rank = model_cfg.get("lora_rank", _DEFAULTS["lora_rank"])
    if rank != _DEFAULTS["lora_rank"]:
        param_parts.append(f"rank={rank}")
    bs = hp.get("batch_size", _DEFAULTS["batch_size"])
    if bs != _DEFAULTS["batch_size"]:
        param_parts.append(f"bs={bs}")

    dataset = data_cfg.get("dataset", "").capitalize() or "?"
    base = f"Exp {exp_num}: {model_short} (self) vs {gen_short}, {dataset}, pairwise, {algorithm}"
    if param_parts:
        base += ", " + ", ".join(param_parts)
    return base


def _smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    # Pad to avoid edge effects
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")


def generate_summary_plot(run_dir: str | Path) -> Path:
    """Generate a 3-subplot summary figure for a training run.

    Args:
        run_dir: Path to the run directory containing metrics/ and config.yaml.

    Returns:
        Path to the saved summary.png.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics" / "metrics.jsonl"
    config_path = run_dir / "config.yaml"

    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics file found: {metrics_path}")

    # Load metrics
    records = []
    with open(metrics_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Load config for title and batches_per_epoch
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    title = _build_title(config)

    # Determine batches_per_epoch from data
    # Find steps where val/nll is reported (epoch boundaries)
    val_steps = sorted(set(r["step"] for r in records if "val/nll" in r))

    if len(val_steps) >= 2:
        # Epoch boundaries are at val_steps[0]=0, val_steps[1]=bpe, val_steps[2]=2*bpe, ...
        batches_per_epoch = val_steps[1] - val_steps[0]
    else:
        # Fallback: compute from config
        hp = config.get("hyperparameters", {})
        bs = hp.get("batch_size", 16)
        # Try to count training samples from train file
        train_file = config.get("data", {}).get("train_file", "")
        if train_file and Path(train_file).exists():
            with open(train_file) as f:
                n_train = sum(1 for _ in f)
            batches_per_epoch = n_train // bs
        else:
            batches_per_epoch = 10  # fallback

    # Extract series
    train_steps = []
    train_nll = []
    train_acc = []
    val_step_list = []
    val_nll_list = []
    val_acc_list = []
    # benchmark/<name>/accuracy -> {name: (steps[], values[])}
    bench_data: dict[str, tuple[list, list]] = {}

    for r in records:
        step = r["step"]
        if "train/nll" in r:
            train_steps.append(step)
            train_nll.append(r["train/nll"])
            train_acc.append(r.get("train/accuracy", 0))
        if "val/nll" in r:
            val_step_list.append(step)
            val_nll_list.append(r["val/nll"])
            val_acc_list.append(r.get("val/accuracy", 0))
        # Collect benchmark metrics
        for key, val in r.items():
            m = re.match(r"benchmark/(.+)/accuracy$", key)
            if m:
                name = m.group(1)
                if name not in bench_data:
                    bench_data[name] = ([], [])
                bench_data[name][0].append(step)
                bench_data[name][1].append(val)

    # Convert to epochs (float)
    bpe = batches_per_epoch
    train_epochs = np.array(train_steps) / bpe
    val_epochs = np.array(val_step_list) / bpe
    bench_epochs = {name: (np.array(s) / bpe, v) for name, (s, v) in bench_data.items()}

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title, fontsize=11, fontweight="bold")

    ax = axes[0]
    train_nll_arr = np.array(train_nll)
    ax.plot(train_epochs, train_nll_arr, alpha=0.3, color="C0", lw=0.8)
    ax.plot(
        train_epochs, _smooth(train_nll_arr),
        color="C0", lw=1.5, label="train NLL (smoothed)",
    )
    ax.plot(
        val_epochs, val_nll_list, "o-",
        color="C1", markersize=4, lw=1.5, label="val NLL",
    )
    ax.set_ylabel("NLL Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    train_acc_arr = np.array(train_acc)
    ax.plot(train_epochs, train_acc_arr, alpha=0.3, color="C0", lw=0.8)
    ax.plot(
        train_epochs, _smooth(train_acc_arr),
        color="C0", lw=1.5, label="train acc (smoothed)",
    )
    ax.plot(
        val_epochs, val_acc_list, "o-",
        color="C1", markersize=4, lw=1.5, label="val acc",
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    colors = ["C2", "C3", "C4", "C5", "C6"]
    for i, (name, (epochs, values)) in enumerate(sorted(bench_epochs.items())):
        color = colors[i % len(colors)]
        display_name = _BENCH_DISPLAY.get(name, name)
        ax.plot(epochs, values, "o-", color=color, markersize=4, linewidth=1.5, label=display_name)
    ax.set_ylabel("Benchmark Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = run_dir / "summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Summary plot saved to {out_path}")
    return out_path
