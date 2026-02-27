"""Generate all plots for the results writeup."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Allow running from repo root or analysis/
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.plotting import set_style
from analysis.utils import load_run, RESULTS_DIR


PLOT_DIR = Path(__file__).parent / "plots" / "writeup"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
C_BASELINE = "#888888"
C_GRPO = "#1f77b4"
C_SFT = "#e07b39"
CHANCE_COLOR = "red"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _savefig(fig, name: str):
    path = PLOT_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ── Plot 1: Summary bar chart ────────────────────────────────────────────────

def plot_summary_bar():
    set_style()

    labels = [
        "Baseline (IND CoT)",
        "Baseline (PW aug.)",
        "Baseline (PW raw)",
        "Exp 03: GRPO full",
        "Exp 04: GRPO overfit",
        "Exp 05: GRPO trivial (all 1)",
        "Exp 06: GRPO trivial (all 2)",
        "Exp 07: GRPO high temp",
        "Exp 10: SFT all-1s",
        "Exp 11: SFT all-2s",
        "Exp 12: SFT mixed",
        "SFT mixed (lr=1e-4, 30ep)",
        "SFT mixed (lr=5e-5, 50ep)",
        "SFT mixed (lr=1e-5, 50ep)",
    ]
    # Best accuracy for each (train or val, whichever is the relevant metric)
    accs = [
        47.5,   # baseline IND COT val
        42.4,   # baseline PW aug val
        32.5,   # baseline PW raw val
        51.2,   # exp 03 final running acc
        50.0,   # exp 04 stuck at ~50%
        92.8,   # exp 05 final running acc (100% per-batch from ep4)
        98.1,   # exp 06 final running acc (100% per-batch from ep2)
        54.2,   # exp 07 peak before collapse (epoch 5)
        50.0,   # exp 10 val acc (train 100%, val 50% -- all 1s)
        50.0,   # exp 11 val acc (train 100%, val 50% -- all 2s)
        51.6,   # exp 12 val acc at epoch 10
        100.0,  # SFT mixed lr=1e-4 train acc at ep 30
        100.0,  # SFT mixed lr=5e-5 train acc at ep 50
        87.5,   # SFT mixed lr=1e-5 train acc at ep 50 (14/16)
    ]
    colors = (
        [C_BASELINE] * 3
        + [C_GRPO] * 5
        + [C_SFT] * 6
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(labels))
    bars = ax.barh(y, accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(50, color=CHANCE_COLOR, linestyle=":", linewidth=1.2, alpha=0.7, label="chance (50%)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Summary: Best Accuracy Across All Experiments")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.legend(fontsize=9)

    # Annotate values
    for bar, val in zip(bars, accs):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)

    _savefig(fig, "summary_accuracy_bar.png")


# ── Plot 2: Trivial sanity (exp 05 + 06) ─────────────────────────────────────

def plot_trivial_success():
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, exp_dir, title in [
        (ax1, "05_trivial_sanity__20260224_020402", "Exp 05: All targets = 1"),
        (ax2, "06_trivial_sanity_2__20260224_022331", "Exp 06: All targets = 2"),
    ]:
        run = load_run(RESULTS_DIR / exp_dir)
        steps = [b.global_step for b in run.batches]
        accs = [b.acc for b in run.batches]
        running = [b.running_acc for b in run.batches]

        ax.scatter(steps, accs, alpha=0.3, s=12, color=C_GRPO, label="per batch")
        ax.plot(steps, running, color="C1", linewidth=2, label="cumulative")
        ax.axhline(0.5, color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="chance")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("GRPO Trivial Sanity Check: Pipeline Works", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, "grpo_trivial_success.png")


# ── Plot 3: Exp 03 flat accuracy ─────────────────────────────────────────────

def plot_grpo_03_flat():
    set_style()
    run = load_run(RESULTS_DIR / "03_RL_grpo_IND_ShareGPT_CoT__20260223_192913")
    steps = [b.global_step for b in run.batches]
    accs = [b.acc for b in run.batches]
    running = [b.running_acc for b in run.batches]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(steps, accs, alpha=0.15, s=8, color=C_GRPO, label="per batch")

    # Smoothed
    window = 20
    if len(accs) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(accs, kernel, mode="valid")
        offset = window // 2
        ax.plot(steps[offset:offset + len(smoothed)], smoothed,
                color=C_GRPO, linewidth=2, label=f"MA({window})")

    ax.plot(steps, running, color="C1", linewidth=1.5, alpha=0.7, label="cumulative")
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="chance")

    # Epoch boundaries
    n = run.n_batches_per_epoch
    for i in range(1, run.n_epochs):
        ax.axvline(i * n, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 03: GRPO on Real Self-Recognition Task (3 epochs, 320 samples)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _savefig(fig, "grpo_03_flat.png")


# ── Plot 4: Exp 07 collapse ──────────────────────────────────────────────────

def plot_grpo_07_collapse():
    set_style()
    run = load_run(RESULTS_DIR / "07_overfit_high_temp__20260224_025419")
    steps = [b.global_step for b in run.batches]
    accs = [b.acc for b in run.batches]
    running = [b.running_acc for b in run.batches]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(steps, accs, alpha=0.25, s=12, color=C_GRPO, label="per batch")
    ax.plot(steps, running, color="C1", linewidth=1.5, alpha=0.7, label="cumulative")
    ax.axhline(0.5, color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="chance")

    # Epoch boundaries
    n = run.n_batches_per_epoch
    for i in range(1, run.n_epochs):
        ax.axvline(i * n, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Annotate collapse
    ax.annotate("collapse: gibberish output",
                xy=(33, 0.03), xytext=(45, 0.15),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=10, color="red")

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 07: GRPO with temp=1.5 (collapse at epoch 9)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    _savefig(fig, "grpo_07_collapse.png")


# ── Plot 5: SFT NLL curves (exp 10/11/12) ────────────────────────────────────

def plot_sft_overfit_nll():
    set_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    # Parse NLL from logs
    for exp_dir, label, color in [
        ("10_sft_pw_debug_all1s__20260224_144254", 'All targets = "1"', "C0"),
        ("11_sft_pw_debug_all2s__20260224_144255", 'All targets = "2"', "C1"),
        ("12_sft_pw_debug_mixed__20260224_144256", "Mixed (8x1, 8x2)", "C2"),
    ]:
        run = load_run(RESULTS_DIR / exp_dir)
        # SFT batches have NLL in the log -- we need to re-parse
        log_text = (RESULTS_DIR / exp_dir / "train.log").read_text()
        import re
        nll_matches = re.findall(r"nll=([\d.]+)", log_text)
        nlls = [float(x) for x in nll_matches]
        epochs = list(range(1, len(nlls) + 1))
        ax.plot(epochs, nlls, marker="o", markersize=5, linewidth=2, label=label, color=color)

    ax.axhline(np.log(2), color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="ln(2) = chance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.set_title("SFT Debug: NLL on 16 Training Samples")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 2.0)
    fig.tight_layout()
    _savefig(fig, "sft_overfit_nll.png")


# ── Plot 6: LR sweep ─────────────────────────────────────────────────────────

def plot_lr_sweep():
    set_style()

    # Hardcoded data from Tinker runs
    lr_1e4_nll = [
        0.9580, 0.4636, 0.4737, 0.3782, 0.3454, 0.3463, 0.3439, 0.3539, 0.3509, 0.3542,
        0.3438, 0.3515, 0.3488, 0.3428, 0.3444, 0.3433, 0.3427, 0.3410, 0.3362, 0.3419,
        0.3345, 0.3294, 0.3236, 0.3113, 0.3013, 0.2719, 0.2135, 0.1023, 1.3849, 0.4506,
    ]
    lr_5e5_nll = [
        0.9616, 0.7139, 0.4345, 0.4454, 0.4200, 0.3743, 0.3597, 0.3447, 0.3455, 0.3421,
        0.3520, 0.3493, 0.3522, 0.3489, 0.3451, 0.3483, 0.3398, 0.3436, 0.3397, 0.3421,
        0.3429, 0.3360, 0.3486, 0.3430, 0.3371, 0.3420, 0.3380, 0.3415, 0.3396, 0.3324,
        0.3412, 0.3391, 0.3350, 0.3284, 0.3323, 0.3250, 0.3247, 0.3158, 0.3106, 0.2973,
        0.2904, 0.2711, 0.2377, 0.1994, 0.1428, 0.0710, 0.0409, 0.2074, 0.0016, 0.6820,
    ]
    lr_1e5_nll = [
        0.9580, 0.9262, 0.8812, 0.8127, 0.7288, 0.6090, 0.5402, 0.4746, 0.4440, 0.4193,
        0.4291, 0.4306, 0.4171, 0.4233, 0.4132, 0.3977, 0.3801, 0.3755, 0.3705, 0.3601,
        0.3535, 0.3528, 0.3564, 0.3451, 0.3442, 0.3505, 0.3392, 0.3465, 0.3403, 0.3457,
        0.3434, 0.3358, 0.3358, 0.3340, 0.3417, 0.3378, 0.3322, 0.3400, 0.3345, 0.3459,
        0.3340, 0.3326, 0.3324, 0.3233, 0.3324, 0.3267, 0.3267, 0.3216, 0.3178, 0.3233,
    ]

    # Train accuracy checkpoints
    lr_1e4_acc = {5: 8/16, 10: 9/16, 15: 9/16, 20: 8/16, 25: 15/16, 30: 16/16}
    lr_5e5_acc = {10: 8/16, 20: 8/16, 30: 9/16, 40: 15/16, 50: 16/16}
    lr_1e5_acc = {10: 8/16, 20: 8/16, 30: 13/16, 40: 9/16, 50: 14/16}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # NLL
    for nll, label, color in [
        (lr_1e4_nll, "lr=1e-4 (30 ep)", "C3"),
        (lr_5e5_nll, "lr=5e-5 (50 ep)", "C0"),
        (lr_1e5_nll, "lr=1e-5 (50 ep)", "C2"),
    ]:
        epochs = list(range(1, len(nll) + 1))
        ax1.plot(epochs, nll, linewidth=2, label=label, color=color)

    ax1.axhline(np.log(2), color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="ln(2)")
    ax1.axhline(0.34, color="gray", linestyle="--", linewidth=1, alpha=0.4, label="bias plateau (~0.34)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NLL")
    ax1.set_title("NLL vs Epoch")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.5)

    # Train accuracy
    for acc_dict, label, color in [
        (lr_1e4_acc, "lr=1e-4", "C3"),
        (lr_5e5_acc, "lr=5e-5", "C0"),
        (lr_1e5_acc, "lr=1e-5", "C2"),
    ]:
        eps = sorted(acc_dict.keys())
        vals = [acc_dict[e] for e in eps]
        ax2.plot(eps, vals, marker="o", markersize=6, linewidth=2, label=label, color=color)

    ax2.axhline(0.5, color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="chance")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train Accuracy")
    ax2.set_title("Train Accuracy vs Epoch")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.3, 1.05)

    fig.suptitle("LR Sweep: Mixed 16 Samples, LoRA rank=32", fontsize=13, y=1.02)
    fig.tight_layout()
    _savefig(fig, "sft_lr_sweep.png")


# ── Plot 7: Rank sweep ───────────────────────────────────────────────────────

def plot_rank_sweep():
    set_style()

    # All at lr=5e-5, 50 epochs
    rank32_nll = [
        0.9616, 0.7139, 0.4345, 0.4454, 0.4200, 0.3743, 0.3597, 0.3447, 0.3455, 0.3421,
        0.3520, 0.3493, 0.3522, 0.3489, 0.3451, 0.3483, 0.3398, 0.3436, 0.3397, 0.3421,
        0.3429, 0.3360, 0.3486, 0.3430, 0.3371, 0.3420, 0.3380, 0.3415, 0.3396, 0.3324,
        0.3412, 0.3391, 0.3350, 0.3284, 0.3323, 0.3250, 0.3247, 0.3158, 0.3106, 0.2973,
        0.2904, 0.2711, 0.2377, 0.1994, 0.1428, 0.0710, 0.0409, 0.2074, 0.0016, 0.6820,
    ]
    rank64_nll = [
        0.9580, 0.7176, 0.4298, 0.4475, 0.4187, 0.3767, 0.3536, 0.3492, 0.3417, 0.3513,
        0.3554, 0.3504, 0.3514, 0.3497, 0.3416, 0.3504, 0.3438, 0.3436, 0.3437, 0.3440,
        0.3503, 0.3434, 0.3420, 0.3373, 0.3462, 0.3399, 0.3447, 0.3388, 0.3416, 0.3437,
        0.3359, 0.3381, 0.3380, 0.3391, 0.3305, 0.3340, 0.3321, 0.3228, 0.3181, 0.3074,
        0.2963, 0.2736, 0.2533, 0.2147, 0.1596, 0.0928, 0.0344, 0.0068, 0.0004, 0.0027,
    ]
    rank128_nll = [
        0.9580, 0.7189, 0.4453, 0.4402, 0.4148, 0.3804, 0.3598, 0.3513, 0.3456, 0.3573,
        0.3475, 0.3521, 0.3548, 0.3445, 0.3433, 0.3443, 0.3476, 0.3455, 0.3457, 0.3486,
        0.3433, 0.3494, 0.3466, 0.3430, 0.3415, 0.3379, 0.3426, 0.3408, 0.3417, 0.3371,
        0.3409, 0.3391, 0.3372, 0.3306, 0.3286, 0.3314, 0.3273, 0.3267, 0.3144, 0.3062,
        0.3012, 0.2849, 0.2630, 0.2390, 0.1898, 0.1289, 0.0748, 0.1563, 0.0640, 0.0107,
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for nll, label, color in [
        (rank32_nll, "rank=32", "C0"),
        (rank64_nll, "rank=64", "C1"),
        (rank128_nll, "rank=128", "C2"),
    ]:
        epochs = list(range(1, len(nll) + 1))
        ax.plot(epochs, nll, linewidth=2, label=label, color=color)

    ax.axhline(np.log(2), color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="ln(2)")
    ax.axhline(0.34, color="gray", linestyle="--", linewidth=1, alpha=0.4, label="bias plateau")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.set_title("Rank Sweep: lr=5e-5, Mixed 16 Samples, 50 Epochs")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    _savefig(fig, "sft_rank_sweep.png")


# ── Plot 8: Exp 13 full SFT training process ────────────────────────────────

def plot_exp13_training():
    """Visualize the successful full-dataset SFT run (exp 13)."""
    set_style()
    import re

    log_path = RESULTS_DIR / "13_sft_pw_full__20260224_184937" / "train.log"
    log_text = log_path.read_text()

    # Parse NLL per batch: [epoch X/50] batch Y/11 | nll=Z.ZZZZ
    nll_re = re.compile(
        r"\[epoch (\d+)/(\d+)\] batch (\d+)/(\d+) \| nll=([\d.]+)"
    )
    steps, nlls, epoch_of_step = [], [], []
    for m in nll_re.finditer(log_text):
        ep, n_ep, batch, n_batch = int(m[1]), int(m[2]), int(m[3]), int(m[4])
        step = (ep - 1) * n_batch + batch
        steps.append(step)
        nlls.append(float(m[5]))
        epoch_of_step.append(ep)

    # Parse val accuracy: val: X/62 = Y.Y% | answers={1:A,2:B,?:C}
    val_re = re.compile(
        r"val: (\d+)/(\d+) = ([\d.]+)% \| answers=\{1:(\d+),2:(\d+),\?:(\d+)\}"
    )
    val_epochs, val_accs, val_n1s, val_n2s = [], [], [], []
    for i, m in enumerate(val_re.finditer(log_text)):
        val_epochs.append(i + 1)
        val_accs.append(float(m[3]) / 100)
        val_n1s.append(int(m[4]))
        val_n2s.append(int(m[5]))

    # --- Create figure ---
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: NLL per step
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(steps, nlls, alpha=0.2, s=6, color=C_SFT, zorder=2)
    # Epoch-averaged NLL
    epoch_avg_nll = []
    for ep in range(1, max(epoch_of_step) + 1):
        ep_nlls = [n for n, e in zip(nlls, epoch_of_step) if e == ep]
        epoch_avg_nll.append(np.mean(ep_nlls))
    # Plot epoch average at middle of each epoch
    n_batch = 11
    epoch_mid_steps = [(ep - 0.5) * n_batch for ep in range(1, len(epoch_avg_nll) + 1)]
    ax1.plot(epoch_mid_steps, epoch_avg_nll, color=C_SFT, linewidth=2, label="epoch avg", zorder=3)
    ax1.axhline(np.log(2), color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="ln(2)")
    ax1.axhline(0.34, color="gray", linestyle="--", linewidth=1, alpha=0.4, label="bias plateau")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("NLL")
    ax1.set_title("Training NLL per Batch")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.02, 1.2)

    # Panel 2: Val accuracy per epoch
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(val_epochs, val_accs, marker="o", markersize=4, linewidth=2, color=C_SFT)
    ax2.axhline(0.5, color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="chance")
    ax2.axhline(0.424, color=C_BASELINE, linestyle="--", linewidth=1, alpha=0.6, label="baseline (42.4%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val Accuracy")
    ax2.set_title("Validation Accuracy (62 held-out samples)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(fontsize=8)
    ax2.set_ylim(0.35, 1.05)

    # Panel 3: Answer distribution over epochs
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(val_epochs, val_n1s, label='Predicted "1"', alpha=0.6, color="C0")
    ax3.fill_between(val_epochs, val_n2s, label='Predicted "2"', alpha=0.6, color="C1")
    ax3.axhline(31, color="gray", linestyle="--", linewidth=1, alpha=0.4, label="balanced (31)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Count (out of 62)")
    ax3.set_title("Val Answer Distribution")
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 65)

    # Panel 4: Epoch-averaged NLL (zoomed in on the interesting part)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(range(1, len(epoch_avg_nll) + 1), epoch_avg_nll,
             marker="o", markersize=4, linewidth=2, color=C_SFT)
    ax4.axhline(np.log(2), color=CHANCE_COLOR, linestyle=":", linewidth=1, alpha=0.6, label="ln(2)")
    ax4.axhline(0.34, color="gray", linestyle="--", linewidth=1, alpha=0.4, label="bias plateau")

    # Annotate key phases
    ax4.annotate("bias plateau",
                 xy=(4, 0.34), xytext=(8, 0.55),
                 arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
                 fontsize=9, color="gray")
    ax4.annotate("phase transition",
                 xy=(9, 0.20), xytext=(15, 0.35),
                 arrowprops=dict(arrowstyle="->", color=C_SFT, lw=1.2),
                 fontsize=9, color=C_SFT)

    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Avg NLL")
    ax4.set_title("Epoch-Averaged NLL")
    ax4.legend(fontsize=8)
    ax4.set_ylim(-0.02, 0.8)

    fig.suptitle("Exp 13: SFT on Full PW Dataset (184 train, 62 val, lr=5e-5, rank=32)",
                 fontsize=14, y=1.01)
    _savefig(fig, "sft_exp13_training.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating writeup plots...")
    plot_summary_bar()
    plot_trivial_success()
    plot_grpo_03_flat()
    plot_grpo_07_collapse()
    plot_sft_overfit_nll()
    plot_lr_sweep()
    plot_rank_sweep()
    plot_exp13_training()
    print("Done!")


if __name__ == "__main__":
    main()
