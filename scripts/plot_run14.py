"""Quick plot of experiment 14 training run."""
import matplotlib.pyplot as plt
import numpy as np
import re

log_path = "results/14_sft_pw_uuid_split__20260225_000514/train.log"

# Untrained baseline (from eval_baseline.py run on PW val set)
untrained_val_acc = 45.0

# Parse val accuracy per epoch
val_accs = []
with open(log_path) as f:
    for line in f:
        m = re.search(r"val: (\d+)/(\d+) = ([\d.]+)%.*1:(\d+),2:(\d+)", line)
        if m:
            val_accs.append(float(m.group(3)))

# Parse train NLL per step
train_nlls = []
with open(log_path) as f:
    for line in f:
        m = re.search(r"nll=([\d.]+)", line)
        if m:
            train_nlls.append(float(m.group(1)))

# Compute per-epoch averages (10 batches per epoch)
batches_per_epoch = 10
epoch_nlls = []
for i in range(0, len(train_nlls), batches_per_epoch):
    nll_chunk = train_nlls[i:i + batches_per_epoch]
    if nll_chunk:
        epoch_nlls.append(np.mean(nll_chunk))

n_epochs = min(len(val_accs), len(epoch_nlls))
epochs = np.arange(1, n_epochs + 1)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Accuracy subplot — val only (train acc is always 100% in SFT, meaningless)
ax1 = axes[0]
ax1.plot(epochs, val_accs[:n_epochs], "o-", color="#2196F3", markersize=4,
         linewidth=1.5, label="Val accuracy")
ax1.plot(0, untrained_val_acc, "D", color="#9C27B0", markersize=8, zorder=5,
         label=f"Untrained baseline ({untrained_val_acc:.0f}%)")
ax1.axhline(50, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(35, 100)
ax1.set_xlim(-1, n_epochs + 1)
ax1.legend(loc="lower right")
ax1.set_title("SFT on Pairwise — 160 train / 40 val (80/20 UUIDs)\nLlama-3.1-8B (self) vs Qwen-2.5-7B (other) · ShareGPT", fontsize=12)
ax1.grid(alpha=0.3)

# NLL subplot — train only (val NLL not available for this run)
ax2 = axes[1]
ax2.plot(epochs, epoch_nlls[:n_epochs], "s-", color="#FF5722", markersize=4,
         linewidth=1.5, label="Train NLL (epoch avg)")
ax2.set_ylabel("NLL")
ax2.set_xlabel("Epoch")
ax2.set_yscale("log")
ax2.legend(loc="upper right")
ax2.grid(alpha=0.3)

plt.tight_layout()
out_path = "results/14_sft_pw_uuid_split__20260225_000514/training_plot.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
