#!/usr/bin/env python3
"""Generate training analysis plots for a run.

Usage:
    python -m analysis.plot_run                          # latest run
    python -m analysis.plot_run results/my_run_dir/      # specific run
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.utils import list_runs, load_run, load_latest_run
from analysis.plotting import (
    plot_dashboard, plot_accuracy, plot_accuracy_over_time,
    plot_reward, plot_datums, plot_batch_time,
    plot_progress, plot_cumulative_time, plot_epoch_summary,
)

PLOTS_DIR = Path(__file__).parent / "plots"


def main():
    # Load run
    if len(sys.argv) > 1:
        run = load_run(sys.argv[1])
    else:
        print("Available runs:")
        for p in list_runs():
            print(f"  {p.name}")
        run = load_latest_run()

    print(f"\n{run.summary()}\n")

    PLOTS_DIR.mkdir(exist_ok=True)

    # Dashboard (2x3)
    fig = plot_dashboard(run, window=10)
    fig.savefig(PLOTS_DIR / "dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved dashboard.png")

    # Epoch summary
    fig, ax = plt.subplots()
    plot_epoch_summary(run, ax=ax)
    fig.savefig(PLOTS_DIR / "epoch_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved epoch_summary.png")

    # Per-epoch breakdown (text)
    print("\nPer-Epoch Breakdown:")
    for epoch_num in range(1, run.n_epochs + 1):
        epoch_batches = [b for b in run.batches if b.epoch == epoch_num]
        accs = [b.acc for b in epoch_batches]
        datums = [b.datums for b in epoch_batches]
        times = [b.elapsed for b in epoch_batches]

        print(f"  Epoch {epoch_num}:")
        print(f"    Accuracy: mean={np.mean(accs):.1%}, std={np.std(accs):.1%}")
        print(f"    Datums:   mean={np.mean(datums):.1f}, zero={sum(1 for d in datums if d==0)}/{len(datums)} batches")
        print(f"    Time:     mean={np.mean(times):.1f}s, total={sum(times)/60:.1f}min")

    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
