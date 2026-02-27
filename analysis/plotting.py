"""Plotting utilities for SGTR-RL training runs."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from analysis.utils import RunData


def set_style():
    """Set a clean plot style."""
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })


def _epoch_boundaries(run: RunData) -> list[int]:
    """Return global steps where each epoch starts."""
    if not run.batches:
        return []
    n = run.n_batches_per_epoch
    return [i * n for i in range(1, run.n_epochs)]


def _add_epoch_lines(ax, run: RunData):
    """Add vertical dashed lines at epoch boundaries."""
    for step in _epoch_boundaries(run):
        ax.axvline(step, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)


def _smooth(values: list[float], window: int) -> np.ndarray:
    """Simple moving average."""
    if window <= 1:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_reward(run: RunData, window: int = 10, ax: plt.Axes | None = None):
    """Plot mean reward per batch over training.

    Args:
        run: Parsed run data.
        window: Smoothing window size.
        ax: Optional axes to plot on.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots()

    steps = [b.global_step for b in run.batches]
    rewards = [b.reward for b in run.batches]

    ax.scatter(steps, rewards, alpha=0.15, s=8, color="C0", label="per batch")

    if len(rewards) >= window:
        smoothed = _smooth(rewards, window)
        offset = window // 2
        ax.plot(steps[offset:offset + len(smoothed)], smoothed,
                color="C0", linewidth=2, label=f"MA({window})")

    ax.axhline(0.5, color="red", linestyle=":", linewidth=1, alpha=0.6, label="chance")
    _add_epoch_lines(ax, run)

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title(f"Reward — {run.experiment_name}")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)


def plot_accuracy(run: RunData, window: int = 10, ax: plt.Axes | None = None):
    """Plot batch accuracy and running accuracy over training.

    Args:
        run: Parsed run data.
        window: Smoothing window for batch accuracy.
        ax: Optional axes to plot on.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots()

    steps = [b.global_step for b in run.batches]
    accs = [b.acc for b in run.batches]
    running = [b.running_acc for b in run.batches]

    ax.scatter(steps, accs, alpha=0.15, s=8, color="C0", label="per batch")

    if len(accs) >= window:
        smoothed = _smooth(accs, window)
        offset = window // 2
        ax.plot(steps[offset:offset + len(smoothed)], smoothed,
                color="C0", linewidth=2, label=f"MA({window})")

    ax.plot(steps, running, color="C1", linewidth=1.5, alpha=0.7, label="cumulative")
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1, alpha=0.6, label="chance")
    _add_epoch_lines(ax, run)

    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy — {run.experiment_name}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)


def plot_datums(run: RunData, window: int = 10, ax: plt.Axes | None = None):
    """Plot datums (gradient signal) per batch.

    Higher = more prompts had mixed rollouts (learning signal).
    Lower/zero = model is unanimous on those prompts.

    Args:
        run: Parsed run data.
        window: Smoothing window.
        ax: Optional axes to plot on.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots()

    steps = [b.global_step for b in run.batches]
    datums = [b.datums for b in run.batches]

    ax.bar(steps, datums, alpha=0.4, width=0.8, color="C2", label="per batch")

    if len(datums) >= window:
        smoothed = _smooth([float(d) for d in datums], window)
        offset = window // 2
        ax.plot(steps[offset:offset + len(smoothed)], smoothed,
                color="C2", linewidth=2, label=f"MA({window})")

    max_datums = run.batches[0].n_batches if run.batches else 16
    # max possible datums = batch_size * group_size (4*4=16 typically)
    _add_epoch_lines(ax, run)

    ax.set_xlabel("Step")
    ax.set_ylabel("Datums")
    ax.set_title(f"Gradient Signal (Datums) — {run.experiment_name}")
    ax.legend(fontsize=9)


def plot_batch_time(run: RunData, ax: plt.Axes | None = None):
    """Plot time per batch (useful for spotting API slowdowns).

    Args:
        run: Parsed run data.
        ax: Optional axes to plot on.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots()

    steps = [b.global_step for b in run.batches]
    times = [b.elapsed for b in run.batches]

    ax.plot(steps, times, color="C3", linewidth=1, alpha=0.7)
    ax.axhline(np.median(times), color="C3", linestyle="--", linewidth=1,
               alpha=0.5, label=f"median={np.median(times):.1f}s")
    _add_epoch_lines(ax, run)

    ax.set_xlabel("Step")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Batch Time — {run.experiment_name}")
    ax.legend(fontsize=9)


def plot_epoch_summary(run: RunData, ax: plt.Axes | None = None):
    """Plot per-epoch average reward and accuracy.

    Args:
        run: Parsed run data.
        ax: Optional axes to plot on.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots()

    epochs = [e.epoch for e in run.epochs]
    rewards = [e.avg_reward for e in run.epochs]
    accs = [e.running_acc for e in run.epochs if e.running_acc is not None]

    ax.bar([e - 0.15 for e in epochs], rewards, width=0.3, color="C0",
           alpha=0.7, label="avg reward")
    if accs:
        ax.bar([e + 0.15 for e in epochs[:len(accs)]], accs, width=0.3,
               color="C1", alpha=0.7, label="running acc")

    ax.axhline(0.5, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(f"Epoch Summary — {run.experiment_name}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.set_xticks(epochs)


def plot_accuracy_over_time(run: RunData, window: int = 10, ax: plt.Axes | None = None):
    """Plot accuracy over wall-clock time (minutes).

    Args:
        run: Parsed run data.
        window: Smoothing window.
        ax: Optional axes to plot on.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots()

    mins = run.wall_minutes
    if not mins:
        ax.text(0.5, 0.5, "No timestamp data", ha="center", va="center", transform=ax.transAxes)
        return

    accs = [b.acc for b in run.batches]
    running = [b.running_acc for b in run.batches]

    ax.scatter(mins, accs, alpha=0.15, s=8, color="C0", label="per batch")

    if len(accs) >= window:
        smoothed = _smooth(accs, window)
        offset = window // 2
        ax.plot(mins[offset:offset + len(smoothed)], smoothed,
                color="C0", linewidth=2, label=f"MA({window})")

    ax.plot(mins, running, color="C1", linewidth=1.5, alpha=0.7, label="cumulative")
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1, alpha=0.6, label="chance")

    # Epoch boundaries by time
    for boundary_step in _epoch_boundaries(run):
        idx = boundary_step - 1  # steps are 1-indexed
        if 0 <= idx < len(mins):
            ax.axvline(mins[idx], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Wall Time (min)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy over Time — {run.experiment_name}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)


def plot_progress(run: RunData, ax: plt.Axes | None = None):
    """Plot training progress: steps completed over wall time.

    Useful for spotting stalls and estimating remaining time.

    Args:
        run: Parsed run data.
        ax: Optional axes to plot on.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots()

    mins = run.wall_minutes
    if not mins:
        ax.text(0.5, 0.5, "No timestamp data", ha="center", va="center", transform=ax.transAxes)
        return

    steps = [b.global_step for b in run.batches]
    total = run.n_epochs * run.n_batches_per_epoch

    ax.plot(mins, steps, color="C4", linewidth=2)
    ax.axhline(total, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label=f"total={total} steps")

    # Ideal pace line
    if mins[-1] > 0:
        ideal_rate = steps[-1] / mins[-1]
        ax.plot([0, total / ideal_rate], [0, total], color="gray",
                linestyle=":", linewidth=1, alpha=0.4, label="ideal pace")

    for boundary_step in _epoch_boundaries(run):
        idx = boundary_step - 1
        if 0 <= idx < len(mins):
            ax.axvline(mins[idx], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Wall Time (min)")
    ax.set_ylabel("Step")
    ax.set_title(f"Progress — {run.experiment_name}")
    ax.legend(fontsize=9)


def plot_cumulative_time(run: RunData, ax: plt.Axes | None = None):
    """Plot cumulative time spent, broken down by batch time.

    Shows where time was spent (steady vs spikes).

    Args:
        run: Parsed run data.
        ax: Optional axes to plot on.
    """
    set_style()
    if ax is None:
        fig, ax = plt.subplots()

    steps = [b.global_step for b in run.batches]
    cum_time = np.cumsum([b.elapsed for b in run.batches]) / 60  # minutes

    ax.plot(steps, cum_time, color="C3", linewidth=2)
    _add_epoch_lines(ax, run)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Time (min)")
    ax.set_title(f"Cumulative Time — {run.experiment_name}")


def plot_dashboard(run: RunData, window: int = 10):
    """Plot a 2x3 dashboard of key training metrics.

    Args:
        run: Parsed run data.
        window: Smoothing window.

    Returns:
        matplotlib Figure.
    """
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"{run.experiment_name} — {run.model_name}", fontsize=14, y=0.98)

    plot_accuracy(run, window=window, ax=axes[0, 0])
    plot_accuracy_over_time(run, window=window, ax=axes[0, 1])
    plot_reward(run, window=window, ax=axes[0, 2])
    plot_datums(run, window=window, ax=axes[1, 0])
    plot_batch_time(run, ax=axes[1, 1])
    plot_progress(run, ax=axes[1, 2])

    fig.tight_layout()
    return fig
