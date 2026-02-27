"""Utilities for loading and parsing SGTR-RL training run data."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class BatchRecord:
    """Single batch log entry."""

    epoch: int
    n_epochs: int
    batch: int
    n_batches: int
    reward: float
    acc: float
    running_acc: float
    datums: int
    elapsed: float
    global_step: int  # computed: (epoch-1)*n_batches + batch
    timestamp: str


@dataclass
class EpochRecord:
    """Epoch summary log entry."""

    epoch: int
    avg_reward: float
    running_correct: int | None
    running_total: int | None
    running_acc: float | None


@dataclass
class RunData:
    """All parsed data from a training run."""

    log_file: Path
    experiment_name: str
    model_name: str
    config: dict  # raw config values from log header
    batches: list[BatchRecord]
    epochs: list[EpochRecord]

    @property
    def n_epochs(self) -> int:
        return self.batches[0].n_epochs if self.batches else 0

    @property
    def n_batches_per_epoch(self) -> int:
        return self.batches[0].n_batches if self.batches else 0

    @property
    def total_steps(self) -> int:
        return len(self.batches)

    @property
    def wall_minutes(self) -> list[float]:
        """Minutes elapsed since first batch for each batch."""
        if not self.batches or not self.batches[0].timestamp:
            return []
        t0 = datetime.strptime(self.batches[0].timestamp, "%Y-%m-%d %H:%M:%S")
        mins = []
        for b in self.batches:
            if not b.timestamp:
                mins.append(0.0)
                continue
            t = datetime.strptime(b.timestamp, "%Y-%m-%d %H:%M:%S")
            mins.append((t - t0).total_seconds() / 60)
        return mins

    @property
    def total_wall_minutes(self) -> float:
        """Total wall time in minutes."""
        mins = self.wall_minutes
        return mins[-1] if mins else 0.0

    def summary(self) -> str:
        """Print a human-readable summary of the run."""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Model: {self.model_name}",
            f"Steps: {self.total_steps} ({self.n_epochs} epochs x {self.n_batches_per_epoch} batches)",
            f"Wall time: {self.total_wall_minutes:.0f} min ({self.total_wall_minutes/60:.1f} hr)",
        ]
        if self.epochs:
            for e in self.epochs:
                acc_str = f"{e.running_acc:.1%}" if e.running_acc is not None else "N/A"
                lines.append(f"  Epoch {e.epoch}: reward={e.avg_reward:.3f}, acc={acc_str}")
        if self.batches:
            all_datums = [b.datums for b in self.batches]
            zero_pct = sum(1 for d in all_datums if d == 0) / len(all_datums)
            lines.append(f"Datums: avg={sum(all_datums)/len(all_datums):.1f}, "
                         f"zero={zero_pct:.0%} of batches")
        return "\n".join(lines)


# Regex patterns for parsing log lines
_BATCH_RE = re.compile(
    r"\[epoch (\d+)/(\d+)\] batch (\d+)/(\d+) \| "
    r"reward=([\d.]+) \| acc=([\d.]+)% \(running=([\d.]+)%\) \| "
    r"datums=(\d+) \|.*?([\d.]+)s\s*$"
)

_EPOCH_RE = re.compile(
    r"Epoch (\d+) complete \| avg reward=([\d.]+)"
    r"(?: \| running acc=(\d+)/(\d+) = ([\d.]+)%)?"
)

_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def list_runs(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """List all run directories (that contain train.log), most recent first."""
    if not results_dir.exists():
        return []
    runs = [p for p in results_dir.iterdir() if p.is_dir() and (p / "train.log").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def load_run(log_path: Path | str) -> RunData:
    """Parse a training log file into structured data.

    Accepts either a path to train.log directly, or a run directory
    containing train.log.
    """
    log_path = Path(log_path)
    if log_path.is_dir():
        log_path = log_path / "train.log"
    lines = log_path.read_text().splitlines()

    batches: list[BatchRecord] = []
    epochs: list[EpochRecord] = []
    config: dict = {}
    experiment_name = ""
    model_name = ""

    for line in lines:
        # Parse header info
        if "Experiment:" in line:
            experiment_name = line.split("Experiment:")[-1].strip()
        elif "Model:" in line:
            model_name = line.split("Model:")[-1].strip()
        elif "Hyperparameters:" in line:
            config["hyperparameters"] = line.split("Hyperparameters:")[-1].strip()
        elif "Data:" in line and "train=" in line:
            config["data"] = line.split("Data:")[-1].strip()

        # Parse batch lines
        m = _BATCH_RE.search(line)
        if m:
            epoch, n_epochs, batch, n_batches = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            ts_m = _TIMESTAMP_RE.match(line)
            batches.append(BatchRecord(
                epoch=epoch,
                n_epochs=n_epochs,
                batch=batch,
                n_batches=n_batches,
                reward=float(m[5]),
                acc=float(m[6]) / 100,
                running_acc=float(m[7]) / 100,
                datums=int(m[8]),
                elapsed=float(m[9]),
                global_step=(epoch - 1) * n_batches + batch,
                timestamp=ts_m[1] if ts_m else "",
            ))
            continue

        # Parse epoch lines
        m = _EPOCH_RE.search(line)
        if m:
            epochs.append(EpochRecord(
                epoch=int(m[1]),
                avg_reward=float(m[2]),
                running_correct=int(m[3]) if m[3] else None,
                running_total=int(m[4]) if m[4] else None,
                running_acc=float(m[5]) / 100 if m[5] else None,
            ))

    return RunData(
        log_file=log_path,
        experiment_name=experiment_name,
        model_name=model_name,
        config=config,
        batches=batches,
        epochs=epochs,
    )


def load_latest_run(results_dir: Path = RESULTS_DIR) -> RunData:
    """Load the most recent training run."""
    runs = list_runs(results_dir)
    if not runs:
        raise FileNotFoundError(f"No run directories found in {results_dir}")
    return load_run(runs[0])


def load_training_data(jsonl_path: Path | str) -> list[dict]:
    """Load a training/val JSONL file."""
    path = Path(jsonl_path)
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records
