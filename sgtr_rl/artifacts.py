"""Artifact helpers shared by local training and launch workflows."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import wandb

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to avoid partial files after interruption."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(target)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """Append one JSON line and flush it to disk immediately."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a") as f:
        f.write(json.dumps(payload) + "\n")
        f.flush()
        os.fsync(f.fileno())


def update_run_status(
    run_dir: str | Path,
    status: str,
    *,
    backend: str,
    algorithm: str,
    step: int | None = None,
    epoch: int | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a compact run status record for remote monitoring and recovery."""
    path = Path(run_dir) / "status.json"
    payload: dict[str, Any] = {
        "status": status,
        "backend": backend,
        "algorithm": algorithm,
        "updated_at": _utc_now(),
    }
    if path.exists():
        with open(path, "r") as f:
            existing = json.load(f)
        payload = {**existing, **payload}
    else:
        payload["started_at"] = payload["updated_at"]
    if step is not None:
        payload["step"] = step
    if epoch is not None:
        payload["epoch"] = epoch
    if error is not None:
        payload["error"] = error
    if status in {"completed", "failed"}:
        payload["finished_at"] = payload["updated_at"]
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)


class JsonlMetricsLogger:
    """Append-only metrics logger with optional W&B mirroring."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        experiment_name: str,
        config_payload: dict[str, Any],
        wandb_project: str | None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.metrics_dir / "metrics.jsonl"
        atomic_write_json(self.metrics_dir / "config.json", config_payload)
        self._wandb_run = None
        if wandb_project:
            self._wandb_run = wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=config_payload,
                reinit=True,
            )

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        payload = {"step": step, **metrics}
        append_jsonl(self.metrics_path, payload)
        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=step)

    def close(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None
