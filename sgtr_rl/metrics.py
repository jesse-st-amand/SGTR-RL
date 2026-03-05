"""Metric logging and prediction saving helpers."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def log_val_result(val_result: dict) -> None:
    """Log validation results in standard format."""
    nll_str = f" | nll={val_result['nll']:.4f}" if "nll" in val_result else ""
    logger.info(
        f"  val: {val_result['correct']}/{val_result['total']} "
        f"= {val_result['accuracy']:.1%}{nll_str} | "
        f"answers={{1:{val_result['answers']['1']},"
        f"2:{val_result['answers']['2']},"
        f"?:{val_result['answers']['other']}}}"
    )


def log_val_metrics(ml_logger, val_result: dict, step: int) -> None:
    """Log validation metrics via ml_logger (W&B, JSON, etc.)."""
    total = max(val_result["total"], 1)
    metrics = {
        "val/accuracy": val_result["accuracy"],
        "val/answers_1_pct": val_result["answers"]["1"] / total,
        "val/answers_other_pct": val_result["answers"]["other"] / total,
    }
    if "nll" in val_result:
        metrics["val/nll"] = val_result["nll"]
    ml_logger.log_metrics(metrics, step=step)


def save_val_predictions(val_result: dict, run_dir: str, epoch: int) -> None:
    """Save per-sample val predictions to JSON."""
    pred_dir = Path(run_dir) / "val_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"epoch_{epoch}.json"
    with open(pred_path, "w") as f:
        json.dump({
            "epoch": epoch,
            "accuracy": val_result["accuracy"],
            "predictions": val_result["predictions"],
        }, f, indent=2)
    logger.debug(f"  val predictions saved to {pred_path}")
