"""Metric logging and prediction saving helpers."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def log_binary_eval_result(label: str, result: dict) -> None:
    """Log binary classification eval results in a standard format."""
    nll_str = f" | nll={result['nll']:.4f}" if "nll" in result else ""
    logger.info(
        f"  {label}: {result['correct']}/{result['total']} "
        f"= {result['accuracy']:.1%}{nll_str} | "
        f"answers={{1:{result['answers']['1']},"
        f"2:{result['answers']['2']},"
        f"?:{result['answers']['other']}}}"
    )


def log_val_result(val_result: dict) -> None:
    """Log validation results in standard format."""
    log_binary_eval_result("val", val_result)


def log_split_metrics(ml_logger, result: dict, step: int, *, prefix: str) -> None:
    """Log binary eval metrics with a configurable prefix."""
    total = max(result["total"], 1)
    metrics = {
        f"{prefix}/accuracy": result["accuracy"],
        f"{prefix}/answers_1_pct": result["answers"]["1"] / total,
        f"{prefix}/answers_other_pct": result["answers"]["other"] / total,
    }
    if "nll" in result:
        metrics[f"{prefix}/nll"] = result["nll"]
    ml_logger.log_metrics(metrics, step=step)


def log_val_metrics(ml_logger, val_result: dict, step: int) -> None:
    """Log validation metrics via ml_logger (W&B, JSON, etc.)."""
    log_split_metrics(ml_logger, val_result, step, prefix="val")


def make_eval_artifact_name(*, epoch: int, step: int, eval_trigger: str) -> str:
    """Build a stable artifact name for an eval event."""
    if eval_trigger == "step":
        return f"step_{step}"
    return f"epoch_{epoch}"


def save_val_predictions(
    val_result: dict,
    run_dir: str,
    *,
    epoch: int,
    step: int,
    eval_trigger: str,
) -> None:
    """Save per-sample val predictions to JSON."""
    save_split_predictions(
        val_result,
        run_dir,
        split_name="val",
        epoch=epoch,
        step=step,
        eval_trigger=eval_trigger,
    )


def save_split_predictions(
    result: dict,
    run_dir: str,
    *,
    split_name: str,
    epoch: int,
    step: int,
    eval_trigger: str,
) -> None:
    """Save per-sample predictions for a named split to JSON."""
    pred_dir = Path(run_dir) / f"{split_name}_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / (
        f"{make_eval_artifact_name(epoch=epoch, step=step, eval_trigger=eval_trigger)}.json"
    )
    with open(pred_path, "w") as f:
        json.dump({
            "epoch": epoch,
            "step": step,
            "eval_trigger": eval_trigger,
            "accuracy": result["accuracy"],
            "predictions": result["predictions"],
        }, f, indent=2)
    logger.debug("  %s predictions saved to %s", split_name, pred_path)


def save_val_diagnostics(
    payload: dict,
    run_dir: str,
    *,
    epoch: int,
    step: int,
    eval_trigger: str,
) -> None:
    """Save fixed-panel validation diagnostics to JSON."""
    save_split_diagnostics(
        payload,
        run_dir,
        split_name="val",
        epoch=epoch,
        step=step,
        eval_trigger=eval_trigger,
    )


def save_split_diagnostics(
    payload: dict,
    run_dir: str,
    *,
    split_name: str,
    epoch: int,
    step: int,
    eval_trigger: str,
) -> None:
    """Save fixed-panel diagnostics for a named split to JSON."""
    diag_dir = Path(run_dir) / f"{split_name}_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / (
        f"{make_eval_artifact_name(epoch=epoch, step=step, eval_trigger=eval_trigger)}.json"
    )
    with open(diag_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.debug("  %s diagnostics saved to %s", split_name, diag_path)
