"""Evaluation utilities for SGTR-RL checkpoints.

Algorithm-agnostic: takes a model string and runs SGTR + auxiliary eval tasks
using inspect-ai and the self-rec-framework prompt/task system.
"""

import json
from pathlib import Path

from self_rec_framework.src.helpers.model_names import INSPECT_MODEL_NAMES


def get_model_str(checkpoint_or_name: str, backend: str = "hf") -> str:
    """Build an inspect-ai model string.

    Args:
        checkpoint_or_name: Either a local checkpoint path or a short model
            name from the framework's model registry.
        backend: Inference backend to use when resolving a local path.
            One of ``"hf"``, ``"vllm"``, ``"together"``, or any provider
            supported by inspect-ai.

    Returns:
        An inspect-ai model string such as ``"hf/path/to/checkpoint"`` or
        ``"together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"``.
    """
    path = Path(checkpoint_or_name)
    if path.exists():
        return f"{backend}/{checkpoint_or_name}"

    if checkpoint_or_name in INSPECT_MODEL_NAMES:
        return INSPECT_MODEL_NAMES[checkpoint_or_name]

    # Assume it's already a full inspect model string
    return checkpoint_or_name


def evaluate_checkpoint(
    model_str: str,
    eval_tasks: list[dict],
    results_dir: str | None = None,
) -> dict:
    """Run evaluation tasks against a model.

    Args:
        model_str: Inspect-ai model string (from :func:`get_model_str`).
        eval_tasks: List of task specs from the experiment config YAML.
            Each dict must have a ``type`` key (``"sgtr"`` or ``"inspect"``).
        results_dir: Optional directory to save results JSON.

    Returns:
        Dict mapping task names to accuracy floats (0-1).
    """
    results: dict[str, float] = {}

    for task_spec in eval_tasks:
        task_type = task_spec["type"]
        task_name = task_spec.get("name", task_type)

        if task_type == "sgtr":
            acc = _run_sgtr_eval(model_str, task_spec)
        elif task_type == "inspect":
            acc = _run_inspect_eval(model_str, task_spec)
        else:
            print(f"Unknown eval task type: {task_type!r}, skipping")
            continue

        results[task_name] = acc

    if results_dir:
        _save_results(results, results_dir, model_str)

    return results


def _run_sgtr_eval(model_str: str, task_spec: dict) -> float:
    """Run an SGTR evaluation using self-rec-framework tasks.

    Args:
        model_str: Inspect-ai model string.
        task_spec: Must contain ``config_path``, ``dataset``, ``subset``.

    Returns:
        Accuracy as a float (0-1).
    """
    from inspect_ai import eval as inspect_eval

    from self_rec_framework.src.inspect.config import (
        load_experiment_config,
        ensure_evaluator_reasoning,
    )
    from self_rec_framework.src.inspect.tasks import get_task_function

    config_path = task_spec["config_path"]
    dataset = task_spec["dataset"]
    subset = task_spec["subset"]

    exp_config = load_experiment_config(config_path, dataset)

    # Extract the short model name from the model_str for prompt building.
    # For local checkpoints (hf/...) we fall back to the model_str itself;
    # the evaluator reasoning defaults will apply.
    short_name = model_str.split("/")[-1] if "/" in model_str else model_str
    ensure_evaluator_reasoning(exp_config, short_name)

    task = get_task_function(
        exp_config=exp_config,
        model_name=short_name,
        treatment_name_control=short_name,
        treatment_name_treatment=short_name,
        dataset_name=dataset,
        data_subset=subset,
        is_control=True,
    )

    log = inspect_eval(task, model=model_str)
    if log and log[0].results:
        metrics = log[0].results.metrics
        if "accuracy" in metrics:
            return metrics["accuracy"].value
    return 0.0


def _run_inspect_eval(model_str: str, task_spec: dict) -> float:
    """Run a standard inspect-ai benchmark task.

    Args:
        model_str: Inspect-ai model string.
        task_spec: Must contain ``task`` (e.g. ``"mmlu"``).  May contain
            ``num_samples`` to limit the evaluation size.

    Returns:
        Accuracy as a float (0-1).
    """
    from inspect_ai import eval as inspect_eval

    task_name = task_spec["task"]
    num_samples = task_spec.get("num_samples")

    log = inspect_eval(
        task_name,
        model=model_str,
        limit=num_samples,
    )
    if log and log[0].results:
        metrics = log[0].results.metrics
        if "accuracy" in metrics:
            return metrics["accuracy"].value
    return 0.0


def _save_results(results: dict, results_dir: str, model_str: str) -> None:
    """Persist evaluation results as JSON."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    safe_name = model_str.replace("/", "_").replace("\\", "_")
    out_file = results_path / f"eval_{safe_name}.json"
    with open(out_file, "w") as f:
        json.dump({"model": model_str, "results": results}, f, indent=2)
    print(f"Results saved to {out_file}")


class EvalCallback:
    """Lightweight eval callback for use during training.

    Call :meth:`__call__` at ``eval_steps`` intervals to run a small eval
    suite and collect accuracy over training steps.
    """

    def __init__(self, model_str: str, eval_tasks: list[dict]):
        self.model_str = model_str
        self.eval_tasks = eval_tasks
        self.history: list[dict] = []

    def __call__(self, step: int) -> dict:
        """Run evals and record results for the given training step."""
        results = evaluate_checkpoint(self.model_str, self.eval_tasks)
        entry = {"step": step, **results}
        self.history.append(entry)
        return entry
