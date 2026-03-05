"""Shared validation evaluation logic for Tinker-based trainers."""

import json
import logging
from pathlib import Path

from sgtr_rl.answer import extract_answer
from sgtr_rl.data import build_conversation

logger = logging.getLogger(__name__)


def compute_val_nll(
    val_prompts, training_client, renderer, use_system_prompt: bool = False,
) -> float:
    """Compute mean NLL on validation set via forward pass.

    Builds SFT-style datums from val prompts, runs forward_backward to get
    loss, then clears accumulated gradients with a zero-lr optimizer step
    so subsequent training is unaffected.

    Args:
        val_prompts: List of dicts with 'prompt' and 'target'.
        training_client: Tinker LoRA training client.
        renderer: Tinker renderer for tokenization.

    Returns:
        Mean NLL (negative log-likelihood) over all val tokens.
    """
    from tinker import types
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.common import compute_mean_nll
    from tinker_cookbook.supervised.data import conversation_to_datum

    datums = []
    for item in val_prompts:
        convo = build_conversation(item, use_system_prompt)
        convo.append({"role": "assistant", "content": item["target"]})
        datum = conversation_to_datum(
            convo, renderer, None, TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        datums.append(datum)

    # Forward+backward to get logprobs (backward is needed by API but
    # we discard gradients via a zero-lr optim step below)
    fwd_bwd_result = training_client.forward_backward(
        datums, loss_fn="cross_entropy"
    ).result()

    # Clear accumulated gradients without changing weights
    zero_adam = types.AdamParams(
        learning_rate=0.0, beta1=0.9, beta2=0.95, eps=1e-8
    )
    training_client.optim_step(zero_adam).result()

    logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
    weights = [d.loss_fn_inputs["weights"] for d in datums]
    return compute_mean_nll(logprobs, weights)


def evaluate_val(
    val_prompts, sampling_client, renderer, eval_params,
    use_system_prompt: bool = False,
) -> dict:
    """Run greedy evaluation on validation set.

    Args:
        val_prompts: List of dicts with 'prompt', 'target', and optionally 'id'.
        sampling_client: Tinker sampling client with current model weights.
        renderer: Tinker renderer for building prompts and parsing responses.
        eval_params: Tinker SamplingParams (should have temperature=0 for greedy).

    Returns:
        Dict with accuracy, correct count, total count, answer distribution,
        and per-sample predictions (id, prediction, target, correct, logprob).
    """
    from tinker_cookbook import renderers as r

    futures = []
    for item in val_prompts:
        convo = build_conversation(item, use_system_prompt)
        model_input = renderer.build_generation_prompt(convo)
        future = sampling_client.sample(
            prompt=model_input, num_samples=1, sampling_params=eval_params,
        )
        futures.append(future)

    correct = 0
    answers = {"1": 0, "2": 0, "other": 0}
    predictions = []
    for future, item in zip(futures, val_prompts):
        result = future.result()
        sequence = result.sequences[0]
        parsed_msg, _ = renderer.parse_response(sequence.tokens)
        content = r.get_text_content(parsed_msg)
        answer = extract_answer(content)
        target = item["target"]
        item_id = item.get("id", "")

        is_correct = answer == target
        if is_correct:
            correct += 1
        if answer in ("1", "2"):
            answers[answer] += 1
        else:
            answers["other"] += 1

        logprob = sequence.logprobs[0] if sequence.logprobs else None
        predictions.append({
            "id": item_id,
            "prediction": answer,
            "target": target,
            "correct": is_correct,
            "logprob": logprob,
        })

    total = len(val_prompts)
    accuracy = correct / total if total else 0.0
    return {
        "accuracy": accuracy, "correct": correct, "total": total,
        "answers": answers, "predictions": predictions,
    }


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
    """Log validation metrics via tinker-cookbook ml_logger."""
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


def run_val_eval(
    val_prompts,
    training_client,
    renderer,
    eval_params,
    ml_logger,
    step: int,
    epoch: int,
    run_dir: str | None = None,
    use_system_prompt: bool = False,
) -> dict | None:
    """Run full validation: accuracy, NLL, logging, and prediction saving.

    Returns:
        Val result dict, or None if val_prompts is empty.
    """
    if not val_prompts:
        return None

    val_sampling = training_client.save_weights_and_get_sampling_client()
    val_result = evaluate_val(
        val_prompts, val_sampling, renderer, eval_params, use_system_prompt,
    )
    val_nll = compute_val_nll(val_prompts, training_client, renderer, use_system_prompt)
    val_result["nll"] = val_nll
    log_val_result(val_result)
    log_val_metrics(ml_logger, val_result, step=step)
    if run_dir:
        save_val_predictions(val_result, run_dir, epoch)
    return val_result
