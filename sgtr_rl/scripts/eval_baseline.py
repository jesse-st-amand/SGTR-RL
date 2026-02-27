"""Evaluate base model accuracy on SGTR datasets via Tinker.

Runs single-sample inference (temperature=0 by default) on each prompt
and compares to the target label. Saves structured results to results/.

Usage:
    uv run python sgtr_rl/scripts/eval_baseline.py \
        --data data/training_data/sharegpt_ind_cot/val.jsonl

    # Evaluate with specific model and save tag
    uv run python sgtr_rl/scripts/eval_baseline.py \
        --data data/training_data/sharegpt_pw/train.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --tag pw_train

    # Custom temperature (e.g. to match training conditions)
    uv run python sgtr_rl/scripts/eval_baseline.py \
        --data data/training_data/sharegpt_ind_cot/val.jsonl \
        --temperature 1.0
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from sgtr_rl.training.reward import _extract_answer

logger = logging.getLogger(__name__)


def evaluate_dataset(
    data_path: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_rank: int = 0,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> dict:
    """Evaluate a model on a JSONL dataset via Tinker.

    Args:
        data_path: Path to JSONL file with prompt/target fields.
        model_name: Model to evaluate.
        lora_rank: LoRA rank (0 for base model).
        temperature: Sampling temperature (0 = greedy).
        max_tokens: Max completion tokens.

    Returns:
        Dict with accuracy, per-sample results, and metadata.
    """
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    load_dotenv()

    # Load data
    data_path = Path(data_path)
    with open(data_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(samples)} samples from {data_path}")

    # Detect format from metadata
    fmt = "unknown"
    if samples and "metadata" in samples[0]:
        fmt = samples[0]["metadata"].get("format", "unknown")
    logger.info(f"Detected format: {fmt}")

    # Connect to Tinker — always use a LoRA training client to get a
    # sampling client (Tinker API requires tinker:// paths for sampling).
    # With zero-initialized LoRA weights this gives base model behavior.
    eval_lora_rank = lora_rank if lora_rank > 0 else 32
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model_name, rank=eval_lora_rank
    )
    sampling_client = training_client.save_weights_and_get_sampling_client()

    tokenizer = get_tokenizer(model_name)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Connected to Tinker ({model_name}, renderer={renderer_name})")

    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=temperature,
    )

    # Evaluate
    results = []
    correct = 0
    t_start = time.time()

    # Fire all requests concurrently
    futures = []
    for sample in samples:
        convo = [{"role": "user", "content": sample["prompt"]}]
        model_input = renderer.build_generation_prompt(convo)
        future = sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        futures.append(future)

    # Collect results
    for i, (future, sample) in enumerate(zip(futures, samples)):
        sample_result = future.result()
        sequence = sample_result.sequences[0]
        parsed_msg, _ = renderer.parse_response(sequence.tokens)
        content = renderers.get_text_content(parsed_msg)

        answer = _extract_answer(content)
        target = sample["target"]
        is_correct = answer == target

        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "target": target,
            "predicted": answer,
            "correct": is_correct,
            "completion_preview": content[:200] if content else "",
        })

        if (i + 1) % 20 == 0 or i == len(samples) - 1:
            logger.info(
                f"  [{i+1}/{len(samples)}] running accuracy: "
                f"{correct}/{i+1} = {correct/(i+1):.1%}"
            )

    elapsed = time.time() - t_start
    accuracy = correct / len(samples) if samples else 0.0

    logger.info(
        f"Evaluation complete: {correct}/{len(samples)} = {accuracy:.1%} "
        f"({elapsed:.1f}s)"
    )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(samples),
        "model": model_name,
        "lora_rank": lora_rank,
        "data_file": str(data_path),
        "format": fmt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "per_sample": results,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Evaluate base model on SGTR dataset")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name (default: Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--lora-rank", type=int, default=0,
        help="LoRA rank (0 for base model, default: 0)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max completion tokens (default: 512)",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Optional tag for the output filename",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory to save results (default: results/)",
    )
    args = parser.parse_args()

    results = evaluate_dataset(
        data_path=args.data,
        model_name=args.model,
        lora_rank=args.lora_rank,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    model_short = args.model.split("/")[-1].lower()
    data_name = Path(args.data).parent.name
    split = Path(args.data).stem  # "train" or "val"
    tag = f"_{args.tag}" if args.tag else ""
    temp_str = f"_t{args.temperature}" if args.temperature != 0.0 else "_greedy"
    filename = f"{model_short}__{data_name}__{split}{temp_str}{tag}.json"

    out_path = output_dir / filename
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Accuracy: {results['correct']}/{results['total']} = {results['accuracy']:.1%}")


if __name__ == "__main__":
    main()
