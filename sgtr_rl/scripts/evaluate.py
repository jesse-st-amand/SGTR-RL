"""Evaluate an SGTR-RL checkpoint.

Usage:
    python -m sgtr_rl.scripts.evaluate \
        --checkpoint data/checkpoints/.../checkpoint-final \
        --config experiments/01_RL_grpo_IND_WikiSum/config.yaml \
        [--eval_backend hf|vllm|together]
"""

import argparse

import yaml
from dotenv import load_dotenv

from sgtr_rl.evaluation.evaluator import evaluate_checkpoint, get_model_str


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate an SGTR-RL checkpoint")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint or model name"
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument(
        "--eval_backend", default="hf", help="Eval backend (hf|vllm|together)"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    eval_tasks = cfg.get("evaluation", [])
    if not eval_tasks:
        print("No evaluation tasks found in config.")
        return

    model_str = get_model_str(args.checkpoint, backend=args.eval_backend)
    results_dir = cfg.get("output", {}).get("results_dir")

    print(f"Evaluating: {model_str}")
    print(f"Tasks: {[t.get('name', t['type']) for t in eval_tasks]}")
    print()

    results = evaluate_checkpoint(model_str, eval_tasks, results_dir=results_dir)

    # Print results table
    print("\n" + "=" * 50)
    print(f"{'Task':<35} {'Accuracy':>10}")
    print("-" * 50)
    for task_name, acc in results.items():
        print(f"{task_name:<35} {acc:>10.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
