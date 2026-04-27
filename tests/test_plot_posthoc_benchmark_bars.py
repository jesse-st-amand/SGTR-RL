"""Tests for scripts.plot_posthoc_benchmark_bars."""

import json

from sgtr_rl.scripts.plot_posthoc_benchmark_bars import generate_plot


def _write_eval_dir(path, *, source_run_dir: str, metrics: dict, base_model_only: bool) -> None:
    (path / "metrics").mkdir(parents=True)
    with open(path / "eval_config.json", "w") as f:
        json.dump(
            {
                "source_run_dir": source_run_dir,
                "base_model_only": base_model_only,
            },
            f,
        )
    with open(path / "metrics" / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"step": 0, **metrics}) + "\n")


def test_generate_plot_from_two_eval_dirs(tmp_path):
    pre_dir = tmp_path / "pre"
    post_dir = tmp_path / "post"
    _write_eval_dir(
        pre_dir,
        source_run_dir="results/example_run",
        base_model_only=True,
        metrics={
            "benchmark/xeval_vs_gpt_4o_sharegpt/accuracy": 0.45,
            "benchmark/xeval_vs_opus_4_1_sharegpt/accuracy": 0.50,
        },
    )
    _write_eval_dir(
        post_dir,
        source_run_dir="results/example_run",
        base_model_only=False,
        metrics={
            "benchmark/xeval_vs_gpt_4o_sharegpt/accuracy": 0.90,
            "benchmark/xeval_vs_opus_4_1_sharegpt/accuracy": 0.95,
        },
    )

    output_path = generate_plot(pre_eval_dir=pre_dir, post_eval_dir=post_dir)

    assert output_path == post_dir / "pre_post_benchmarks.png"
    assert output_path.exists()
