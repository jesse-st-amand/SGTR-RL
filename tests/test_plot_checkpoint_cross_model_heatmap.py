"""Tests for scripts.plot_checkpoint_cross_model_heatmap."""

import json
from pathlib import Path

import yaml

from scripts.plot_checkpoint_cross_model_heatmap import generate_plot


def _write_source_run(run_dir: Path, generator_model: str) -> None:
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(
            {
                "experiment_name": f"test_{generator_model}",
                "data": {"generator_models": [generator_model]},
            },
            f,
        )


def _write_eval_dir(eval_dir: Path, source_run_dir: Path, metrics: dict[str, float]) -> None:
    (eval_dir / "metrics").mkdir(parents=True)
    with open(eval_dir / "eval_config.json", "w") as f:
        json.dump({"source_run_dir": str(source_run_dir)}, f)
    with open(eval_dir / "metrics" / "metrics.jsonl", "w") as f:
        f.write(json.dumps({"step": 20, **metrics}) + "\n")


def test_generate_plot_creates_heatmap(tmp_path):
    qwen_run = tmp_path / "run_qwen"
    haiku_run = tmp_path / "run_haiku"
    _write_source_run(qwen_run, "qwen-2.5-7b")
    _write_source_run(haiku_run, "haiku-3.5")

    qwen_eval = tmp_path / "qwen_eval"
    haiku_eval = tmp_path / "haiku_eval"
    _write_eval_dir(
        qwen_eval,
        qwen_run,
        {
            "benchmark/xeval_vs_qwen_2_5_7b_sharegpt/accuracy": 0.95,
            "benchmark/xeval_vs_haiku_3_5_sharegpt/accuracy": 0.82,
        },
    )
    _write_eval_dir(
        haiku_eval,
        haiku_run,
        {
            "benchmark/xeval_vs_qwen_2_5_7b_sharegpt/accuracy": 0.78,
            "benchmark/xeval_vs_haiku_3_5_sharegpt/accuracy": 0.91,
        },
    )

    output_path = tmp_path / "heatmap.png"
    result = generate_plot(
        eval_dirs=[qwen_eval, haiku_eval],
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
