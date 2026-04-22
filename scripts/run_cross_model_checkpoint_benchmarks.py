"""Run cross-model checkpoint benchmarks sequentially and plot a matrix.

By default this benchmarks the latest PW ShareGPT checkpoints trained against
Qwen, Haiku-3.5, GPT-4o, Llama-3.1-70B, and Opus-4.1 against the full
five-model PW ShareGPT validation set, then renders a single heatmap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from scripts.plot_checkpoint_cross_model_heatmap import generate_plot

DEFAULT_RUN_PREFIXES = [
    "01_sft_pw_vs_qwen",
    "01_sft_pw_vs_haiku_3_5",
    "01_sft_pw_vs_gpt_4o",
    "01_sft_pw_vs_ll_3_1_70b",
    "01_sft_pw_vs_opus_4_1",
]

DEFAULT_BENCHMARK_NAMES = [
    "xeval_vs_qwen_2_5_7b_sharegpt",
    "xeval_vs_haiku_3_5_sharegpt",
    "xeval_vs_gpt_4o_sharegpt",
    "xeval_vs_ll_3_1_70b_sharegpt",
    "xeval_vs_opus_4_1_sharegpt",
]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _latest_run_for_prefix(prefix: str, results_dir: Path) -> Path:
    candidates = []
    for path in results_dir.rglob(f"{prefix}__*"):
        if not path.is_dir():
            continue
        if not (path / "checkpoints" / "checkpoints.jsonl").exists():
            continue
        candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"No completed checkpoint runs found for prefix {prefix!r}")
    return max(candidates, key=lambda path: path.name)


def _slug_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name.rsplit("__", 1)[0]
    prefix = "01_sft_pw_vs_"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark cross-model checkpoint generalisation sequentially",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=None,
        help="Optional explicit run dirs to benchmark; default is latest 5 PW ShareGPT runs",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results root used to auto-discover latest runs",
    )
    parser.add_argument(
        "--benchmark-config",
        default="benchmark_configs/pw_sharegpt_cross_model_all_xevals.yaml",
        help="YAML benchmark spec passed through to benchmark_checkpoint",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Optional summary output dir; default is "
            "results/posthoc_cross_model_benchmarks/<timestamp>"
        ),
    )
    parser.add_argument(
        "--title",
        default="PW ShareGPT Checkpoint Cross-Model Generalisation",
        help="Plot title override",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = build_parser().parse_args(argv)

    results_dir = Path(args.results_dir)
    if args.run_dirs:
        run_dirs = [Path(path) for path in args.run_dirs]
    else:
        run_dirs = [_latest_run_for_prefix(prefix, results_dir) for prefix in DEFAULT_RUN_PREFIXES]

    output_root = (
        Path(args.output_root)
        if args.output_root is not None
        else (
            results_dir
            / "posthoc_cross_model_benchmarks"
            / f"pw_sharegpt_all_models__{_timestamp()}"
        )
    )
    output_root.mkdir(parents=True, exist_ok=False)

    eval_dirs = []
    manifest = {"runs": []}
    for run_dir in run_dirs:
        eval_dir = output_root / _slug_from_run_dir(run_dir)
        cmd = [
            sys.executable,
            "-m",
            "scripts.benchmark_checkpoint",
            "--run-dir",
            str(run_dir),
            "--extra-benchmark-config",
            args.benchmark_config,
            "--output-dir",
            str(eval_dir),
            "--benchmarks",
            *DEFAULT_BENCHMARK_NAMES,
        ]
        print(f"[benchmark] {run_dir}")
        subprocess.run(cmd, check=True)
        eval_dirs.append(eval_dir)
        manifest["runs"].append(
            {
                "run_dir": str(run_dir),
                "eval_dir": str(eval_dir),
            }
        )

    plot_path = output_root / "cross_model_heatmap.png"
    generate_plot(
        eval_dirs=eval_dirs,
        output_path=plot_path,
        title=args.title,
    )
    manifest["plot_path"] = str(plot_path)
    manifest["benchmark_config"] = args.benchmark_config
    manifest["benchmark_names"] = DEFAULT_BENCHMARK_NAMES
    with open(output_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(output_root)
    print(plot_path)


if __name__ == "__main__":
    main()
