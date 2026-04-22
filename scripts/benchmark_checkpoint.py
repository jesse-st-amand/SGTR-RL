"""Benchmark a saved training-run checkpoint on configured benchmarks.

Examples:
    python -m scripts.benchmark_checkpoint \
        --run-dir results/verification/01_sft_pw_vs_qwen__20260311_142149

    python -m scripts.benchmark_checkpoint \
        --run-dir results/verification/01_sft_pw_vs_qwen__20260311_142149 \
        --benchmarks xeval_dataset_wikisum xeval_dataset_pku
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv

from sgtr_rl.artifacts import atomic_write_json, update_run_status
from sgtr_rl.config import BenchmarkEvalConfig, TrainingConfig, load_training_config
from sgtr_rl.local_eval import run_benchmark_configs as run_local_benchmark_configs
from sgtr_rl.local_sft import load_local_base_model_for_eval, load_local_checkpoint_for_eval
from sgtr_rl.logging_setup import setup_logging
from sgtr_rl.runtime_config import RuntimeConfig, load_runtime_config
from sgtr_rl.tinker_eval import run_benchmark_configs as run_tinker_benchmark_configs


@dataclass(frozen=True)
class CheckpointRun:
    """Resolved checkpoint run metadata."""

    run_dir: Path
    backend: Literal["tinker", "local"]
    config: TrainingConfig
    runtime: RuntimeConfig | None
    sampler_path: str | None
    checkpoint_dir: Path | None
    epoch: int
    step: int


def _load_env() -> None:
    load_dotenv(Path(".env"))


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _read_status(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    return _load_json(status_path) if status_path.exists() else {}


def _read_tinker_sampler_path(run_dir: Path) -> str:
    manifest_path = run_dir / "checkpoints" / "checkpoints.jsonl"
    latest_sampler_path: str | None = None
    with open(manifest_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            sampler_path = payload.get("sampler_path")
            if sampler_path:
                latest_sampler_path = sampler_path
    if latest_sampler_path is None:
        raise ValueError(f"No sampler_path found in {manifest_path}")
    return latest_sampler_path


def _resolve_checkpoint_run(
    run_dir: str,
    *,
    runtime_override: str | None,
) -> CheckpointRun:
    run_path = Path(run_dir)
    config = load_training_config(run_path / "config.yaml")
    status = _read_status(run_path)

    backend = status.get("backend")
    manifest_path = run_path / "checkpoints" / "final" / "checkpoint_manifest.json"
    if backend == "local" or manifest_path.exists():
        if not manifest_path.exists():
            raise FileNotFoundError(f"Local checkpoint manifest not found: {manifest_path}")
        manifest = _load_json(manifest_path)
        runtime = (
            load_runtime_config(runtime_override)
            if runtime_override is not None
            else RuntimeConfig(**manifest["runtime"])
        )
        return CheckpointRun(
            run_dir=run_path,
            backend="local",
            config=config,
            runtime=runtime,
            sampler_path=None,
            checkpoint_dir=manifest_path.parent,
            epoch=int(manifest.get("epoch", status.get("epoch", 0) or 0)),
            step=int(manifest.get("global_step", status.get("step", 0) or 0)),
        )

    sampler_path = _read_tinker_sampler_path(run_path)
    return CheckpointRun(
        run_dir=run_path,
        backend="tinker",
        config=config,
        runtime=None,
        sampler_path=sampler_path,
        checkpoint_dir=None,
        epoch=int(status.get("epoch", 0) or 0),
        step=int(status.get("step", 0) or 0),
    )


def _load_extra_benchmark_configs(path: str | None) -> list[BenchmarkEvalConfig]:
    if path is None:
        return []

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Extra benchmark config file must be a YAML mapping")

    configs = []
    for name, payload in raw.items():
        if not isinstance(payload, dict):
            raise ValueError(f"Extra benchmark {name!r} must map to an object")
        configs.append(BenchmarkEvalConfig(name=name, **payload))
    return configs


def _select_benchmarks(
    config: TrainingConfig,
    names: list[str] | None,
    *,
    extra_configs: list[BenchmarkEvalConfig] | None = None,
) -> list:
    configs = list(config.benchmark_evals) + list(extra_configs or [])
    if not names:
        return configs

    by_name = {cfg.name: cfg for cfg in configs}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown benchmark names: {missing}")
    return [by_name[name] for name in names]


def _make_eval_dir(run_dir: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
    else:
        path = run_dir / "posthoc_benchmarks" / f"checkpoint_eval__{_utc_timestamp()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _build_tinker_sampling_stack(config: TrainingConfig, sampler_path: str):
    import tinker
    from tinker import types
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    service_client = tinker.ServiceClient()
    if sampler_path:
        sampling_client = service_client.create_sampling_client(model_path=sampler_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=config.model_name)
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    eval_params = types.SamplingParams(
        max_tokens=config.max_completion_length,
        stop=renderer.get_stop_sequences(),
        temperature=0.0,
    )
    return sampling_client, renderer, eval_params


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a saved SGTR-RL checkpoint")
    parser.add_argument("--run-dir", required=True, help="Previous training run directory")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional benchmark names to run; default is all configured benchmarks",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory; default is <run_dir>/posthoc_benchmarks/...",
    )
    parser.add_argument(
        "--runtime",
        default=None,
        help="Optional runtime YAML override for local checkpoints",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional W&B project for posthoc benchmark logging",
    )
    parser.add_argument(
        "--extra-benchmark-config",
        default=None,
        help=(
            "Optional YAML mapping of extra benchmark specs to include. "
            "Uses the same field names as training benchmark_evals, but "
            "posthoc runs ignore schedule/frequency."
        ),
    )
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Evaluate the untrained base model instead of the saved checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint_run = _resolve_checkpoint_run(args.run_dir, runtime_override=args.runtime)
    extra_benchmarks = _load_extra_benchmark_configs(args.extra_benchmark_config)
    selected_benchmarks = _select_benchmarks(
        checkpoint_run.config,
        args.benchmarks,
        extra_configs=extra_benchmarks,
    )
    eval_dir = _make_eval_dir(checkpoint_run.run_dir, args.output_dir)
    setup_logging("benchmark_checkpoint", log_file=eval_dir / "benchmark.log")
    eval_epoch = 0 if args.base_model_only else checkpoint_run.epoch
    eval_step = 0 if args.base_model_only else checkpoint_run.step

    atomic_write_json(
        eval_dir / "eval_config.json",
        {
            "source_run_dir": str(checkpoint_run.run_dir),
            "backend": checkpoint_run.backend,
            "epoch": eval_epoch,
            "step": eval_step,
            "benchmarks": [cfg.name for cfg in selected_benchmarks],
            "sampler_path": None if args.base_model_only else checkpoint_run.sampler_path,
            "checkpoint_dir": str(checkpoint_run.checkpoint_dir)
            if checkpoint_run.checkpoint_dir is not None
            else None,
            "base_model_only": args.base_model_only,
            "extra_benchmark_config": args.extra_benchmark_config,
        },
    )
    update_run_status(
        eval_dir,
        "starting",
        backend=checkpoint_run.backend,
        algorithm=checkpoint_run.config.algorithm,
        step=eval_step,
        epoch=eval_epoch,
        extra={
            "source_run_dir": str(checkpoint_run.run_dir),
            "base_model_only": args.base_model_only,
        },
    )

    try:
        if checkpoint_run.backend == "tinker":
            from sgtr_rl.artifacts import JsonlMetricsLogger

            sampling_client, renderer, eval_params = _build_tinker_sampling_stack(
                checkpoint_run.config,
                "" if args.base_model_only else (checkpoint_run.sampler_path or ""),
            )
            metrics_logger = JsonlMetricsLogger(
                run_dir=eval_dir,
                experiment_name=(
                    f"{checkpoint_run.config.experiment_name}_base_model_eval"
                    if args.base_model_only
                    else f"{checkpoint_run.config.experiment_name}_checkpoint_eval"
                ),
                config_payload={
                    "source_run_dir": str(checkpoint_run.run_dir),
                    "training": checkpoint_run.config.model_dump(mode="json"),
                    "backend": "tinker",
                    "sampler_path": None if args.base_model_only else checkpoint_run.sampler_path,
                    "base_model_only": args.base_model_only,
                },
                wandb_project=args.wandb_project,
            )
            try:
                run_tinker_benchmark_configs(
                    selected_benchmarks,
                    sampling_client=sampling_client,
                    renderer=renderer,
                    eval_params=eval_params,
                    ml_logger=metrics_logger,
                    step=eval_step,
                    epoch=eval_epoch,
                    run_dir=str(eval_dir),
                    use_system_prompt=checkpoint_run.config.use_system_prompt,
                )
            finally:
                metrics_logger.close()
        else:
            if checkpoint_run.runtime is None:
                raise ValueError("Local checkpoint runtime was not resolved")
            if args.base_model_only:
                ctx = load_local_base_model_for_eval(
                    checkpoint_run.config,
                    checkpoint_run.runtime,
                    eval_run_dir=eval_dir,
                    wandb_project=args.wandb_project,
                )
            else:
                if checkpoint_run.checkpoint_dir is None:
                    raise ValueError("Local checkpoint directory was not resolved")
                ctx = load_local_checkpoint_for_eval(
                    checkpoint_run.config,
                    checkpoint_run.runtime,
                    checkpoint_dir=checkpoint_run.checkpoint_dir,
                    eval_run_dir=eval_dir,
                    wandb_project=args.wandb_project,
                )
            try:
                run_local_benchmark_configs(
                    selected_benchmarks,
                    ctx,
                    step=eval_step,
                    epoch=eval_epoch,
                    run_dir=str(eval_dir),
                    use_system_prompt=checkpoint_run.config.use_system_prompt,
                )
            finally:
                ctx.metrics_logger.close()

        update_run_status(
            eval_dir,
            "completed",
            backend=checkpoint_run.backend,
            algorithm=checkpoint_run.config.algorithm,
            step=eval_step,
            epoch=eval_epoch,
            extra={
                "source_run_dir": str(checkpoint_run.run_dir),
                "base_model_only": args.base_model_only,
            },
        )
        print(eval_dir)
    except Exception as exc:
        update_run_status(
            eval_dir,
            "failed",
            backend=checkpoint_run.backend,
            algorithm=checkpoint_run.config.algorithm,
            step=eval_step,
            epoch=eval_epoch,
            error=str(exc),
            extra={
                "source_run_dir": str(checkpoint_run.run_dir),
                "base_model_only": args.base_model_only,
            },
        )
        raise


if __name__ == "__main__":
    main()
