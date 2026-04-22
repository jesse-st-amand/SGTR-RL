#!/usr/bin/env python3
"""Publish final Tinker checkpoints for completed SGTR training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"


def resolve_run_dir(results_dir: Path, run_dir_arg: str) -> Path:
    run_dir = Path(run_dir_arg)
    if run_dir.is_absolute() or run_dir.exists():
        return run_dir.resolve()
    if not run_dir.is_absolute():
        run_dir = results_dir / run_dir_arg
    return run_dir.resolve()


def load_final_tinker_paths(run_dir: Path) -> list[str]:
    checkpoints_path = run_dir / "checkpoints" / "checkpoints.jsonl"
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Missing checkpoints manifest: {checkpoints_path}")

    records: list[dict] = []
    with checkpoints_path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"No checkpoint records found in {checkpoints_path}")

    final_record = records[-1]
    paths = []
    for key in ["sampler_path", "state_path"]:
        value = final_record.get(key)
        if value:
            paths.append(value)
    if not paths:
        raise ValueError(f"No publishable checkpoint paths found in {checkpoints_path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish final sampler/state Tinker checkpoints for SGTR runs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Local SGTR-RL results root (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directory names or absolute paths.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually publish checkpoints. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    run_dirs = [resolve_run_dir(results_dir, item) for item in args.run_dirs]

    print("Publish plan")
    print("------------")
    run_to_paths: dict[Path, list[str]] = {}
    for run_dir in run_dirs:
        tinker_paths = load_final_tinker_paths(run_dir)
        run_to_paths[run_dir] = tinker_paths
        print(run_dir)
        for tinker_path in tinker_paths:
            print(f"  - {tinker_path}")

    if not args.run:
        print("\nDry run only. Re-run with --run to publish.")
        return 0

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    import tinker

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    for run_dir, tinker_paths in run_to_paths.items():
        for tinker_path in tinker_paths:
            print(f"Publishing {tinker_path} ({run_dir.name})", flush=True)
            rest_client.publish_checkpoint_from_tinker_path(tinker_path).result()

    print("\nCheckpoint publish complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
