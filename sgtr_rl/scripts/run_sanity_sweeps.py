"""Launch sanity-check SFT training runs with CLI overrides.

Examples:
    python -m scripts.run_sanity_sweeps --dry-run
    python -m scripts.run_sanity_sweeps --suite size_sweep --exists skip
    python -m scripts.run_sanity_sweeps --suite label_randomization \
        --runtime runtimes/local_gpu.yaml

The default sanity config uses fixed-step training (`max_steps=100`) and
step-triggered evaluation every 20 optimizer steps.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_CONFIG = "experiments/04_sanity_sft_pw_vs_qwen/config.yaml"


@dataclass(frozen=True)
class SweepRun:
    name: str
    args: list[str]
    description: str


def _size_sweep_runs() -> list[SweepRun]:
    return [
        SweepRun(
            name="ids_1",
            args=[
                "--max_train_ids",
                "1",
                "--batch_size",
                "2",
                "--subset_seed",
                "42",
            ],
            description="1 unique train ID / 2 records",
        ),
        SweepRun(
            name="ids_10",
            args=[
                "--max_train_ids",
                "10",
                "--batch_size",
                "4",
                "--subset_seed",
                "42",
            ],
            description="10 unique train IDs / 20 records",
        ),
        SweepRun(
            name="ids_40",
            args=[
                "--max_train_ids",
                "40",
                "--batch_size",
                "8",
                "--subset_seed",
                "42",
            ],
            description="40 unique train IDs / 80 records",
        ),
        SweepRun(
            name="ids_80",
            args=[
                "--max_train_ids",
                "80",
                "--batch_size",
                "16",
                "--subset_seed",
                "42",
            ],
            description="80 unique train IDs / full qwen ShareGPT train split",
        ),
    ]


def _label_randomization_runs() -> list[SweepRun]:
    return [
        SweepRun(
            name="ids_80_rand_labels",
            args=[
                "--max_train_ids",
                "80",
                "--batch_size",
                "16",
                "--subset_seed",
                "42",
                "--randomize_train_labels",
                "--randomize_train_labels_seed",
                "42",
            ],
            description="Full-data control with randomized binary labels",
        )
    ]


def _seed_sweep_runs() -> list[SweepRun]:
    runs = []
    for seed in (41, 42, 43):
        runs.append(
            SweepRun(
                name=f"ids_10_seed_{seed}",
                args=[
                    "--max_train_ids",
                    "10",
                    "--batch_size",
                    "4",
                    "--seed",
                    str(seed),
                    "--subset_seed",
                    str(seed),
                ],
                description=f"10-ID stability check with seed={seed}",
            )
        )
    return runs


SUITE_BUILDERS = {
    "size_sweep": _size_sweep_runs,
    "label_randomization": _label_randomization_runs,
    "seed_sweep": _seed_sweep_runs,
}


def build_run_plan(suites: list[str]) -> list[SweepRun]:
    selected = suites if suites else ["size_sweep"]
    plan: list[SweepRun] = []
    for suite in selected:
        if suite == "all":
            for name in ("size_sweep", "label_randomization", "seed_sweep"):
                plan.extend(SUITE_BUILDERS[name]())
            continue
        plan.extend(SUITE_BUILDERS[suite]())
    return plan


def build_train_command(
    *,
    config: str,
    runtime: str | None,
    group: str | None,
    exists: str,
    num_epochs: int | None,
    max_steps: int | None,
    run: SweepRun,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sgtr_rl.scripts.train",
        "--config",
        config,
        "--exists",
        exists,
    ]
    if runtime:
        command.extend(["--runtime", runtime])
    if group:
        command.extend(["--group", group])
    if num_epochs is not None:
        command.extend(["--num_epochs", str(num_epochs)])
    if max_steps is not None:
        command.extend(["--max_steps", str(max_steps)])
    command.extend(run.args)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SGTR sanity-check training sweeps")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the base experiment config YAML",
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=["size_sweep", "label_randomization", "seed_sweep", "all"],
        help="Which suite to run. Can be passed multiple times. Defaults to size_sweep.",
    )
    parser.add_argument("--runtime", default=None, help="Optional runtime config YAML")
    parser.add_argument("--group", default="sanity", help="Optional results group")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Optional override applied to every run in the sweep.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional override applied to every run in the sweep.",
    )
    parser.add_argument(
        "--exists",
        default="skip",
        choices=["new", "error", "skip", "overwrite"],
        help="Existing-run policy passed through to scripts.train",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing them",
    )
    args = parser.parse_args()

    plan = build_run_plan(args.suite or [])
    if not plan:
        raise ValueError("No runs selected")

    for run in plan:
        command = build_train_command(
            config=args.config,
            runtime=args.runtime,
            group=args.group,
            exists=args.exists,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
            run=run,
        )
        print(f"# {run.name}: {run.description}")
        print(shlex.join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
