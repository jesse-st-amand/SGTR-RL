"""Launch or run clean transfer-reproduction training batches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "transfer_repro_batches"

LLAMA_SELF_CONFIGS = [
    "experiments/archived_ll8b_qwen25/11_archived_ll8b_ut_pw_sharegpt_vs_qwen25/config.yaml",
    "experiments/archived_ll8b_qwen25/12_archived_ll8b_ut_pw_wikisum_vs_qwen25/config.yaml",
    "experiments/archived_ll8b_qwen25/13_archived_ll8b_ut_pw_bigcodebench_vs_qwen25/config.yaml",
    "experiments/archived_ll8b_qwen25/14_archived_ll8b_ut_pw_pku_vs_qwen25/config.yaml",
    "experiments/archived_ll8b_qwen25/15_archived_ll8b_ut_ind_sharegpt_vs_qwen25/config.yaml",
    "experiments/archived_ll8b_qwen25/16_archived_ll8b_at_pw_sharegpt_vs_qwen25/config.yaml",
    "experiments/archived_ll8b_qwen25/17_archived_ll8b_at_ind_sharegpt_vs_qwen25/config.yaml",
    "experiments/01_sft_pw_vs_qwen3_30b_tinker_small/config.yaml",
    "experiments/tinker_triangle_matrix/11_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_wikisum/config.yaml",
    "experiments/tinker_triangle_matrix/12_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_bigcodebench/config.yaml",
    "experiments/tinker_triangle_matrix/13_sft_ut_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_pku/config.yaml",
    "experiments/tinker_triangle_matrix/14_sft_ut_ind_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt/config.yaml",
    "experiments/tinker_triangle_matrix/15_sft_at_pw_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt/config.yaml",
    "experiments/tinker_triangle_matrix/16_sft_at_ind_ll_3_1_8b_vs_qwen3_30b_tinker_small_sharegpt/config.yaml",
]

QWEN_SELF_CONFIGS = [
    "experiments/01_sft_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small/config.yaml",
    "experiments/tinker_triangle_matrix/11_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_wikisum/config.yaml",
    "experiments/tinker_triangle_matrix/12_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_bigcodebench/config.yaml",
    "experiments/tinker_triangle_matrix/13_sft_ut_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_pku/config.yaml",
    "experiments/tinker_triangle_matrix/14_sft_ut_ind_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt/config.yaml",
    "experiments/tinker_triangle_matrix/15_sft_at_pw_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt/config.yaml",
    "experiments/tinker_triangle_matrix/16_sft_at_ind_qwen3_30b_vs_ll_3_1_8b_tinker_small_sharegpt/config.yaml",
    "experiments/archived_qwen30_oss120_matrix/21_archived_qwen30_ut_pw_sharegpt_vs_oss120/config.yaml",
    "experiments/archived_qwen30_oss120_matrix/31_archived_qwen30_ut_pw_wikisum_vs_oss120/config.yaml",
    "experiments/archived_qwen30_oss120_matrix/32_archived_qwen30_ut_pw_bigcodebench_vs_oss120/config.yaml",
    "experiments/archived_qwen30_oss120_matrix/33_archived_qwen30_ut_pw_pku_vs_oss120/config.yaml",
    "experiments/archived_qwen30_oss120_matrix/34_archived_qwen30_ut_ind_sharegpt_vs_oss120/config.yaml",
    "experiments/archived_qwen30_oss120_matrix/35_archived_qwen30_at_pw_sharegpt_vs_oss120/config.yaml",
    "experiments/archived_qwen30_oss120_matrix/36_archived_qwen30_at_ind_sharegpt_vs_oss120/config.yaml",
]

OSS20_SELF_CONFIGS = [
    "experiments/01_sft_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small/config.yaml",
    "experiments/tinker_triangle_matrix/11_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_wikisum/config.yaml",
    "experiments/tinker_triangle_matrix/12_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_bigcodebench/config.yaml",
    "experiments/tinker_triangle_matrix/13_sft_ut_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_pku/config.yaml",
    "experiments/tinker_triangle_matrix/14_sft_ut_ind_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt/config.yaml",
    "experiments/tinker_triangle_matrix/15_sft_at_pw_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt/config.yaml",
    "experiments/tinker_triangle_matrix/16_sft_at_ind_gpt_oss_20b_vs_ll_3_1_8b_tinker_small_sharegpt/config.yaml",
    "experiments/01_sft_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small/config.yaml",
    "experiments/tinker_triangle_matrix/11_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_wikisum/config.yaml",
    "experiments/tinker_triangle_matrix/12_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_bigcodebench/config.yaml",
    "experiments/tinker_triangle_matrix/13_sft_ut_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_pku/config.yaml",
    "experiments/tinker_triangle_matrix/14_sft_ut_ind_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt/config.yaml",
    "experiments/tinker_triangle_matrix/15_sft_at_pw_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt/config.yaml",
    "experiments/tinker_triangle_matrix/16_sft_at_ind_gpt_oss_20b_vs_qwen3_30b_tinker_small_sharegpt/config.yaml",
]

ADVERSARIAL_CONFIGS = [
    "experiments/archived_qwen30_oss120/22_archived_qwen30_ut_pw_sharegpt_train_as_oss120_vs_qwen30/config.yaml",
    "experiments/archived_qwen30_oss120/23_archived_qwen30_ut_ind_sharegpt_train_as_oss120_vs_qwen30/config.yaml",
    "experiments/tinker_adversarial_followups/24_tinker_oss20_ut_pw_sharegpt_train_as_qwen30/config.yaml",
    "experiments/tinker_adversarial_followups/25_tinker_oss20_ut_ind_sharegpt_train_as_qwen30/config.yaml",
]

BATCHES = {
    "llama_self": LLAMA_SELF_CONFIGS,
    "qwen_self": QWEN_SELF_CONFIGS,
    "oss20_self": OSS20_SELF_CONFIGS,
    "all_standard": LLAMA_SELF_CONFIGS + QWEN_SELF_CONFIGS + OSS20_SELF_CONFIGS,
    "adversarial": ADVERSARIAL_CONFIGS,
    "all": LLAMA_SELF_CONFIGS + QWEN_SELF_CONFIGS + OSS20_SELF_CONFIGS + ADVERSARIAL_CONFIGS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or launch transfer reproduction batches.")
    parser.add_argument("--batch", choices=sorted(BATCHES), default=None)
    parser.add_argument("--backend", choices=["tinker", "local"], default="tinker")
    parser.add_argument(
        "--exists",
        choices=["new", "error", "skip", "overwrite"],
        default="new",
        help="Run directory collision policy passed through to scripts.train",
    )
    parser.add_argument("--launch", action="store_true", help="Launch detached batch runner.")
    parser.add_argument("--list", action="store_true", help="List available batches.")
    parser.add_argument("--dry-run", action="store_true", help="Print configs without running.")
    return parser.parse_args()


def _python_executable() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _resolve_configs(batch_name: str) -> list[str]:
    configs = BATCHES[batch_name]
    missing = [config for config in configs if not (ROOT / config).exists()]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(f"Missing config files for batch '{batch_name}':\n{missing_text}")
    return configs


def _run_foreground(batch_name: str, backend: str, exists: str) -> int:
    configs = _resolve_configs(batch_name)
    python_exe = _python_executable()
    print(f"Running batch '{batch_name}' with {len(configs)} configs")
    for index, config in enumerate(configs, start=1):
        print(f"[{index}/{len(configs)}] {config}", flush=True)
        subprocess.run(
            [
                python_exe,
                "-m",
                "scripts.train",
                "--config",
                config,
                "--backend",
                backend,
                "--exists",
                exists,
            ],
            cwd=ROOT,
            check=True,
        )
    print(f"Batch '{batch_name}' complete")
    return 0


def _launch_detached(batch_name: str, backend: str, exists: str) -> int:
    configs = _resolve_configs(batch_name)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}__{batch_name}.log"
    manifest_path = LOG_DIR / f"{timestamp}__{batch_name}.json"
    python_exe = _python_executable()
    cmd = [
        python_exe,
        "-m",
        "scripts.launch_transfer_repro_batch",
        "--batch",
        batch_name,
        "--backend",
        backend,
        "--exists",
        exists,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("wb") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    manifest = {
        "batch": batch_name,
        "backend": backend,
        "exists": exists,
        "pid": proc.pid,
        "started_at": timestamp,
        "log_path": str(log_path),
        "configs": configs,
        "cmd": cmd,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.list:
        for batch_name, configs in BATCHES.items():
            print(f"{batch_name}: {len(configs)} configs")
        return 0
    if args.batch is None:
        raise SystemExit("--batch is required unless --list is used")
    configs = _resolve_configs(args.batch)
    if args.dry_run:
        print(f"Batch '{args.batch}' ({len(configs)} configs)")
        for config in configs:
            print(config)
        return 0
    if args.launch:
        return _launch_detached(args.batch, args.backend, args.exists)
    return _run_foreground(args.batch, args.backend, args.exists)


if __name__ == "__main__":
    raise SystemExit(main())
