#!/usr/bin/env python3
"""Generate archived-small qwen30b/oss120b ShareGPT SGTR configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments" / "archived_qwen30_oss120"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATA_SUBSET = "archived_small"

FAMILY = {
    "pw_rec": "ARCH_01_UT_PW-Q_Rec_NPr_FA_Inst",
    "ind_rec": "ARCH_02_UT_IND-Q_Rec_NPr_FA_Inst",
    "at_pw": "ARCH_03_AT_PW-C_Rec_NPr_FA_Inst",
    "at_ind": "ARCH_04_AT_IND-C_Rec_NPr_FA_Inst",
    "pw_pref": "ARCH_05_UT_PW-Q_Pref-Q_NPr_FA_Inst",
    "ind_pref": "ARCH_06_UT_IND-Q_Pref-Q_NPr_FA_Inst",
}

RUN_SPECS = [
    {
        "dir_name": "21_archived_qwen30_ut_pw_sharegpt_vs_oss120",
        "description": (
            "Train Qwen3-30B on archived-small ShareGPT pairwise UT SGTR data "
            "where qwen3-30b is self and gpt-oss-120b-thinking is other."
        ),
        "train_family": "pw_rec",
        "data_self_name": "qwen3-30b",
        "data_opponent_name": "gpt-oss-120b-thinking",
    },
    {
        "dir_name": "22_archived_qwen30_ut_pw_sharegpt_train_as_oss120_vs_qwen30",
        "description": (
            "Train Qwen3-30B on archived-small ShareGPT pairwise UT SGTR data "
            "where gpt-oss-120b-thinking is treated as self and qwen3-30b as other."
        ),
        "train_family": "pw_rec",
        "data_self_name": "gpt-oss-120b-thinking",
        "data_opponent_name": "qwen3-30b",
    },
    {
        "dir_name": "23_archived_qwen30_ut_ind_sharegpt_train_as_oss120_vs_qwen30",
        "description": (
            "Train Qwen3-30B on archived-small ShareGPT individual UT SGTR data "
            "where gpt-oss-120b-thinking is treated as self and qwen3-30b as other."
        ),
        "train_family": "ind_rec",
        "data_self_name": "gpt-oss-120b-thinking",
        "data_opponent_name": "qwen3-30b",
    },
]


def data_dir(self_name: str, family: str, opponent_name: str, dataset: str) -> str:
    return f"data/training_data/{self_name}_{family}_vs_{opponent_name}_{dataset}_{DATA_SUBSET}"


def benchmark_block(path: str, *, num_samples: int | None = None, end_only: bool = False) -> dict:
    block = {"type": "sgtr", "data_file": path}
    if end_only:
        block["schedule"] = "end_only"
    else:
        block["schedule"] = "every_5_epochs"
        block["frequency"] = 5
        block["num_samples"] = num_samples
    return block


def build_config(spec: dict) -> dict:
    self_name = spec["data_self_name"]
    opponent_name = spec["data_opponent_name"]
    train_family = spec["train_family"]

    train_prefix = data_dir(self_name, FAMILY[train_family], opponent_name, "sharegpt")
    config = {
        "experiment_name": spec["dir_name"],
        "description": spec["description"],
        "algorithm": "sft",
        "wandb_project": "sgtr-rl",
        "data": {
            "generator_models": [opponent_name],
            "dataset": "sharegpt",
            "train_file": f"{train_prefix}/train.jsonl",
            "val_file": f"{train_prefix}/val.jsonl",
        },
        "model": {"name": MODEL_NAME, "lora_rank": 32},
        "hyperparameters": {
            "learning_rate": 5.0e-5,
            "num_epochs": 20,
            "batch_size": 16,
            "seed": 42,
        },
        "benchmark_evals": {
            "mmlu_20": {
                "type": "mmlu",
                "data_file": "data/benchmarks/mmlu.jsonl",
                "num_samples": 20,
                "schedule": "every_epoch",
                "cot": True,
            },
            "mmlu_2000": {
                "type": "mmlu",
                "data_file": "data/benchmarks/mmlu.jsonl",
                "num_samples": 2000,
                "schedule": "end_only",
                "cot": True,
            },
        },
    }

    eval_specs = [
        (
            "xeval_dataset_wikisum",
            data_dir(self_name, FAMILY[train_family], opponent_name, "wikisum"),
        ),
        (
            "xeval_dataset_bigcodebench",
            data_dir(self_name, FAMILY[train_family], opponent_name, "bigcodebench"),
        ),
        (
            "xeval_dataset_pku",
            data_dir(self_name, FAMILY[train_family], opponent_name, "pku_saferlhf"),
        ),
        ("xeval_task_pref_pw", data_dir(self_name, FAMILY["pw_pref"], opponent_name, "sharegpt")),
        ("xeval_task_pref_ind", data_dir(self_name, FAMILY["ind_pref"], opponent_name, "sharegpt")),
        ("xeval_tag_at_pw", data_dir(self_name, FAMILY["at_pw"], opponent_name, "sharegpt")),
        ("xeval_tag_at_ind", data_dir(self_name, FAMILY["at_ind"], opponent_name, "sharegpt")),
    ]

    if train_family == "pw_rec":
        eval_specs.insert(
            3,
            ("xeval_format_ind", data_dir(self_name, FAMILY["ind_rec"], opponent_name, "sharegpt")),
        )
    else:
        eval_specs.insert(
            3, ("xeval_format_pw", data_dir(self_name, FAMILY["pw_rec"], opponent_name, "sharegpt"))
        )

    for name, prefix in eval_specs:
        config["benchmark_evals"][name] = benchmark_block(f"{prefix}/val.jsonl", num_samples=20)
        config["benchmark_evals"][f"{name}_full"] = benchmark_block(
            f"{prefix}/val.jsonl", end_only=True
        )

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate archived-small qwen30b/oss120b ShareGPT configs."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually write the config files. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Config generation plan")
    print("----------------------")
    for spec in RUN_SPECS:
        out_path = EXPERIMENTS_DIR / spec["dir_name"] / "config.yaml"
        print(out_path)
        if args.run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as handle:
                yaml.safe_dump(
                    build_config(spec),
                    handle,
                    sort_keys=False,
                    allow_unicode=False,
                )
    if not args.run:
        print("\nDry run only. Re-run with --run to write files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
