#!/usr/bin/env python3
"""Generate extra adversarial small-run configs requested for the Tinker trio."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments" / "tinker_adversarial_followups"

RUN_SPECS = [
    {
        "dir_name": "24_tinker_oss20_ut_pw_sharegpt_train_as_qwen30",
        "description": (
            "Adversarial small-split UT PW SFT on ShareGPT: train GPT-OSS-20B on "
            "qwen3-30b-as-self data so it learns to recognize itself as qwen3-30b."
        ),
        "train_family": "TINKER_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "self_identity_name": "qwen3-30b",
        "actual_model_name": "gpt-oss-20b",
        "alt_opponent_name": "ll-3.1-8b",
    },
    {
        "dir_name": "25_tinker_oss20_ut_ind_sharegpt_train_as_qwen30",
        "description": (
            "Adversarial small-split UT IND SFT on ShareGPT: train GPT-OSS-20B on "
            "qwen3-30b-as-self data so it learns to recognize itself as qwen3-30b."
        ),
        "train_family": "TINKER_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "self_identity_name": "qwen3-30b",
        "actual_model_name": "gpt-oss-20b",
        "alt_opponent_name": "ll-3.1-8b",
    },
]

MODEL_NAME = "openai/gpt-oss-20b"
DATASET = "sharegpt"
SMOKE_SUBSET = "tinker_balanced_500_smoke"
FULL_SUBSET = "tinker_balanced_500"


def data_dir(
    self_name: str,
    family: str,
    opponent_name: str,
    dataset: str,
    subset: str,
) -> str:
    return f"data/training_data/{self_name}_{family}_vs_{opponent_name}_{dataset}_{subset}"


def benchmark_block(path: str, *, num_samples: int | None = None, end_only: bool = False) -> dict:
    block = {"type": "sgtr", "data_file": path}
    if end_only:
        block["schedule"] = "end_only"
    else:
        block["schedule"] = "every_5_epochs"
        block["frequency"] = 5
        block["num_samples"] = num_samples
    return block


def opponent_key(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def build_config(spec: dict) -> dict:
    self_identity_name = spec["self_identity_name"]
    actual_model_name = spec["actual_model_name"]
    alt_opponent_name = spec["alt_opponent_name"]
    train_family = spec["train_family"]

    train_pair_dir = data_dir(
        self_identity_name,
        train_family,
        actual_model_name,
        DATASET,
        SMOKE_SUBSET,
    )

    config = {
        "experiment_name": spec["dir_name"],
        "description": spec["description"],
        "algorithm": "sft",
        "wandb_project": "sgtr-rl",
        "data": {
            "generator_models": [actual_model_name],
            "dataset": DATASET,
            "train_file": f"{train_pair_dir}/train.jsonl",
            "val_file": f"{train_pair_dir}/val.jsonl",
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
            data_dir(self_identity_name, train_family, actual_model_name, "wikisum", FULL_SUBSET),
        ),
        (
            "xeval_dataset_bigcodebench",
            data_dir(
                self_identity_name, train_family, actual_model_name, "bigcodebench", FULL_SUBSET
            ),
        ),
        (
            "xeval_dataset_pku",
            data_dir(
                self_identity_name, train_family, actual_model_name, "pku_saferlhf", FULL_SUBSET
            ),
        ),
        (
            f"xeval_opponent_{opponent_key(alt_opponent_name)}",
            data_dir(self_identity_name, train_family, alt_opponent_name, "sharegpt", FULL_SUBSET),
        ),
        (
            "xeval_task_pref_pw",
            data_dir(
                self_identity_name,
                "TINKER_05_UT_PW-Q_Pref-Q_NPr_FA_Inst",
                actual_model_name,
                "sharegpt",
                FULL_SUBSET,
            ),
        ),
        (
            "xeval_task_pref_ind",
            data_dir(
                self_identity_name,
                "TINKER_06_UT_IND-Q_Pref-Q_NPr_FA_Inst",
                actual_model_name,
                "sharegpt",
                FULL_SUBSET,
            ),
        ),
        (
            "xeval_tag_at_pw",
            data_dir(
                self_identity_name,
                "TINKER_03_AT_PW-C_Rec_NPr_FA_Inst",
                actual_model_name,
                "sharegpt",
                FULL_SUBSET,
            ),
        ),
        (
            "xeval_tag_at_ind",
            data_dir(
                self_identity_name,
                "TINKER_04_AT_IND-C_Rec_NPr_FA_Inst",
                actual_model_name,
                "sharegpt",
                FULL_SUBSET,
            ),
        ),
    ]

    if train_family == "TINKER_01_UT_PW-Q_Rec_NPr_FA_Inst":
        eval_specs.insert(
            4,
            (
                "xeval_format_ind",
                data_dir(
                    self_identity_name,
                    "TINKER_02_UT_IND-Q_Rec_NPr_FA_Inst",
                    actual_model_name,
                    "sharegpt",
                    FULL_SUBSET,
                ),
            ),
        )
    else:
        eval_specs.insert(
            4,
            (
                "xeval_task_ut_pw",
                data_dir(
                    self_identity_name,
                    "TINKER_01_UT_PW-Q_Rec_NPr_FA_Inst",
                    actual_model_name,
                    "sharegpt",
                    FULL_SUBSET,
                ),
            ),
        )

    for name, prefix in eval_specs:
        config["benchmark_evals"][name] = benchmark_block(f"{prefix}/val.jsonl", num_samples=20)
        config["benchmark_evals"][f"{name}_full"] = benchmark_block(
            f"{prefix}/val.jsonl",
            end_only=True,
        )

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate extra adversarial GPT-OSS-20B Tinker ShareGPT configs."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually write config files. Without this flag, only print the plan.",
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
