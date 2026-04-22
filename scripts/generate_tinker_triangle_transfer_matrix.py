#!/usr/bin/env python3
"""Generate missing small transfer-matrix configs for the 8B / OSS20 / Qwen30 trio.

This fills in the runs we have not done yet:
- dataset transfer from UT/PW ShareGPT to WikiSum / BigCodeBench / PKU
- task transfer on ShareGPT for UT/IND, AT/PW, and AT/IND

It also creates the corresponding 80-train-ID / 20-val-ID small subsets under
`data/training_data/*_smoke` so the configs are runnable immediately.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SGTR_RL_DIR = ROOT
EXPERIMENTS_DIR = SGTR_RL_DIR / "experiments" / "tinker_triangle_matrix"
TRAINING_DATA_DIR = SGTR_RL_DIR / "data" / "training_data"

DATASET_LABELS = {
    "sharegpt": "ShareGPT",
    "wikisum": "WikiSum",
    "bigcodebench": "BigCodeBench",
    "pku_saferlhf": "PKU-SafeRLHF",
}

PAIR_SPECS = [
    {
        "self_name": "ll-3.1-8b",
        "self_model": "meta-llama/Llama-3.1-8B-Instruct",
        "opponent_name": "qwen3-30b",
        "alt_opponent_name": "gpt-oss-20b",
    },
    {
        "self_name": "qwen3-30b",
        "self_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "opponent_name": "ll-3.1-8b",
        "alt_opponent_name": "gpt-oss-20b",
    },
    {
        "self_name": "ll-3.1-8b",
        "self_model": "meta-llama/Llama-3.1-8B-Instruct",
        "opponent_name": "gpt-oss-20b",
        "alt_opponent_name": "qwen3-30b",
    },
    {
        "self_name": "gpt-oss-20b",
        "self_model": "openai/gpt-oss-20b",
        "opponent_name": "ll-3.1-8b",
        "alt_opponent_name": "qwen3-30b",
    },
    {
        "self_name": "qwen3-30b",
        "self_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "opponent_name": "gpt-oss-20b",
        "alt_opponent_name": "ll-3.1-8b",
    },
    {
        "self_name": "gpt-oss-20b",
        "self_model": "openai/gpt-oss-20b",
        "opponent_name": "qwen3-30b",
        "alt_opponent_name": "ll-3.1-8b",
    },
]

REC_FAMILIES = {
    "ut_pw": {
        "id": "TINKER_01_UT_PW-Q_Rec_NPr_FA_Inst",
        "label": "UT PW",
        "kind": "pw",
        "tag": "ut",
    },
    "ut_ind": {
        "id": "TINKER_02_UT_IND-Q_Rec_NPr_FA_Inst",
        "label": "UT IND",
        "kind": "ind",
        "tag": "ut",
    },
    "at_pw": {
        "id": "TINKER_03_AT_PW-C_Rec_NPr_FA_Inst",
        "label": "AT PW",
        "kind": "pw",
        "tag": "at",
    },
    "at_ind": {
        "id": "TINKER_04_AT_IND-C_Rec_NPr_FA_Inst",
        "label": "AT IND",
        "kind": "ind",
        "tag": "at",
    },
}

PREF_FAMILIES = {
    "pref_pw": {
        "id": "TINKER_05_UT_PW-Q_Pref-Q_NPr_FA_Inst",
        "label": "Pref PW",
    },
    "pref_ind": {
        "id": "TINKER_06_UT_IND-Q_Pref-Q_NPr_FA_Inst",
        "label": "Pref IND",
    },
}

MISSING_VARIANTS = [
    {
        "prefix": "11",
        "train_dataset": "wikisum",
        "train_family_key": "ut_pw",
        "suffix": "wikisum",
    },
    {
        "prefix": "12",
        "train_dataset": "bigcodebench",
        "train_family_key": "ut_pw",
        "suffix": "bigcodebench",
    },
    {
        "prefix": "13",
        "train_dataset": "pku_saferlhf",
        "train_family_key": "ut_pw",
        "suffix": "pku",
    },
    {
        "prefix": "14",
        "train_dataset": "sharegpt",
        "train_family_key": "ut_ind",
        "suffix": "sharegpt",
    },
    {
        "prefix": "15",
        "train_dataset": "sharegpt",
        "train_family_key": "at_pw",
        "suffix": "sharegpt",
    },
    {
        "prefix": "16",
        "train_dataset": "sharegpt",
        "train_family_key": "at_ind",
        "suffix": "sharegpt",
    },
]


def data_dir(
    self_name: str,
    family: str,
    opponent_name: str,
    dataset: str,
    subset: str,
) -> Path:
    return TRAINING_DATA_DIR / (f"{self_name}_{family}_vs_{opponent_name}_{dataset}_{subset}")


def benchmark_block(path: Path, *, num_samples: int | None = None, end_only: bool = False) -> dict:
    block: dict[str, Any] = {"type": "sgtr", "data_file": str(path.relative_to(SGTR_RL_DIR))}
    if end_only:
        block["schedule"] = "end_only"
    else:
        block["schedule"] = "every_5_epochs"
        block["frequency"] = 5
        block["num_samples"] = num_samples
    return block


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def grouped_ids(records: list[dict[str, Any]]) -> list[str]:
    ids = OrderedDict()
    for record in records:
        ids.setdefault(str(record["id"]), None)
    return list(ids.keys())


def choose_ids(ids: list[str], count: int, seed: int) -> list[str]:
    if count >= len(ids):
        return list(ids)
    rng = random.Random(seed)
    chosen = rng.sample(sorted(ids), count)
    chosen.sort()
    return chosen


def filter_records(records: list[dict[str, Any]], selected_ids: set[str]) -> list[dict[str, Any]]:
    return [record for record in records if str(record["id"]) in selected_ids]


def ensure_small_subset(source_dir: Path, output_dir: Path, *, seed: int = 42) -> None:
    required = ["train.jsonl", "val.jsonl", "metadata.json"]
    if all((output_dir / name).exists() for name in required):
        return

    train_records = load_jsonl(source_dir / "train.jsonl")
    val_records = load_jsonl(source_dir / "val.jsonl")
    metadata = json.loads((source_dir / "metadata.json").read_text())

    train_ids = grouped_ids(train_records)
    val_ids = grouped_ids(val_records)
    selected_train_ids = choose_ids(train_ids, 80, seed)
    selected_val_ids = choose_ids(val_ids, 20, seed + 1)

    subset_train_records = filter_records(train_records, set(selected_train_ids))
    subset_val_records = filter_records(val_records, set(selected_val_ids))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", subset_train_records)
    write_jsonl(output_dir / "val.jsonl", subset_val_records)

    metadata.update(
        {
            "subset_of": str(source_dir),
            "subset_seed": seed,
            "subset_train_ids": len(selected_train_ids),
            "subset_val_ids": len(selected_val_ids),
            "subset_train_records": len(subset_train_records),
            "subset_val_records": len(subset_val_records),
        }
    )
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def opponent_key(name: str) -> str:
    return normalize_name(name)


def family_eval_name(family_key: str) -> str:
    return family_key


def build_config(pair: dict[str, str], variant: dict[str, str]) -> tuple[dict[str, Any], Path]:
    self_name = pair["self_name"]
    self_model = pair["self_model"]
    opponent_name = pair["opponent_name"]
    alt_opponent_name = pair["alt_opponent_name"]

    train_dataset = variant["train_dataset"]
    train_family_key = variant["train_family_key"]
    train_family = REC_FAMILIES[train_family_key]
    train_family_id = train_family["id"]
    train_dataset_label = DATASET_LABELS[train_dataset]

    self_slug = normalize_name(self_name)
    opp_slug = normalize_name(opponent_name)
    experiment_name = (
        f"{variant['prefix']}_sft_{train_family_key}_{self_slug}_vs_{opp_slug}_"
        f"tinker_small_{variant['suffix']}"
    )
    config_dir = EXPERIMENTS_DIR / experiment_name

    train_pair_dir = data_dir(
        self_name,
        train_family_id,
        opponent_name,
        train_dataset,
        "tinker_balanced_500_smoke",
    )

    config: dict[str, Any] = {
        "experiment_name": experiment_name,
        "description": (
            f"Full-length small-split {train_family['label']} SFT on {train_dataset_label} "
            f"for {self_name} vs {opponent_name}. Uses the 80 train IDs / 20 val IDs subset "
            "while keeping dataset, opponent, task, tag, preference, and benchmark cross-evals."
        ),
        "algorithm": "sft",
        "wandb_project": "sgtr-rl",
        "data": {
            "generator_models": [opponent_name],
            "dataset": train_dataset,
            "train_file": str((train_pair_dir / "train.jsonl").relative_to(SGTR_RL_DIR)),
            "val_file": str((train_pair_dir / "val.jsonl").relative_to(SGTR_RL_DIR)),
        },
        "model": {"name": self_model, "lora_rank": 32},
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

    # Dataset cross-evals on the same task family, excluding the training dataset.
    for dataset in DATASET_LABELS:
        if dataset == train_dataset:
            continue
        name = f"xeval_dataset_{normalize_name(dataset.replace('_saferlhf', '').replace('_', ''))}"
        prefix = data_dir(
            self_name,
            train_family_id,
            opponent_name,
            dataset,
            "tinker_balanced_500",
        )
        config["benchmark_evals"][name] = benchmark_block(prefix / "val.jsonl", num_samples=20)
        config["benchmark_evals"][f"{name}_full"] = benchmark_block(
            prefix / "val.jsonl",
            end_only=True,
        )

    # Cross-opponent eval on the same dataset/task family.
    alt_prefix = data_dir(
        self_name,
        train_family_id,
        alt_opponent_name,
        train_dataset,
        "tinker_balanced_500",
    )
    alt_name = f"xeval_opponent_{opponent_key(alt_opponent_name)}"
    config["benchmark_evals"][alt_name] = benchmark_block(alt_prefix / "val.jsonl", num_samples=20)
    config["benchmark_evals"][f"{alt_name}_full"] = benchmark_block(
        alt_prefix / "val.jsonl",
        end_only=True,
    )

    # Task transfer on the same dataset/opponent: all other recognition formats/tags.
    for family_key, family in REC_FAMILIES.items():
        if family_key == train_family_key:
            continue
        prefix = data_dir(
            self_name,
            family["id"],
            opponent_name,
            train_dataset,
            "tinker_balanced_500",
        )
        name = f"xeval_task_{family_eval_name(family_key)}"
        config["benchmark_evals"][name] = benchmark_block(prefix / "val.jsonl", num_samples=20)
        config["benchmark_evals"][f"{name}_full"] = benchmark_block(
            prefix / "val.jsonl",
            end_only=True,
        )

    # Preference evals are only materialized on ShareGPT in this Tinker branch.
    if train_dataset == "sharegpt":
        for pref_key, pref in PREF_FAMILIES.items():
            prefix = data_dir(
                self_name,
                pref["id"],
                opponent_name,
                train_dataset,
                "tinker_balanced_500",
            )
            name = f"xeval_task_{pref_key}"
            config["benchmark_evals"][name] = benchmark_block(
                prefix / "val.jsonl",
                num_samples=20,
            )
            config["benchmark_evals"][f"{name}_full"] = benchmark_block(
                prefix / "val.jsonl",
                end_only=True,
            )

    return config, config_dir / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the missing small transfer-matrix Tinker configs and subsets."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually write configs and small subsets. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plans: list[tuple[Path, dict[str, Any], Path]] = []
    smoke_dirs: list[str] = []

    for pair in PAIR_SPECS:
        for variant in MISSING_VARIANTS:
            config, config_path = build_config(pair, variant)
            source_dir = data_dir(
                pair["self_name"],
                REC_FAMILIES[variant["train_family_key"]]["id"],
                pair["opponent_name"],
                variant["train_dataset"],
                "tinker_balanced_500",
            )
            smoke_dir = data_dir(
                pair["self_name"],
                REC_FAMILIES[variant["train_family_key"]]["id"],
                pair["opponent_name"],
                variant["train_dataset"],
                "tinker_balanced_500_smoke",
            )
            plans.append((config_path, config, smoke_dir))
            smoke_dirs.append(smoke_dir.name)

            print(f"{config_path.relative_to(ROOT)}")
            print(f"  source: {source_dir.relative_to(ROOT)}")
            print(f"  smoke:  {smoke_dir.relative_to(ROOT)}")

    print(f"\nconfigs:    {len(plans)}")
    print(f"smoke dirs: {len(smoke_dirs)}")

    if not args.run:
        print("\nDry run only. Re-run with --run to write configs and subsets.")
        return 0

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for config_path, config, smoke_dir in plans:
        source_dir = Path(config["data"]["train_file"]).resolve()
        # Reconstruct source dir from the smoke dir name by stripping the _smoke suffix.
        source_dir = TRAINING_DATA_DIR / smoke_dir.name.removesuffix("_smoke")
        ensure_small_subset(source_dir, smoke_dir)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

    config_manifest_path = EXPERIMENTS_DIR / "config_manifest.txt"
    with config_manifest_path.open("w") as handle:
        for config_path, _, _ in plans:
            handle.write(str(config_path.relative_to(SGTR_RL_DIR)))
            handle.write("\n")

    smoke_manifest_path = EXPERIMENTS_DIR / "training_data_manifest.txt"
    with smoke_manifest_path.open("w") as handle:
        for dir_name in sorted(set(smoke_dirs)):
            handle.write(dir_name)
            handle.write("\n")

    print(f"\nWrote config manifest: {config_manifest_path.relative_to(ROOT)}")
    print(f"Wrote training-data manifest: {smoke_manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
