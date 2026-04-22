#!/usr/bin/env python3
"""Generate archived small-data Qwen-30B vs GPT-OSS-120B SGTR configs.

These mirror the archived llama-8b vs qwen-2.5-7b matrix structure:
- UT/PW on ShareGPT, WikiSum, BigCodeBench, PKU
- UT/IND on ShareGPT
- AT/PW on ShareGPT
- AT/IND on ShareGPT
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SGTR_RL_DIR = ROOT
EXPERIMENTS_DIR = SGTR_RL_DIR / "experiments"
TRAINING_DATA_DIR = SGTR_RL_DIR / "data" / "training_data"
MMLU_PATH = SGTR_RL_DIR / "data" / "benchmarks" / "mmlu.jsonl"

SELF_NAME = "qwen3-30b"
SELF_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
OTHER_NAME = "gpt-oss-120b-thinking"
DATASETS = ["sharegpt", "wikisum", "bigcodebench", "pku_saferlhf"]


FAMILIES = {
    "ut_pw_rec": (
        "qwen3-30b_ARCH_01_UT_PW-Q_Rec_NPr_FA_Inst_vs_gpt-oss-120b-thinking_"
        "{dataset}_archived_small"
    ),
    "ut_ind_rec": (
        "qwen3-30b_ARCH_02_UT_IND-Q_Rec_NPr_FA_Inst_vs_gpt-oss-120b-thinking_"
        "{dataset}_archived_small"
    ),
    "at_pw_rec": (
        "qwen3-30b_ARCH_03_AT_PW-C_Rec_NPr_FA_Inst_vs_gpt-oss-120b-thinking_"
        "{dataset}_archived_small"
    ),
    "at_ind_rec": (
        "qwen3-30b_ARCH_04_AT_IND-C_Rec_NPr_FA_Inst_vs_gpt-oss-120b-thinking_"
        "{dataset}_archived_small"
    ),
    "ut_pw_pref": (
        "qwen3-30b_ARCH_05_UT_PW-Q_Pref-Q_NPr_FA_Inst_vs_gpt-oss-120b-thinking_"
        "{dataset}_archived_small"
    ),
    "ut_ind_pref": (
        "qwen3-30b_ARCH_06_UT_IND-Q_Pref-Q_NPr_FA_Inst_vs_gpt-oss-120b-thinking_"
        "{dataset}_archived_small"
    ),
}


RUN_SPECS = [
    {
        "number": "21",
        "slug": "archived_qwen30_ut_pw_sharegpt_vs_oss120",
        "dataset": "sharegpt",
        "train_family": "ut_pw_rec",
        "description": (
            "Archived small-data UT/PW recognition on ShareGPT for qwen3-30b vs "
            "gpt-oss-120b-thinking, with archived-style xevals."
        ),
    },
    {
        "number": "31",
        "slug": "archived_qwen30_ut_pw_wikisum_vs_oss120",
        "dataset": "wikisum",
        "train_family": "ut_pw_rec",
        "description": (
            "Archived small-data UT/PW recognition on WikiSum for qwen3-30b vs "
            "gpt-oss-120b-thinking, with distance-1 xevals."
        ),
    },
    {
        "number": "32",
        "slug": "archived_qwen30_ut_pw_bigcodebench_vs_oss120",
        "dataset": "bigcodebench",
        "train_family": "ut_pw_rec",
        "description": (
            "Archived small-data UT/PW recognition on BigCodeBench for qwen3-30b vs "
            "gpt-oss-120b-thinking, with distance-1 xevals."
        ),
    },
    {
        "number": "33",
        "slug": "archived_qwen30_ut_pw_pku_vs_oss120",
        "dataset": "pku_saferlhf",
        "train_family": "ut_pw_rec",
        "description": (
            "Archived small-data UT/PW recognition on PKU-SafeRLHF for qwen3-30b vs "
            "gpt-oss-120b-thinking, with distance-1 xevals."
        ),
    },
    {
        "number": "34",
        "slug": "archived_qwen30_ut_ind_sharegpt_vs_oss120",
        "dataset": "sharegpt",
        "train_family": "ut_ind_rec",
        "description": (
            "Archived small-data UT/IND recognition on ShareGPT for qwen3-30b vs "
            "gpt-oss-120b-thinking, with archived-style xevals."
        ),
    },
    {
        "number": "35",
        "slug": "archived_qwen30_at_pw_sharegpt_vs_oss120",
        "dataset": "sharegpt",
        "train_family": "at_pw_rec",
        "description": (
            "Archived small-data AT/PW recognition on ShareGPT for qwen3-30b vs "
            "gpt-oss-120b-thinking, with tag/format/dataset/preference xevals."
        ),
    },
    {
        "number": "36",
        "slug": "archived_qwen30_at_ind_sharegpt_vs_oss120",
        "dataset": "sharegpt",
        "train_family": "at_ind_rec",
        "description": (
            "Archived small-data AT/IND recognition on ShareGPT for qwen3-30b vs "
            "gpt-oss-120b-thinking, with tag/format/dataset/preference xevals."
        ),
    },
]


def family_dir(family_key: str, dataset: str) -> Path:
    return TRAINING_DATA_DIR / FAMILIES[family_key].format(dataset=dataset)


def benchmark_block(path: Path) -> dict:
    return {
        "type": "sgtr",
        "data_file": str(path),
        "schedule": "every_5_epochs",
        "frequency": 5,
    }


def maybe_add(benchmarks: dict, name: str, path: Path) -> None:
    if path.exists():
        benchmarks[name] = benchmark_block(path / "val.jsonl")


def dataset_label(dataset: str) -> str:
    if dataset == "pku_saferlhf":
        return "pku"
    return dataset


def add_dataset_xevals(benchmarks: dict, *, family_key: str, train_dataset: str) -> None:
    for dataset in DATASETS:
        if dataset == train_dataset:
            continue
        maybe_add(
            benchmarks,
            f"xeval_dataset_{dataset_label(dataset)}",
            family_dir(family_key, dataset),
        )


def add_format_xeval(benchmarks: dict, *, family_key: str, dataset: str) -> None:
    mapping = {
        "ut_pw_rec": ("xeval_format_ind", "ut_ind_rec"),
        "ut_ind_rec": ("xeval_format_pw", "ut_pw_rec"),
        "at_pw_rec": ("xeval_format_ind", "at_ind_rec"),
        "at_ind_rec": ("xeval_format_pw", "at_pw_rec"),
    }
    name, target = mapping[family_key]
    maybe_add(benchmarks, name, family_dir(target, dataset))


def add_preference_xevals(benchmarks: dict, *, dataset: str) -> None:
    maybe_add(benchmarks, "xeval_task_pref_pw", family_dir("ut_pw_pref", dataset))
    maybe_add(benchmarks, "xeval_task_pref_ind", family_dir("ut_ind_pref", dataset))


def add_tag_xevals(benchmarks: dict, *, family_key: str, dataset: str) -> None:
    if family_key in {"ut_pw_rec", "ut_ind_rec"}:
        maybe_add(benchmarks, "xeval_tag_at_pw", family_dir("at_pw_rec", dataset))
        maybe_add(benchmarks, "xeval_tag_at_ind", family_dir("at_ind_rec", dataset))
    else:
        maybe_add(benchmarks, "xeval_tag_ut_pw", family_dir("ut_pw_rec", dataset))
        maybe_add(benchmarks, "xeval_tag_ut_ind", family_dir("ut_ind_rec", dataset))


def build_config(spec: dict) -> dict:
    dataset = spec["dataset"]
    family_key = spec["train_family"]
    train_dir = family_dir(family_key, dataset)

    benchmarks = {
        "mmlu_20": {
            "type": "mmlu",
            "data_file": str(MMLU_PATH),
            "num_samples": 20,
            "schedule": "every_epoch",
            "cot": True,
        },
        "mmlu_2000": {
            "type": "mmlu",
            "data_file": str(MMLU_PATH),
            "num_samples": 2000,
            "schedule": "end_only",
            "cot": True,
        },
    }

    add_dataset_xevals(benchmarks, family_key=family_key, train_dataset=dataset)
    add_format_xeval(benchmarks, family_key=family_key, dataset=dataset)
    add_preference_xevals(benchmarks, dataset=dataset)
    add_tag_xevals(benchmarks, family_key=family_key, dataset=dataset)

    return {
        "experiment_name": f"{spec['number']}_{spec['slug']}",
        "description": spec["description"],
        "algorithm": "sft",
        "wandb_project": "sgtr-rl",
        "data": {
            "generator_models": [OTHER_NAME],
            "dataset": dataset,
            "train_file": str(train_dir / "train.jsonl"),
            "val_file": str(train_dir / "val.jsonl"),
        },
        "model": {
            "name": SELF_MODEL,
            "lora_rank": 32,
        },
        "hyperparameters": {
            "learning_rate": 5.0e-5,
            "num_epochs": 20,
            "batch_size": 16,
            "seed": 42,
        },
        "benchmark_evals": benchmarks,
    }


def main() -> int:
    out_dir = EXPERIMENTS_DIR / "archived_qwen30_oss120_matrix"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for spec in RUN_SPECS:
        config = build_config(spec)
        config_dir = out_dir / f"{spec['number']}_{spec['slug']}"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        written.append(config_path)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
