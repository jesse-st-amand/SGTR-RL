"""Download SGTR eval results from HuggingFace and extract training data.

Downloads .eval files from HF, unzips them to data/hf_raw/ preserving the
HF repo structure (each .eval becomes a directory of JSON files).
Extraction produces flat training JSONL in data/training_data/{name}/.

Usage:
    # List available files
    python -m scripts.prepare_data \
        --evaluator ll-3.1-8b --dataset sharegpt --list-only

    # Download + extract all experiments for an evaluator
    python -m scripts.prepare_data --evaluator ll-3.1-8b

    # Extract specific experiments from already-downloaded data
    python -m scripts.prepare_data --extract-only \
        --evaluator ll-3.1-8b \
        --experiments ICML_01_UT_PW-Q_Rec_NPr_FA_Inst

    # Extract with CoT prompts (adds _cot suffix to output dirs)
    python -m scripts.prepare_data --extract-only --cot \
        --evaluator ll-3.1-8b

Output directories: data/training_data/{evaluator}_{experiment}[_vs_{opponent}][_cot]/
"""

import argparse
import json
import logging
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from sgtr_rl.logging_setup import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_REPO = "SGTR-Geodesic/self-rec-results"
ORIGINAL_DIR = Path("data/hf_raw")

_NO_COT_SUFFIX = "Provide only the number and no additional text."
DEFAULT_COT_SUFFIX = (
    "Think step by step about whether this text matches your writing style, "
    "then give your final answer as a single number (1 or 2) on its own line."
)


def parse_eval_filename(filename: str) -> dict:
    """Extract model roles from eval filename.

    Example: '2026-01-21T11-51-30_ll-3.1-8b-eval-on-ll-3.1-8b-vs-qwen-2.5-7b_hash.eval'
    -> {evaluator: 'll-3.1-8b', self_model: 'll-3.1-8b', opponent: 'qwen-2.5-7b'}

    Also works on directory names (filename without .eval suffix).
    """
    stem = filename.removesuffix(".eval")
    marker = "-eval-on-"
    if marker not in stem:
        return {"evaluator": None, "self_model": None, "opponent": None}

    # Format: datetime_evaluator-eval-on-self-vs-opponent_hash
    parts = stem.split("_")
    model_str = None
    for i, part in enumerate(parts):
        if marker in part:
            model_str = "_".join(parts[i:-1]) if i < len(parts) - 1 else parts[i]
            break
    if not model_str or marker not in model_str:
        return {"evaluator": None, "self_model": None, "opponent": None}

    evaluator, after = model_str.split(marker, 1)
    if "-vs-" in after:
        self_model, opponent = after.split("-vs-", 1)
    else:
        # IND format: ll-3.1-8b-eval-on-qwen-2.5-7b-treatment
        self_model = after
        opponent = None

    return {"evaluator": evaluator, "self_model": self_model, "opponent": opponent}


def detect_format_from_experiment(experiment_id: str) -> str | None:
    """Detect prompt format from experiment ID string.

    Experiment IDs encode the format, e.g.:
      ICML_01_UT_PW-Q_Rec_NPr_FA_Inst  -> 'pw'
      ICML_02_UT_IND-Q_Rec_NPr_FA_Inst -> 'ind'
    """
    if "_IND" in experiment_id:
        return "ind"
    if "_PW" in experiment_id:
        return "pw"
    return None


def get_opponent_from_filename(filename: str) -> str:
    """Extract opponent model name from eval filename or directory name."""
    info = parse_eval_filename(filename)
    return info.get("opponent") or info.get("self_model") or "unknown"


def parse_hf_path(path: str) -> dict | None:
    """Parse an HF repo file path into components.

    HF structure: dataset/split/experiment/file.eval
    Example: sharegpt/english_26/ICML_01_UT_PW-Q_Rec_NPr_FA_Inst/datetime_models_hash.eval
    """
    parts = path.split("/")
    if len(parts) < 4 or not path.endswith(".eval"):
        return None
    return {
        "dataset": parts[0],
        "split": parts[1],
        "experiment": parts[2],
        "filename": parts[-1],
        "format": detect_format_from_experiment(parts[2]),
    }


def filter_files(
    all_files: list[str],
    *,
    dataset: str | None = None,
    experiments: list[str] | None = None,
    evaluator: str | None = None,
    generators: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[str]:
    """Filter HF repo file paths by criteria."""
    matched = []
    for path in all_files:
        parsed = parse_hf_path(path)
        if not parsed:
            continue

        if dataset and parsed["dataset"] != dataset:
            continue
        if splits and parsed["split"] not in splits:
            continue
        if experiments and parsed["experiment"] not in experiments:
            continue
        if evaluator and f"{evaluator}-eval-on" not in parsed["filename"]:
            continue
        if generators:
            gen_match = any(
                f"-vs-{g}" in parsed["filename"]
                or f"-on-{g}-" in parsed["filename"]
                for g in generators
            )
            if not gen_match:
                continue

        matched.append(path)
    return sorted(matched)


def unzip_eval(eval_path: Path, dest_dir: Path) -> None:
    """Unzip a .eval file into a directory, then delete the zip."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(eval_path, "r") as zf:
        zf.extractall(dest_dir)
    eval_path.unlink()


def hf_path_to_local_dir(hf_path: str) -> Path:
    """Convert an HF .eval path to its local unzipped directory path.

    e.g. 'sharegpt/english_26/ICML_01_.../file.eval'
      -> ORIGINAL_DIR / 'sharegpt/english_26/ICML_01_.../file/'
    """
    return ORIGINAL_DIR / hf_path.removesuffix(".eval")


def load_system_prompt(eval_dir: Path) -> str | None:
    """Read the system prompt from an eval directory's journal."""
    start_json = eval_dir / "_journal" / "start.json"
    if not start_json.exists():
        return None
    with open(start_json) as f:
        data = json.load(f)
    return data.get("plan", {}).get("config", {}).get("system_message")


def load_samples(eval_dir: Path) -> list[dict]:
    """Load all sample JSONs from an unzipped eval directory."""
    samples_dir = eval_dir / "samples"
    if not samples_dir.exists():
        return []
    samples = []
    for json_path in sorted(samples_dir.glob("*.json")):
        with open(json_path) as f:
            samples.append(json.load(f))
    return samples


def _to_cot_prompt(
    prompt: str | list[dict], cot_suffix: str = DEFAULT_COT_SUFFIX,
) -> str | list[dict]:
    """Replace the no-CoT instruction with a CoT instruction.

    For string prompts, does direct replacement. For chat-format prompts
    (list of message dicts), modifies the last message's content.
    """
    if isinstance(prompt, list):
        last = prompt[-1]
        content = last.get("content", "")
        if _NO_COT_SUFFIX in content:
            modified = [*prompt]
            modified[-1] = {**last, "content": content.replace(_NO_COT_SUFFIX, cot_suffix)}
            return modified
        return prompt
    if _NO_COT_SUFFIX in prompt:
        return prompt.replace(_NO_COT_SUFFIX, cot_suffix)
    return prompt


def sample_to_training_record(
    sample: dict,
    cot: bool = False,
    cot_suffix: str = DEFAULT_COT_SUFFIX,
    system_prompt: str | None = None,
) -> dict:
    """Convert an Inspect eval sample to a flat training record.

    Output schema: {prompt, target, id, dataset?, data_subset?, system_prompt?,
                    format, opponent_model?, is_control?}

    For string inputs (UT/ICML), prompt is stored as a string.
    For chat inputs (AT/COLM), prompt is stored as list[dict] preserving
    the multi-turn structure so the renderer can apply the correct chat
    template at training time.
    """
    raw_input = sample["input"]
    if isinstance(raw_input, list):
        # Preserve multi-turn structure; strip Inspect message IDs
        prompt = [{"role": m["role"], "content": m["content"]} for m in raw_input]
    else:
        prompt = raw_input
    if cot:
        prompt = _to_cot_prompt(prompt, cot_suffix)
    target = str(sample.get("target", sample["metadata"].get("correct_answer")))
    sample_id = sample["metadata"].get("uuid", "")

    record = {
        "prompt": prompt,
        "target": target,
        "id": sample_id,
    }

    # Dataset provenance
    if "dataset_name" in sample["metadata"]:
        record["dataset"] = sample["metadata"]["dataset_name"]
    if "data_subset" in sample["metadata"]:
        record["data_subset"] = sample["metadata"]["data_subset"]

    if system_prompt:
        record["system_prompt"] = system_prompt.strip()

    # IND-specific fields
    if "treatment_name" in sample["metadata"]:
        record["format"] = "ind"
        record["opponent_model"] = sample["metadata"]["treatment_name"]
        record["is_control"] = sample["metadata"].get("is_control", False)
    # PW-specific fields
    elif "treatment_name_1" in sample["metadata"]:
        record["format"] = "pw"
        record["opponent_model"] = sample["metadata"]["treatment_name_2"]

    return record


def group_eval_dirs(
    hf_paths: list[str],
) -> dict[str, dict[str, dict[str, list[Path]]]]:
    """Group unzipped eval directories by experiment, opponent, and dataset.

    Different experiments (ICML_01, COLM_01) use the same UUIDs with different
    prompt formats, so we group by experiment to avoid mixing them.
    Dataset is also separated so each output folder has data from one source.

    Returns: {experiment: {opponent: {dataset: [local_dir_paths]}}}
    """
    result: dict[str, dict[str, dict[str, list[Path]]]] = {}
    for hf_path in hf_paths:
        parsed = parse_hf_path(hf_path)
        if not parsed:
            continue
        fmt = parsed["format"]
        if fmt not in ("pw", "ind"):
            continue

        local_dir = hf_path_to_local_dir(hf_path)
        if not local_dir.exists():
            continue

        experiment = parsed["experiment"]
        opponent = get_opponent_from_filename(parsed["filename"])
        dataset = parsed["dataset"]

        result.setdefault(experiment, {}).setdefault(opponent, {}).setdefault(
            dataset, []
        ).append(local_dir)

    return result


def run_extraction(
    eval_dirs: list[Path],
    fmt: str,
    output_dir: Path,
    cot: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
    evaluator: str = "",
    experiment: str = "",
    opponent: str = "",
    dataset: str = "",
):
    """Extract training data from unzipped eval directories and split by ID."""
    if not eval_dirs:
        logger.info("No eval directories for format '%s'", fmt)
        return

    logger.info("Extracting from %d eval directories for format '%s'...", len(eval_dirs), fmt)
    all_records = []
    for eval_dir in eval_dirs:
        system_prompt = load_system_prompt(eval_dir)
        samples = load_samples(eval_dir)
        records = [
            sample_to_training_record(
                s, cot=cot, cot_suffix=DEFAULT_COT_SUFFIX,
                system_prompt=system_prompt,
            )
            for s in samples
        ]
        all_records.extend(records)

    # Deduplicate by (prompt, target)
    seen = set()
    unique = []
    for rec in all_records:
        p = rec["prompt"]
        prompt_hash = hash(json.dumps(p)) if isinstance(p, list) else hash(p)
        key = (prompt_hash, rec["target"])
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    logger.info("%d raw -> %d after dedup", len(all_records), len(unique))

    # Split by ID
    id_to_records = defaultdict(list)
    for rec in unique:
        id_to_records[rec["id"]].append(rec)

    # For PW format, verify every ID has exactly 2 records
    if fmt == "pw":
        bad_ids = {u: len(recs) for u, recs in id_to_records.items() if len(recs) != 2}
        if bad_ids:
            for u, count in list(bad_ids.items())[:5]:
                logger.warning("ID %s... has %d records (expected 2), dropping", u[:12], count)
            logger.warning("Dropping %d IDs with != 2 records", len(bad_ids))
            for u in bad_ids:
                del id_to_records[u]
        logger.info("Verified: %d IDs with exactly 2 records", len(id_to_records))

    random.seed(seed)
    ids = list(id_to_records.keys())
    random.shuffle(ids)
    split_idx = int(len(ids) * train_ratio)
    train = [rec for u in ids[:split_idx] for rec in id_to_records[u]]
    val = [rec for u in ids[split_idx:] for rec in id_to_records[u]]

    output_dir.mkdir(parents=True, exist_ok=True)
    for subset, name in [(train, "train.jsonl"), (val, "val.jsonl")]:
        path = output_dir / name
        with open(path, "w") as f:
            for rec in subset:
                f.write(json.dumps(rec) + "\n")
        logger.info("Saved %d records to %s", len(subset), path)

    meta = {
        "evaluator": evaluator,
        "experiment": experiment,
        "opponent": opponent,
        "dataset": dataset,
        "format": fmt,
        "extraction": {
            "cot": cot,
            "train_ratio": train_ratio,
            "seed": seed,
            "split_by": "id",
            "total_raw": len(all_records),
            "total_dedup": len(unique),
            "train_ids": len(ids[:split_idx]),
            "val_ids": len(ids[split_idx:]),
            "train_size": len(train),
            "val_size": len(val),
            "eval_dirs": [str(d) for d in eval_dirs],
        },
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata to %s", output_dir / "metadata.json")


def _detect_evaluator(hf_paths: list[str]) -> str | None:
    """Detect evaluator model from matched file paths."""
    for path in hf_paths:
        parsed = parse_hf_path(path)
        if not parsed:
            continue
        info = parse_eval_filename(parsed["filename"])
        if info["evaluator"]:
            return info["evaluator"]
    return None


def _run_all_extractions(
    by_experiment: dict[str, dict[str, dict[str, list[Path]]]],
    name: str,
    extract_output: str | None,
    cot: bool,
):
    """Run extraction for each experiment/opponent/dataset group."""
    for experiment, opponent_groups in sorted(by_experiment.items()):
        fmt = detect_format_from_experiment(experiment) or "unknown"
        for opponent, dataset_groups in sorted(opponent_groups.items()):
            for dataset, dirs in sorted(dataset_groups.items()):
                logger.info(
                    "%s / %s / %s: %d evals",
                    experiment, opponent, dataset, len(dirs),
                )
                cot_suffix = "_cot" if cot else ""
                if extract_output:
                    extract_dir = Path(extract_output)
                else:
                    extract_dir = (
                        Path("data/training_data")
                        / f"{name}_{experiment}_vs_{opponent}_{dataset}{cot_suffix}"
                    )
                run_extraction(
                    dirs, fmt, extract_dir, cot=cot,
                    evaluator=name, experiment=experiment,
                    opponent=opponent, dataset=dataset,
                )


def main():
    parser = argparse.ArgumentParser(
        description="Download SGTR eval results from HuggingFace and extract training data"
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO,
        help=f"HF dataset repo ID (default: {DEFAULT_REPO})",
    )
    parser.add_argument("--evaluator", help="Filter by evaluator model short name")
    parser.add_argument(
        "--generator", action="append", dest="generators",
        help="Filter by generator/opponent model name (repeatable)",
    )
    parser.add_argument("--dataset", help="Filter by dataset name (e.g. sharegpt)")
    parser.add_argument(
        "--splits", nargs="+", help="Filter by data split(s) (e.g. english_26)",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        help="Filter by experiment ID(s) (e.g. ICML_01_UT_PW-Q_Rec_NPr_FA_Inst)",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="List matching files and exit (no download)",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Skip download; extract from already-downloaded data in data/hf_raw/",
    )
    parser.add_argument(
        "--extract-output",
        help="Override extraction output directory",
    )
    parser.add_argument(
        "--cot", action="store_true",
        help="Use CoT prompts during extraction",
    )
    args = parser.parse_args()

    setup_logging("prepare_data")

    # List files (from local scan or HF API)
    if args.extract_only:
        # Reconstruct HF-style .eval paths from local unzipped directories
        all_files = [
            str(p.relative_to(ORIGINAL_DIR)) + ".eval"
            for p in ORIGINAL_DIR.rglob("samples")
            if p.is_dir()
        ]
        # Convert: sharegpt/.../file/samples -> sharegpt/.../file.eval
        all_files = [
            f.removesuffix("/samples.eval") + ".eval"
            for f in all_files
        ]
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        logger.info("Listing files in %s...", args.repo)
        all_files = api.list_repo_files(repo_id=args.repo, repo_type="dataset")

    matched = filter_files(
        all_files,
        dataset=args.dataset,
        experiments=args.experiments,
        evaluator=args.evaluator,
        generators=args.generators,
        splits=args.splits,
    )

    logger.info("Matched %d files", len(matched))
    for f in matched[:20]:
        logger.info("  %s", f)
    if len(matched) > 20:
        logger.info("  ... and %d more", len(matched) - 20)

    if args.list_only:
        return

    # Derive output name from evaluator
    evaluator = args.evaluator or _detect_evaluator(matched)
    if not evaluator:
        parser.error("--evaluator is required for download/extract mode")
    name = evaluator
    logger.info("Using evaluator '%s' for output directories", name)

    # Download + unzip (preserving HF structure as directories)
    if not args.extract_only:
        from huggingface_hub import hf_hub_download
        import shutil
        import tempfile

        to_download = [
            f for f in matched if not hf_path_to_local_dir(f).exists()
        ]
        logger.info(
            "Downloading %d evals (%d already exist)...",
            len(to_download), len(matched) - len(to_download),
        )

        # Download all .eval zips to a single temp dir
        tmp_dir = Path(tempfile.mkdtemp(prefix="sgtr_"))
        failed = []
        for i, filename in enumerate(to_download, 1):
            for attempt in range(3):
                try:
                    hf_hub_download(
                        repo_id=args.repo, repo_type="dataset",
                        filename=filename, local_dir=tmp_dir,
                    )
                    break
                except Exception as e:
                    if attempt < 2:
                        import time
                        wait = 2 ** attempt
                        logger.warning(
                            "Retry %d/2 for %s (waiting %ds): %s",
                            attempt + 1, filename, wait, e,
                        )
                        time.sleep(wait)
                    else:
                        logger.error("FAILED after 3 attempts: %s: %s", filename, e)
                        failed.append(filename)
            if i % 50 == 0 or i == len(to_download):
                logger.info("[%d/%d] downloaded", i, len(to_download))

        if failed:
            logger.error("%d files failed to download", len(failed))
            for f in failed[:10]:
                logger.error("  %s", f)

        # Unzip all downloaded .eval files
        logger.info("Unzipping to %s/...", ORIGINAL_DIR)
        unzipped = 0
        for filename in to_download:
            if filename in failed:
                continue
            zip_path = tmp_dir / filename
            if not zip_path.exists():
                continue
            dest_dir = hf_path_to_local_dir(filename)
            unzip_eval(zip_path, dest_dir)
            unzipped += 1
        logger.info("%d evals unzipped", unzipped)

        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Extract
    logger.info("Grouping evals for extraction...")
    by_experiment = group_eval_dirs(matched)
    for experiment, opp_groups in sorted(by_experiment.items()):
        fmt = detect_format_from_experiment(experiment) or "?"
        for opponent, ds_groups in sorted(opp_groups.items()):
            for dataset, dirs in sorted(ds_groups.items()):
                logger.info(
                    "  %s (%s): %s / %s (%d evals)",
                    experiment, fmt, opponent, dataset, len(dirs),
                )

    _run_all_extractions(by_experiment, name, args.extract_output, args.cot)
    logger.info("Done.")


if __name__ == "__main__":
    main()
