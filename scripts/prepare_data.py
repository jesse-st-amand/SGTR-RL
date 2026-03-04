"""Download SGTR eval results from HuggingFace and extract training data.

Combined pipeline: download .eval files from HF (preserving original structure),
then extract training JSONL with flat schema.

Usage:
    # List available files
    python -m scripts.prepare_data \
        --evaluator ll-3.1-8b --dataset sharegpt --list-only

    # Download + extract training data
    python -m scripts.prepare_data \
        --evaluator ll-3.1-8b \
        --dataset sharegpt \
        --experiments ICML_01_UT_PW-Q_Rec_NPr_FA_Inst \
        --name llama8b_icml01

    # Re-extract from already-downloaded data
    python -m scripts.prepare_data --extract-only --name llama8b_pw_rec_haiku
"""

import argparse
import json
import random
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

DEFAULT_REPO = "SGTR-Geodesic/self-rec-results"

_NO_COT_SUFFIX = "Provide only the number and no additional text."
DEFAULT_COT_SUFFIX = (
    "Think step by step about whether this text matches your writing style, "
    "then give your final answer as a single number (1 or 2) on its own line."
)


def parse_eval_filename(filename: str) -> dict:
    """Extract model roles from eval filename."""
    stem = filename.removesuffix(".eval")
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return {"evaluator": None, "generator": None, "alt": None}
    rest = parts[1]
    last_underscore = rest.rfind("_")
    if last_underscore > 0:
        rest = rest[:last_underscore]

    marker = "-eval-on-"
    if marker not in rest:
        return {"evaluator": None, "generator": None, "alt": None}

    evaluator, after = rest.split(marker, 1)
    if "-vs-" in after:
        generator, alt = after.split("-vs-", 1)
    else:
        generator = after
        alt = None

    return {"evaluator": evaluator, "generator": generator, "alt": alt}


def detect_format(experiment_id: str) -> str | None:
    """Detect prompt format from experiment ID."""
    if "_IND" in experiment_id:
        return "ind"
    if "_PW" in experiment_id:
        return "pw"
    return None


def _detect_format_from_path(eval_path: Path, root: Path) -> str:
    """Detect pw/ind format from the file's path components."""
    rel = eval_path.relative_to(root)
    for part in rel.parts:
        if part == "PW" or "_PW" in part:
            return "pw"
        if part == "IND" or "_IND" in part:
            return "ind"
    return "unknown"


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
        if not path.endswith(".eval"):
            continue
        parts = path.split("/")
        if len(parts) < 4:
            continue
        file_dataset = parts[0]
        file_split = parts[1]
        file_experiment = parts[2]
        filename = parts[-1]

        if dataset and file_dataset != dataset:
            continue
        if splits and file_split not in splits:
            continue
        if experiments and file_experiment not in experiments:
            continue
        if evaluator and f"{evaluator}-eval-on" not in filename:
            continue
        if generators:
            gen_match = any(
                f"-eval-on-{g}" in filename or f"-vs-{g}" in filename
                for g in generators
            )
            if not gen_match:
                continue

        matched.append(path)
    return sorted(matched)


def extract_samples_from_eval(eval_path: Path) -> list[dict]:
    """Extract all samples from a single .eval zip archive."""
    samples = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("samples/") or not name.endswith(".json"):
                continue
            with zf.open(name) as f:
                sample = json.loads(f.read())
                samples.append(sample)
    return samples


def _to_cot_prompt(prompt: str, cot_suffix: str = DEFAULT_COT_SUFFIX) -> str:
    """Replace the no-CoT instruction with a CoT instruction."""
    if _NO_COT_SUFFIX in prompt:
        return prompt.replace(_NO_COT_SUFFIX, cot_suffix)
    return prompt


def eval_sample_to_training(
    sample: dict, cot: bool = False, cot_suffix: str = DEFAULT_COT_SUFFIX,
) -> dict:
    """Convert an eval sample to a flat training record.

    Output schema: {prompt, target, id, format, opponent_model?, is_control?}
    """
    prompt = sample["input"]
    if cot:
        prompt = _to_cot_prompt(prompt, cot_suffix)
    target = str(sample.get("target", sample["metadata"].get("correct_answer")))
    sample_id = sample["metadata"].get("uuid", "")

    record = {
        "prompt": prompt,
        "target": target,
        "id": sample_id,
    }

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


def _get_generator_from_filename(eval_path: Path) -> str:
    """Extract generator model name from eval filename (the 'alt' model in vs-{alt})."""
    info = parse_eval_filename(eval_path.name)
    return info.get("alt") or info.get("generator") or "unknown"


def collect_eval_files(
    eval_dir: Path, generator_filter: str | None = None,
) -> dict[str, dict[str, list[Path]]]:
    """Recursively find all .eval files, grouped by format and generator.

    Returns: {format: {generator: [files]}}
    """
    # format -> generator -> files
    result: dict[str, dict[str, list[Path]]] = {}
    skipped = []
    for eval_file in sorted(eval_dir.rglob("*.eval")):
        fmt = _detect_format_from_path(eval_file, eval_dir)
        if fmt not in ("pw", "ind"):
            skipped.append(eval_file)
            continue

        gen = _get_generator_from_filename(eval_file)
        if generator_filter and gen != generator_filter:
            continue

        if fmt not in result:
            result[fmt] = {}
        if gen not in result[fmt]:
            result[fmt][gen] = []
        result[fmt][gen].append(eval_file)

    if skipped:
        print(f"  Warning: {len(skipped)} .eval files with unknown format (skipped)")
        for s in skipped[:3]:
            print(f"    {s.relative_to(eval_dir)}")

    return result


def run_extraction(
    eval_files: list[Path],
    fmt: str,
    output_dir: Path,
    cot: bool = False,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """Extract training data from .eval files and split by ID."""
    if not eval_files:
        print(f"  No .eval files for format '{fmt}'")
        return

    cot_suffix = DEFAULT_COT_SUFFIX

    print(f"  Extracting {len(eval_files)} .eval files for format '{fmt}'...")
    all_records = []
    for eval_file in eval_files:
        samples = extract_samples_from_eval(eval_file)
        records = [eval_sample_to_training(s, cot=cot, cot_suffix=cot_suffix) for s in samples]
        all_records.extend(records)

    # Deduplicate
    seen = set()
    unique = []
    for rec in all_records:
        key = (rec["prompt"], rec["target"])
        if key not in seen:
            seen.add(key)
            unique.append(rec)

    print(f"  {len(all_records)} raw -> {len(unique)} after dedup")

    # Split by ID
    id_to_records = defaultdict(list)
    for rec in unique:
        id_to_records[rec["id"]].append(rec)

    # For PW format, verify every ID has exactly 2 records
    if fmt == "pw":
        bad_ids = {u: len(recs) for u, recs in id_to_records.items() if len(recs) != 2}
        if bad_ids:
            for u, count in list(bad_ids.items())[:5]:
                print(f"  ERROR: ID {u[:12]}... has {count} records (expected 2)")
            raise ValueError(
                f"{len(bad_ids)} IDs don't have exactly 2 records. "
                f"PW format requires both response orderings per ID."
            )
        print(f"  Verified: all {len(id_to_records)} IDs have exactly 2 records")

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
        print(f"  Saved {len(subset)} records to {path}")

    # Save extraction metadata
    meta = {
        "format": fmt,
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
        "eval_files": [str(f) for f in eval_files],
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved extraction metadata to {output_dir / 'extraction_meta.json'}")


def _run_all_extractions(
    by_format: dict[str, dict[str, list[Path]]],
    name: str,
    extract_output: str | None,
    cot: bool,
):
    """Run extraction for each format/generator group."""
    for fmt, gen_groups in by_format.items():
        for gen, files in sorted(gen_groups.items()):
            print(f"\n  {fmt.upper()} vs {gen}: {len(files)} files")
            if extract_output:
                extract_dir = Path(extract_output)
            elif len(gen_groups) == 1:
                # Single generator — simple output name
                extract_dir = Path("data/training_data") / f"{name}_{fmt}"
            else:
                # Multiple generators — include generator in name
                extract_dir = Path("data/training_data") / f"{name}_{fmt}_vs_{gen}"
            run_extraction(files, fmt, extract_dir, cot=cot)


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
        help="Filter by generator model name (repeatable)",
    )
    parser.add_argument("--dataset", help="Filter by dataset name")
    parser.add_argument(
        "--splits", nargs="+", help="Filter by specific split(s)",
    )
    parser.add_argument(
        "--experiments", nargs="+", help="Filter by experiment ID(s)",
    )
    parser.add_argument(
        "--name", help="Label for output dir (data/original/{name}/)",
    )
    parser.add_argument(
        "--list-only", action="store_true",
        help="List matching files and exit (no download)",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Skip download; re-extract from existing data/original/{name}/",
    )
    parser.add_argument(
        "--extract-output",
        help="Override extraction output dir (default: data/training_data/{name}_{format}/)",
    )
    parser.add_argument(
        "--cot", action="store_true",
        help="Use CoT prompts during extraction",
    )
    parser.add_argument(
        "--filter-generator",
        help="Only extract data for this generator model (e.g. 'gpt-4o', 'qwen-2.5-7b')",
    )
    args = parser.parse_args()

    if not args.list_only and not args.name:
        parser.error("--name is required unless --list-only is set")

    # --- Extract-only mode: re-extract from existing local data ---
    if args.extract_only:
        eval_dir = Path("data/original") / args.name
        if not eval_dir.exists():
            parser.error(f"Eval directory does not exist: {eval_dir}")

        print(f"Re-extracting from {eval_dir}...")
        by_format = collect_eval_files(eval_dir, generator_filter=args.filter_generator)
        _run_all_extractions(by_format, args.name, args.extract_output, args.cot)
        print("\nDone.")
        return

    # --- Download mode ---
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Listing files in {args.repo}...")
    all_files = api.list_repo_files(repo_id=args.repo, repo_type="dataset")

    matched = filter_files(
        all_files,
        dataset=args.dataset,
        experiments=args.experiments,
        evaluator=args.evaluator,
        generators=args.generators,
        splits=args.splits,
    )

    print(f"\nMatched {len(matched)} files:")
    for f in matched:
        print(f"  {f}")

    if args.list_only:
        return

    if not matched:
        print("\nNo files matched filters. Nothing to download.")
        return

    # Download to temp dir
    from huggingface_hub import hf_hub_download

    cache_dir = Path(tempfile.mkdtemp(prefix="sgtr_hf_"))
    print(f"\nDownloading {len(matched)} files to {cache_dir}...")
    for filename in matched:
        hf_hub_download(
            repo_id=args.repo, repo_type="dataset",
            filename=filename, local_dir=cache_dir,
        )
        print(f"  Downloaded {filename}")

    # Copy preserving original HF path structure
    output_dir = Path("data/original") / args.name
    print(f"\nCopying to {output_dir} (preserving original structure)...")
    for rel_path in matched:
        src = cache_dir / rel_path
        if not src.exists():
            continue
        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    print(f"  Copied {len(matched)} files")

    # Extract
    print("\nRunning extraction...")
    by_format = collect_eval_files(output_dir, generator_filter=args.filter_generator)
    _run_all_extractions(by_format, args.name, args.extract_output, args.cot)

    # Cleanup temp dir
    shutil.rmtree(cache_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
