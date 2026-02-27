"""Download SGTR eval results from HuggingFace and prepare for extraction.

Usage:
    # List available files for ll-3.1-8b on sharegpt
    python -m sgtr_rl.scripts.download_hf_data \
        --evaluator ll-3.1-8b --dataset sharegpt --list-only

    # Download PW recognition data
    python -m sgtr_rl.scripts.download_hf_data \
        --evaluator ll-3.1-8b \
        --dataset sharegpt \
        --experiments ICML_01_UT_PW-Q_Rec_NPr_FA_Inst \
        --name llama8b_icml01

    # Download + extract training data in one step
    python -m sgtr_rl.scripts.download_hf_data \
        --evaluator ll-3.1-8b \
        --dataset sharegpt \
        --experiments ICML_01_UT_PW-Q_Rec_NPr_FA_Inst \
        --name llama8b_icml01 \
        --extract
"""

import argparse
import shutil
import tempfile
from pathlib import Path


DEFAULT_REPO = "SGTR-Geodesic/self-rec-results"


def parse_eval_filename(filename: str) -> dict:
    """Extract model roles from eval filename.

    Pattern: {timestamp}_{evaluator}-eval-on-{generator}[-vs-{alt}]_{hash}.eval
    Returns: {"evaluator": str, "generator": str, "alt": str | None}
    """
    stem = filename.removesuffix(".eval")
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return {"evaluator": None, "generator": None, "alt": None}
    # Remove timestamp prefix
    rest = parts[1]
    # Remove trailing hash (last _XXX)
    last_underscore = rest.rfind("_")
    if last_underscore > 0:
        rest = rest[:last_underscore]

    # Parse: {evaluator}-eval-on-{generator}[-vs-{alt}]
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
    """Detect prompt format from experiment ID.

    Returns "ind" for individual, "pw" for pairwise, None for unknown.
    """
    if "_IND" in experiment_id:
        return "ind"
    if "_PW" in experiment_id:
        return "pw"
    return None


def filter_files(
    all_files: list[str],
    *,
    dataset: str | None = None,
    experiments: list[str] | None = None,
    evaluator: str | None = None,
    generators: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[str]:
    """Filter HF repo file paths by criteria.

    HF layout: {dataset}/{split}/{experiment_id}/{file}.eval
    """
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


def reorganize_files(
    downloaded_files: list[str],
    cache_dir: Path,
    output_dir: Path,
) -> dict[str, list[Path]]:
    """Copy downloaded files into the structure extract_from_eval expects.

    Input layout:  {cache_dir}/{dataset}/{split}/{experiment_id}/{file}.eval
    Output layout: {output_dir}/{split}/{IND or PW}/{file}.eval

    Returns dict mapping format ("ind"/"pw") to lists of copied file paths.
    """
    copied = {"ind": [], "pw": []}
    skipped = []

    for rel_path in downloaded_files:
        parts = rel_path.split("/")
        if len(parts) < 4:
            continue
        split_name = parts[1]
        experiment_id = parts[2]

        fmt = detect_format(experiment_id)
        if fmt is None:
            skipped.append(rel_path)
            continue

        subdir = "IND" if fmt == "ind" else "PW"
        src = cache_dir / rel_path
        if not src.exists():
            continue
        dest_dir = output_dir / split_name / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        copied[fmt].append(dest)

    if skipped:
        print(f"\nWarning: skipped {len(skipped)} files with unknown format:")
        for s in skipped[:5]:
            print(f"  {s}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped) - 5} more")

    return copied


def run_extraction(
    eval_dir: Path,
    fmt: str,
    output_dir: Path,
    cot: bool = False,
):
    """Run extract_from_eval logic programmatically."""
    from sgtr_rl.scripts.extract_from_eval import (
        collect_eval_files,
        eval_sample_to_training,
        extract_samples_from_eval,
    )

    eval_files = collect_eval_files(eval_dir, fmt)
    if not eval_files:
        print(f"  No .eval files found for format '{fmt}'")
        return

    print(f"  Extracting {len(eval_files)} .eval files for format '{fmt}'...")
    all_records = []
    for eval_file in eval_files:
        samples = extract_samples_from_eval(eval_file)
        records = [eval_sample_to_training(s, cot=cot) for s in samples]
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

    # Split by UUID
    import json
    import random
    from collections import defaultdict

    uuid_to_records = defaultdict(list)
    for rec in unique:
        uuid_to_records[rec["metadata"].get("uuid", "")].append(rec)

    random.seed(42)
    uuids = list(uuid_to_records.keys())
    random.shuffle(uuids)
    split_idx = int(len(uuids) * 0.8)
    train = [rec for u in uuids[:split_idx] for rec in uuid_to_records[u]]
    val = [rec for u in uuids[split_idx:] for rec in uuid_to_records[u]]

    output_dir.mkdir(parents=True, exist_ok=True)
    for subset, name in [(train, "train.jsonl"), (val, "val.jsonl")]:
        path = output_dir / name
        with open(path, "w") as f:
            for rec in subset:
                f.write(json.dumps(rec) + "\n")
        print(f"  Saved {len(subset)} records to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download SGTR eval results from HuggingFace"
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
        "--extract", action="store_true",
        help="After download, run extraction to produce training JSONL",
    )
    parser.add_argument(
        "--extract-output",
        help="Override extraction output dir (default: data/training_data/{name}_{format}/)",
    )
    parser.add_argument(
        "--cot", action="store_true",
        help="Use CoT prompts during extraction",
    )
    args = parser.parse_args()

    if not args.list_only and not args.name:
        parser.error("--name is required unless --list-only is set")

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

    # Download
    from huggingface_hub import hf_hub_download

    cache_dir = Path(tempfile.mkdtemp(prefix="sgtr_hf_"))
    print(f"\nDownloading {len(matched)} files to {cache_dir}...")
    for filename in matched:
        hf_hub_download(
            repo_id=args.repo, repo_type="dataset",
            filename=filename, local_dir=cache_dir,
        )
        print(f"  Downloaded {filename}")

    # Reorganize
    output_dir = Path("data/original") / args.name
    print(f"\nReorganizing into {output_dir}...")
    copied = reorganize_files(matched, cache_dir, output_dir)

    for fmt, files in copied.items():
        if files:
            print(f"  {fmt.upper()}: {len(files)} files")

    # Optional extraction
    if args.extract:
        print("\nRunning extraction...")
        for fmt, files in copied.items():
            if not files:
                continue
            if args.extract_output:
                extract_dir = Path(args.extract_output)
            else:
                extract_dir = Path("data/training_data") / f"{args.name}_{fmt}"
            run_extraction(output_dir, fmt, extract_dir, cot=args.cot)

    # Cleanup temp dir
    shutil.rmtree(cache_dir, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
