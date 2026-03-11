"""List and clean up persistent Tinker checkpoints.

By default, deletion commands run in dry-run mode. Pass ``--apply`` to
actually remove checkpoints from Tinker.

Examples:
    python -m scripts.manage_tinker_checkpoints list
    python -m scripts.manage_tinker_checkpoints delete --before 2026-03-01
    python -m scripts.manage_tinker_checkpoints delete --before 2026-03-01 --keep-recent-runs 3
    python -m scripts.manage_tinker_checkpoints delete --run-id <run-id> --apply
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from sgtr_rl.logging_setup import setup_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointRecord:
    """Local view of a Tinker checkpoint."""

    run_id: str
    checkpoint_id: str
    checkpoint_type: str
    tinker_path: str
    size_bytes: int
    time: datetime
    expires_at: datetime | None


@dataclass(frozen=True)
class RunSummary:
    """Aggregate stats for a Tinker training run's checkpoints."""

    run_id: str
    checkpoints: tuple[CheckpointRecord, ...]
    first_time: datetime
    last_time: datetime
    total_size_bytes: int


def _load_tinker_env() -> None:
    """Load repo .env so Tinker credentials are available."""
    load_dotenv(Path(".env"))


def _list_user_checkpoints() -> list[CheckpointRecord]:
    """Fetch all checkpoints visible to the current Tinker user."""
    import tinker

    _load_tinker_env()
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    limit = 100
    offset = 0
    records: list[CheckpointRecord] = []

    while True:
        response = rest_client.list_user_checkpoints(limit=limit, offset=offset).result()
        batch = list(response.checkpoints)
        if not batch:
            break

        for ckpt in batch:
            run_id, checkpoint_id = _split_tinker_path(ckpt.tinker_path)
            records.append(
                CheckpointRecord(
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    checkpoint_type=ckpt.checkpoint_type,
                    tinker_path=ckpt.tinker_path,
                    size_bytes=ckpt.size_bytes,
                    time=ckpt.time,
                    expires_at=ckpt.expires_at,
                )
            )

        if len(batch) < limit:
            break
        offset += limit

    return records


def _split_tinker_path(tinker_path: str) -> tuple[str, str]:
    """Split ``tinker://<run_id>/<checkpoint_id>`` into pieces."""
    prefix = "tinker://"
    if not tinker_path.startswith(prefix):
        raise ValueError(f"Invalid Tinker path: {tinker_path}")
    remainder = tinker_path[len(prefix):]
    run_id, checkpoint_id = remainder.split("/", 1)
    return run_id, checkpoint_id


def _group_runs(records: Iterable[CheckpointRecord]) -> list[RunSummary]:
    """Group checkpoint records by Tinker run id."""
    by_run: dict[str, list[CheckpointRecord]] = defaultdict(list)
    for record in records:
        by_run[record.run_id].append(record)

    summaries = []
    for run_id, ckpts in by_run.items():
        sorted_ckpts = tuple(sorted(ckpts, key=lambda c: c.time))
        summaries.append(
            RunSummary(
                run_id=run_id,
                checkpoints=sorted_ckpts,
                first_time=sorted_ckpts[0].time,
                last_time=sorted_ckpts[-1].time,
                total_size_bytes=sum(c.size_bytes for c in sorted_ckpts),
            )
        )

    return sorted(summaries, key=lambda s: s.last_time, reverse=True)


def _load_local_run_labels(results_dir: Path) -> dict[str, str]:
    """Map Tinker run ids to local results directory names when available."""
    labels: dict[str, str] = {}
    for manifest_path in results_dir.glob("*/checkpoints/checkpoints.jsonl"):
        try:
            with open(manifest_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    for key in ("state_path", "sampler_path"):
                        tinker_path = payload.get(key)
                        if not tinker_path:
                            continue
                        run_id, _checkpoint_id = _split_tinker_path(tinker_path)
                        labels.setdefault(run_id, manifest_path.parent.parent.name)
        except (OSError, json.JSONDecodeError, ValueError):
            logger.warning("Failed to read %s", manifest_path, exc_info=True)
    return labels


def _parse_cutoff(value: str) -> datetime:
    """Parse YYYY-MM-DD into an aware UTC datetime at midnight."""
    dt = datetime.strptime(value, "%Y-%m-%d")
    return dt.replace(tzinfo=UTC)


def _format_size(size_bytes: int) -> str:
    """Human-readable binary size string."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


def _select_runs_to_delete(
    runs: list[RunSummary],
    run_ids: set[str],
    before: datetime | None,
    keep_recent_runs: int,
) -> list[RunSummary]:
    """Select run groups to delete based on explicit ids and/or age."""
    keep_ids = {run.run_id for run in runs[:keep_recent_runs]}

    selected = []
    for run in runs:
        explicit_match = run.run_id in run_ids if run_ids else False
        age_match = before is not None and run.last_time < before
        if run.run_id in keep_ids:
            continue
        if explicit_match or age_match:
            selected.append(run)
    return selected


def _print_runs(runs: list[RunSummary], local_labels: dict[str, str] | None = None) -> None:
    """Print a compact run-level inventory."""
    local_labels = local_labels or {}
    total_bytes = 0
    for run in runs:
        total_bytes += run.total_size_bytes
        types = ",".join(sorted({ckpt.checkpoint_type for ckpt in run.checkpoints}))
        local_name = local_labels.get(run.run_id)
        local_suffix = f" | local={local_name}" if local_name else ""
        print(
            f"{run.last_time.isoformat()} | {run.run_id} | "
            f"{len(run.checkpoints)} ckpts | {types} | {_format_size(run.total_size_bytes)}"
            f"{local_suffix}"
        )
    print(
        f"\nRuns: {len(runs)} | Checkpoints: {sum(len(run.checkpoints) for run in runs)} "
        f"| Total: {_format_size(total_bytes)}"
    )


def _print_breakdown(
    runs: list[RunSummary],
    local_labels: dict[str, str] | None = None,
) -> None:
    """Print aggregate checkpoint counts and sizes."""
    local_labels = local_labels or {}
    checkpoints = [ckpt for run in runs for ckpt in run.checkpoints]
    by_type: dict[str, int] = defaultdict(int)
    bytes_by_type: dict[str, int] = defaultdict(int)

    for ckpt in checkpoints:
        by_type[ckpt.checkpoint_type] += 1
        bytes_by_type[ckpt.checkpoint_type] += ckpt.size_bytes

    matched_local = sum(1 for run in runs if run.run_id in local_labels)
    unmatched_local = len(runs) - matched_local
    total_bytes = sum(ckpt.size_bytes for ckpt in checkpoints)

    print("\nBreakdown:")
    print(f"  Runs: {len(runs)}")
    print(f"  Checkpoints: {len(checkpoints)}")
    print(f"  Total size: {_format_size(total_bytes)}")
    print(f"  Non-expiring: {sum(ckpt.expires_at is None for ckpt in checkpoints)}")
    print(f"  Expiring: {sum(ckpt.expires_at is not None for ckpt in checkpoints)}")
    print(f"  With local run label: {matched_local}")
    print(f"  Without local run label: {unmatched_local}")
    for checkpoint_type in sorted(by_type):
        print(
            f"  {checkpoint_type}: {by_type[checkpoint_type]} "
            f"({_format_size(bytes_by_type[checkpoint_type])})"
        )


def _delete_runs(runs: list[RunSummary], apply: bool) -> None:
    """Delete all checkpoints for the selected runs."""
    import tinker

    _load_tinker_env()
    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    for run in runs:
        logger.info(
            "%s run %s (%d checkpoints, %s)",
            "Deleting" if apply else "Would delete",
            run.run_id,
            len(run.checkpoints),
            _format_size(run.total_size_bytes),
        )
        for ckpt in run.checkpoints:
            logger.info("  %s", ckpt.tinker_path)
            if apply:
                rest_client.delete_checkpoint_from_tinker_path(ckpt.tinker_path).result()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description="Manage persistent Tinker checkpoints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List checkpoints grouped by run")
    list_parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Print aggregate counts and sizes after the run list",
    )

    delete_parser = subparsers.add_parser(
        "delete",
        help="Delete checkpoints by explicit run id and/or age",
    )
    delete_parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Tinker run id to delete; may be provided multiple times",
    )
    delete_parser.add_argument(
        "--before",
        type=_parse_cutoff,
        default=None,
        help="Delete runs whose latest checkpoint is before YYYY-MM-DD",
    )
    delete_parser.add_argument(
        "--keep-recent-runs",
        type=int,
        default=0,
        help="Always keep the N most recent runs, even if they match other filters",
    )
    delete_parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete checkpoints; default is dry-run",
    )
    delete_parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Print aggregate counts and sizes for the selected runs",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging("manage_tinker_checkpoints")

    records = _list_user_checkpoints()
    runs = _group_runs(records)
    local_labels = _load_local_run_labels(Path("results"))

    if args.command == "list":
        _print_runs(runs, local_labels=local_labels)
        if args.breakdown:
            _print_breakdown(runs, local_labels=local_labels)
        return

    if not args.run_id and args.before is None:
        parser.error("delete requires at least one of --run-id or --before")

    selected = _select_runs_to_delete(
        runs,
        run_ids=set(args.run_id),
        before=args.before,
        keep_recent_runs=args.keep_recent_runs,
    )

    if not selected:
        logger.info("No runs matched the deletion criteria")
        return

    _print_runs(selected, local_labels=local_labels)
    if args.breakdown:
        _print_breakdown(selected, local_labels=local_labels)
    if not args.apply:
        logger.info("Dry run only. Re-run with --apply to delete these checkpoints.")
        return

    _delete_runs(selected, apply=True)
    logger.info("Deleted %d runs", len(selected))


if __name__ == "__main__":
    main()
