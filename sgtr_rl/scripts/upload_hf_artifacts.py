"""Upload local result/checkpoint artifacts to the Hugging Face Hub.

Examples:
    python -m scripts.upload_hf_artifacts \
        results/verification/01_sft_pw_vs_qwen__20260311_142149

    python -m scripts.upload_hf_artifacts \
        --repo callumcanavan/sgtr-rl-qwen-pw-checkpoint \
        --repo-type model \
        results/local_run/checkpoints/final
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from dotenv import load_dotenv
from huggingface_hub import HfApi

DEFAULT_REPO = "SGTR-Geodesic/self-rec-results"
DEFAULT_IGNORE_PATTERNS = [
    ".DS_Store",
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
]


@dataclass(frozen=True)
class UploadTarget:
    """One local file or folder mapped to a target Hub path."""

    local_path: Path
    path_in_repo: str
    is_dir: bool


def _load_env() -> None:
    load_dotenv(Path(".env"))


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _normalize_repo_path(path: str) -> str:
    pure = PurePosixPath(path)
    normalized = pure.as_posix().lstrip("/")
    if normalized in {"", "."}:
        raise ValueError("path_in_repo cannot resolve to the repository root")
    return normalized


def _default_path_in_repo(local_path: Path, *, repo_root: Path) -> str:
    try:
        return local_path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return local_path.name


def build_upload_target(
    local_path: str | Path,
    *,
    repo_root: Path,
    path_in_repo_prefix: str | None = None,
) -> UploadTarget:
    """Map a local file/folder to its destination path in the Hub repo."""
    path = Path(local_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Upload path does not exist: {path}")

    if path_in_repo_prefix is None:
        path_in_repo = _default_path_in_repo(path, repo_root=repo_root)
    else:
        if path_in_repo_prefix in {"", ".", "/"}:
            path_in_repo = "" if path.is_dir() else path.name
        else:
            prefix = _normalize_repo_path(path_in_repo_prefix)
            path_in_repo = (PurePosixPath(prefix) / path.name).as_posix()

    return UploadTarget(local_path=path, path_in_repo=path_in_repo, is_dir=path.is_dir())


def _repo_url(repo_id: str, repo_type: str, path_in_repo: str) -> str:
    base = "https://huggingface.co"
    suffix = f"/tree/main/{path_in_repo}" if path_in_repo else "/tree/main"
    if repo_type == "dataset":
        return f"{base}/datasets/{repo_id}{suffix}"
    if repo_type == "space":
        return f"{base}/spaces/{repo_id}{suffix}"
    return f"{base}/{repo_id}{suffix}"


def _upload_target(
    api: HfApi,
    *,
    target: UploadTarget,
    repo_id: str,
    repo_type: str,
    commit_message: str,
    allow_patterns: list[str] | None,
    ignore_patterns: list[str] | None,
    revision: str | None,
    token: str | None,
) -> str:
    if target.is_dir:
        kwargs = {
            "folder_path": str(target.local_path),
            "repo_id": repo_id,
            "repo_type": repo_type,
            "commit_message": commit_message,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
            "revision": revision,
            "token": token,
        }
        if target.path_in_repo:
            kwargs["path_in_repo"] = target.path_in_repo
        api.upload_folder(**kwargs)
    else:
        api.upload_file(
            path_or_fileobj=str(target.local_path),
            path_in_repo=target.path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
            revision=revision,
            token=token,
        )
    return _repo_url(repo_id, repo_type, target.path_in_repo)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload local artifacts to the Hugging Face Hub")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Local file or folder paths to upload",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Target Hub repo id (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--repo-type",
        choices=["dataset", "model", "space"],
        default="dataset",
        help="Target repo type",
    )
    parser.add_argument(
        "--path-in-repo-prefix",
        default=None,
        help=(
            "Optional remote prefix. By default, paths under the workspace are "
            "mirrored relative to the repo root."
        ),
    )
    parser.add_argument(
        "--commit-message",
        default="Upload SGTR-RL artifacts",
        help="Commit message to use for uploaded files",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch or revision to upload to",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        dest="allow_patterns",
        default=None,
        help="Optional glob pattern to include (repeatable)",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        dest="ignore_patterns",
        default=None,
        help="Optional glob pattern to ignore (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads without contacting the Hub",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the target repo if needed",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    _load_env()
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    token = os.environ.get("HF_TOKEN")
    targets = [
        build_upload_target(
            path,
            repo_root=repo_root,
            path_in_repo_prefix=args.path_in_repo_prefix,
        )
        for path in args.paths
    ]

    ignore_patterns = list(DEFAULT_IGNORE_PATTERNS)
    if args.ignore_patterns:
        ignore_patterns.extend(args.ignore_patterns)

    if args.dry_run:
        payload = {
            "repo": args.repo,
            "repo_type": args.repo_type,
            "revision": args.revision,
            "targets": [
                {
                    **asdict(target),
                    "local_path": str(target.local_path),
                    "url": _repo_url(args.repo, args.repo_type, target.path_in_repo),
                }
                for target in targets
            ],
            "allow_patterns": args.allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        print(json.dumps(payload, indent=2))
        return

    api = HfApi(token=token)
    if args.create_repo:
        api.create_repo(
            repo_id=args.repo,
            repo_type=args.repo_type,
            exist_ok=True,
            private=False,
        )

    for target in targets:
        url = _upload_target(
            api,
            target=target,
            repo_id=args.repo,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            allow_patterns=args.allow_patterns,
            ignore_patterns=ignore_patterns,
            revision=args.revision,
            token=token,
        )
        print(url)


if __name__ == "__main__":
    main()
