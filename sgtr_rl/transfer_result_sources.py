from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_RESULTS_DIR = ROOT / "results"
DEFAULT_EXTERNAL_RESULTS_DIR = (
    ROOT.parent / "self-rec-research" / "_external" / "SGTR-RL" / "results"
)
DEFAULT_BATCH_LOG_DIR = ROOT / "logs" / "transfer_repro_batches"
DEFAULT_STANDARD_BATCHES = ("llama_self", "qwen_self", "oss20_self")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _run_status(run_dir: Path) -> str | None:
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return None
    try:
        return _read_json(status_path).get("status")
    except Exception:
        return None


def _run_timestamp(run_dir: Path) -> str | None:
    if "__" not in run_dir.name:
        return None
    return run_dir.name.rsplit("__", 1)[-1]


def _prefix_from_glob(glob_pattern: str) -> str:
    if not glob_pattern.endswith("__*"):
        raise ValueError(
            f"Expected an exact run-prefix glob ending in '__*', got: {glob_pattern}"
        )
    return glob_pattern[:-3]


def latest_batch_manifests(
    log_dir: Path = DEFAULT_BATCH_LOG_DIR,
    batch_names: tuple[str, ...] = DEFAULT_STANDARD_BATCHES,
) -> list[Path]:
    manifests: list[Path] = []
    for batch_name in batch_names:
        candidates = sorted(log_dir.glob(f"*__{batch_name}.json"))
        if not candidates:
            raise FileNotFoundError(f"No batch manifests found for {batch_name} in {log_dir}")
        preferred = [
            path for path in candidates if bool(_read_json(path).get("disable_wandb", False))
        ]
        manifests.append(preferred[-1] if preferred else candidates[-1])
    return manifests


def _manifest_prefix_bounds(manifest_paths: list[Path]) -> dict[str, str]:
    prefix_bounds: dict[str, str] = {}
    for manifest_path in manifest_paths:
        manifest = _read_json(manifest_path)
        started_at = str(manifest["started_at"])
        for config in manifest["configs"]:
            prefix = Path(config).parent.name
            existing = prefix_bounds.get(prefix)
            if existing is None or started_at < existing:
                prefix_bounds[prefix] = started_at
    return prefix_bounds


@dataclass(frozen=True)
class ResolvedRun:
    glob_pattern: str
    prefix: str
    run_dir: Path
    source_name: str


class TransferRunResolver:
    def __init__(
        self,
        *,
        source_name: str,
        results_dir: Path,
        manifest_paths: list[Path] | None = None,
    ) -> None:
        self.source_name = source_name
        self.results_dir = results_dir
        self.manifest_paths = manifest_paths or []
        self._prefix_bounds = _manifest_prefix_bounds(self.manifest_paths)

    @classmethod
    def old(
        cls,
        *,
        results_dir: Path = DEFAULT_EXTERNAL_RESULTS_DIR,
    ) -> TransferRunResolver:
        return cls(source_name="old", results_dir=results_dir)

    @classmethod
    def clean(
        cls,
        *,
        results_dir: Path = DEFAULT_LOCAL_RESULTS_DIR,
        manifest_paths: list[Path] | None = None,
    ) -> TransferRunResolver:
        selected = manifest_paths or latest_batch_manifests()
        return cls(source_name="clean", results_dir=results_dir, manifest_paths=selected)

    def describe(self) -> dict:
        return {
            "source": self.source_name,
            "results_dir": str(self.results_dir),
            "manifest_paths": [str(path) for path in self.manifest_paths],
            "prefix_count": len(self._prefix_bounds),
        }

    def resolve(self, glob_pattern: str, *, required: bool = True) -> ResolvedRun | None:
        prefix = _prefix_from_glob(glob_pattern)
        lower_bound = self._prefix_bounds.get(prefix)

        if self.source_name == "clean" and lower_bound is None:
            if required:
                raise FileNotFoundError(
                    f"{prefix} is not present in the selected clean manifests: "
                    f"{self.manifest_paths}"
                )
            return None

        matches = sorted(self.results_dir.glob(glob_pattern))
        completed: list[Path] = []
        candidate_statuses: list[str] = []
        for match in matches:
            timestamp = _run_timestamp(match)
            if lower_bound is not None and (timestamp is None or timestamp < lower_bound):
                continue
            status = _run_status(match)
            candidate_statuses.append(f"{match.name}:{status}")
            if status == "completed":
                completed.append(match)

        if not completed:
            if required:
                lower_text = f" after {lower_bound}" if lower_bound else ""
                status_text = ", ".join(candidate_statuses) if candidate_statuses else "none"
                raise FileNotFoundError(
                    f"No completed {self.source_name} runs for {glob_pattern}{lower_text}. "
                    f"Candidates: {status_text}"
                )
            return None

        return ResolvedRun(
            glob_pattern=glob_pattern,
            prefix=prefix,
            run_dir=completed[-1],
            source_name=self.source_name,
        )
