from __future__ import annotations

import json
from pathlib import Path

import pytest

from sgtr_rl.transfer_result_sources import TransferRunResolver


def _write_status(run_dir: Path, status: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps({"status": status}) + "\n")


def test_old_resolver_picks_latest_completed(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_status(results_dir / "demo__20260301_000000", "completed")
    _write_status(results_dir / "demo__20260302_000000", "failed")
    _write_status(results_dir / "demo__20260303_000000", "completed")

    resolver = TransferRunResolver.old(results_dir=results_dir)
    resolved = resolver.resolve("demo__*")

    assert resolved.run_dir.name == "demo__20260303_000000"


def test_clean_resolver_filters_to_manifest_window(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_status(results_dir / "demo__20260324_130227", "completed")
    _write_status(results_dir / "demo__20260422_175803", "starting")
    _write_status(results_dir / "demo__20260422_180500", "completed")

    manifest_path = tmp_path / "20260422_175759__llama_self.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch": "llama_self",
                "started_at": "20260422_175759",
                "disable_wandb": True,
                "configs": ["experiments/demo/config.yaml"],
            }
        )
        + "\n"
    )

    resolver = TransferRunResolver.clean(results_dir=results_dir, manifest_paths=[manifest_path])
    resolved = resolver.resolve("demo__*")

    assert resolved.run_dir.name == "demo__20260422_180500"


def test_clean_resolver_rejects_prefix_outside_manifests(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_status(results_dir / "demo__20260422_180500", "completed")

    manifest_path = tmp_path / "20260422_175759__llama_self.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch": "llama_self",
                "started_at": "20260422_175759",
                "disable_wandb": True,
                "configs": ["experiments/other/config.yaml"],
            }
        )
        + "\n"
    )

    resolver = TransferRunResolver.clean(results_dir=results_dir, manifest_paths=[manifest_path])

    with pytest.raises(FileNotFoundError, match="not present in the selected clean manifests"):
        resolver.resolve("demo__*")
