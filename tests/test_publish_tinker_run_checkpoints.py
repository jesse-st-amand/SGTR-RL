from __future__ import annotations

import json
from pathlib import Path

from scripts.publish_tinker_run_checkpoints import (
    load_final_tinker_paths,
    resolve_run_dir,
)


def test_resolve_run_dir_accepts_existing_repo_relative_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "example_run"
    run_dir.mkdir(parents=True)

    resolved = resolve_run_dir(tmp_path / "results", str(run_dir))

    assert resolved == run_dir.resolve()


def test_load_final_tinker_paths_returns_final_sampler_and_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "example_run"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    manifest_path = checkpoints_dir / "checkpoints.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                json.dumps({"sampler_path": "tinker://old/sampler"}),
                json.dumps(
                    {
                        "sampler_path": "tinker://final/sampler",
                        "state_path": "tinker://final/state",
                    }
                ),
            ]
        )
        + "\n"
    )

    assert load_final_tinker_paths(run_dir) == [
        "tinker://final/sampler",
        "tinker://final/state",
    ]
