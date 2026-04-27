"""Tests for scripts.upload_hf_artifacts."""

from sgtr_rl.scripts.upload_hf_artifacts import build_upload_target


def test_build_upload_target_defaults_to_repo_relative_path(tmp_path):
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "results" / "verification" / "run_01"
    run_dir.mkdir(parents=True)

    target = build_upload_target(run_dir, repo_root=repo_root.resolve())

    assert target.local_path == run_dir.resolve()
    assert target.path_in_repo == "results/verification/run_01"
    assert target.is_dir is True


def test_build_upload_target_uses_prefix_when_requested(tmp_path):
    repo_root = tmp_path / "repo"
    file_path = repo_root / "checkpoints" / "final" / "adapter_model.safetensors"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("weights")

    target = build_upload_target(
        file_path,
        repo_root=repo_root.resolve(),
        path_in_repo_prefix="shared/checkpoints",
    )

    assert target.path_in_repo == "shared/checkpoints/adapter_model.safetensors"
    assert target.is_dir is False


def test_build_upload_target_uses_basename_outside_repo_root(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside_dir = tmp_path / "external_run"
    outside_dir.mkdir()

    target = build_upload_target(outside_dir, repo_root=repo_root.resolve())

    assert target.path_in_repo == outside_dir.name


def test_build_upload_target_allows_repo_root_for_direct_model_upload(tmp_path):
    repo_root = tmp_path / "repo"
    checkpoint_dir = repo_root / "results" / "run_01" / "checkpoints" / "final"
    checkpoint_dir.mkdir(parents=True)

    target = build_upload_target(
        checkpoint_dir,
        repo_root=repo_root.resolve(),
        path_in_repo_prefix=".",
    )

    assert target.path_in_repo == ""
    assert target.is_dir is True
