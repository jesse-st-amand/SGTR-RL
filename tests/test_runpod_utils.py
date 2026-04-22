"""Tests for RunPod launch helpers."""
from unittest.mock import patch

from scripts.runpod_utils import build_pod_request, build_startup_script
from sgtr_rl.config import TrainingConfig
from sgtr_rl.runtime_config import RuntimeConfig


def test_build_startup_script_embeds_runtime_and_train_command():
    runtime = RuntimeConfig(
        runpod={
            "editable_deps": [
                {
                    "repo_url": "https://github.com/example/dep.git",
                    "path": "_external/dep",
                    "ref": "main",
                }
            ]
        }
    )
    script = build_startup_script(
        repo_url="https://github.com/example/SGTR-RL.git",
        repo_ref="abc123",
        runtime_yaml_text="backend: local\nartifacts:\n  root_dir: /runpod-volume/results\n",
        experiment_config_yaml="experiment_name: demo\n",
        editable_deps=runtime.runpod.editable_deps,
        group="smoke",
        exists="new",
        workspace_subdir="SGTR-RL",
        cache_dir="/runpod-volume/hf-cache",
    )

    assert "git clone https://github.com/example/SGTR-RL.git SGTR-RL" in script
    assert "git checkout abc123" in script
    assert "git clone https://github.com/example/dep.git _external/dep" in script
    assert "uv sync --frozen --python 3.12" in script
    assert "--group smoke" in script
    assert "backend: local" in script
    assert "experiment_name: demo" in script


def test_build_pod_request_uses_runtime_and_env(monkeypatch, tmp_path):
    runtime = RuntimeConfig(
        backend="local",
        artifacts={"root_dir": "/runpod-volume/sgtr-rl/results"},
        local={"cache_dir": "/runpod-volume/sgtr-rl/hf-cache"},
        runpod={
            "gpu_type_ids": ["NVIDIA A100 80GB PCIe"],
            "network_volume_id": "vol-123",
        },
    )
    training = TrainingConfig(experiment_name="demo")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: demo\n")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-test")

    with (
        patch(
            "scripts.runpod_utils.infer_repo_url",
            return_value="https://github.com/example/repo.git",
        ),
        patch("scripts.runpod_utils.infer_repo_ref", return_value="deadbeef"),
    ):
        payload = build_pod_request(
            training_config=training,
            runtime=runtime,
            experiment_config_path=str(config_path),
            group=None,
            exists="new",
        )

    assert payload["imageName"] == runtime.runpod.image_name
    assert payload["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"]
    assert payload["networkVolumeId"] == "vol-123"
    assert payload["env"]["HF_TOKEN"] == "hf-test"
    assert payload["dockerEntrypoint"] == ["bash", "-lc"]


def test_build_pod_request_uses_env_volume_fallback(monkeypatch, tmp_path):
    runtime = RuntimeConfig(
        backend="local",
        runpod={"gpu_type_ids": ["NVIDIA A100 80GB PCIe"]},
    )
    training = TrainingConfig(experiment_name="demo")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_name: demo\n")
    monkeypatch.setenv("HF_TOKEN", "hf-test")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-test")
    monkeypatch.setenv("RUNPOD_NETWORK_VOLUME_ID", "vol-from-env")

    with (
        patch("scripts.runpod_utils.infer_repo_url", return_value="https://github.com/example/repo.git"),
        patch("scripts.runpod_utils.infer_repo_ref", return_value="deadbeef"),
    ):
        payload = build_pod_request(
            training_config=training,
            runtime=runtime,
            experiment_config_path=str(config_path),
            group=None,
            exists="new",
        )

    assert payload["networkVolumeId"] == "vol-from-env"
