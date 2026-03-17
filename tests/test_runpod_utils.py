"""Tests for RunPod launch helpers."""

from unittest.mock import patch

from sgtr_rl.scripts.runpod_utils import build_pod_request, build_startup_script
from sgtr_rl.config import TrainingConfig
from sgtr_rl.runtime_config import RuntimeConfig


def test_build_startup_script_embeds_runtime_and_train_command():
    script = build_startup_script(
        repo_url="https://github.com/example/SGTR-RL.git",
        repo_ref="abc123",
        runtime_yaml_text="backend: local\nartifacts:\n  root_dir: /runpod-volume/results\n",
        experiment_config_path="experiments/demo/config.yaml",
        group="smoke",
        exists="new",
        workspace_subdir="SGTR-RL",
        cache_dir="/runpod-volume/hf-cache",
    )

    assert "git clone https://github.com/example/SGTR-RL.git SGTR-RL" in script
    assert "git checkout abc123" in script
    assert "uv sync --frozen --python 3.12" in script
    assert "--group smoke" in script
    assert "backend: local" in script


def test_build_pod_request_uses_runtime_and_env(monkeypatch):
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
            experiment_config_path="experiments/demo/config.yaml",
            group=None,
            exists="new",
        )

    assert payload["imageName"] == runtime.runpod.image_name
    assert payload["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"]
    assert payload["networkVolumeId"] == "vol-123"
    assert payload["env"]["HF_TOKEN"] == "hf-test"
    assert payload["dockerEntrypoint"] == ["bash", "-lc"]
