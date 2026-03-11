"""Runtime configuration for local GPU and provider-specific execution."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ArtifactRuntimeConfig(BaseModel):
    """Artifact and cache paths for a training run."""

    model_config = ConfigDict(extra="forbid")

    root_dir: str = "results"


class LocalRuntimeConfig(BaseModel):
    """Runtime options for local GPU / single-node HF training."""

    model_config = ConfigDict(extra="forbid")

    device: Literal["auto", "cuda", "cpu"] = "auto"
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    max_seq_length: int = 4096
    eval_batch_size: int = 8
    gradient_checkpointing: bool = True
    load_in_4bit: bool = False
    attention_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "sdpa"
    cache_dir: str | None = None
    lora_alpha: int | None = None
    lora_dropout: float = 0.0
    target_modules: str | list[str] = "all-linear"


class RunPodRuntimeConfig(BaseModel):
    """Runtime options for launching a one-shot RunPod job."""

    model_config = ConfigDict(extra="forbid")

    image_name: str = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
    gpu_type_ids: list[str] = Field(default_factory=list)
    gpu_count: int = 1
    cloud_type: Literal["ALL", "SECURE", "COMMUNITY"] = "SECURE"
    container_disk_gb: int = 50
    volume_mount_path: str = "/runpod-volume"
    network_volume_id: str | None = None
    repo_url: str | None = None
    repo_ref: str | None = None
    workspace_subdir: str = "SGTR-RL"
    env_passthrough: list[str] = Field(default_factory=lambda: ["HF_TOKEN", "WANDB_API_KEY"])
    env: dict[str, str] = Field(default_factory=dict)
    poll_interval_seconds: int = 30
    terminate_on_exit: bool = True


class RuntimeConfig(BaseModel):
    """Runtime configuration separate from experiment/training config."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["tinker", "local"] = "tinker"
    artifacts: ArtifactRuntimeConfig = Field(default_factory=ArtifactRuntimeConfig)
    local: LocalRuntimeConfig = Field(default_factory=LocalRuntimeConfig)
    runpod: RunPodRuntimeConfig = Field(default_factory=RunPodRuntimeConfig)


def load_runtime_config(yaml_path: str | Path | None) -> RuntimeConfig:
    """Load a RuntimeConfig from YAML, or return defaults when omitted."""
    if yaml_path is None:
        return RuntimeConfig()

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Runtime config not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Runtime config must be a YAML mapping")

    return RuntimeConfig(**raw)
