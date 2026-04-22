"""Tests for runtime config loading and validation."""

import pytest
import yaml
from pydantic import ValidationError

from sgtr_rl.runtime_config import load_runtime_config


def test_load_default_runtime_config():
    runtime = load_runtime_config(None)
    assert runtime.backend == "tinker"
    assert runtime.artifacts.root_dir == "results"
    assert runtime.local.max_seq_length == 4096


def test_load_runtime_config_from_yaml(tmp_path):
    path = tmp_path / "runtime.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(
            {
                "backend": "local",
                "artifacts": {"root_dir": "/mnt/results"},
                "local": {
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "cache_dir": "/mnt/cache",
                },
                "runpod": {
                    "gpu_type_ids": ["NVIDIA A100 80GB PCIe"],
                    "network_volume_id": "vol-123",
                    "editable_deps": [
                        {
                            "repo_url": "https://github.com/example/dep.git",
                            "path": "_external/dep",
                            "ref": "main",
                        }
                    ],
                },
            },
            f,
        )

    runtime = load_runtime_config(path)
    assert runtime.backend == "local"
    assert runtime.artifacts.root_dir == "/mnt/results"
    assert runtime.local.device == "cuda"
    assert runtime.local.cache_dir == "/mnt/cache"
    assert runtime.runpod.gpu_type_ids == ["NVIDIA A100 80GB PCIe"]
    assert runtime.runpod.editable_deps[0].path == "_external/dep"


def test_runtime_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "runtime.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"backend": "local", "local": {"not_a_field": True}}, f)

    with pytest.raises(ValidationError):
        load_runtime_config(path)
