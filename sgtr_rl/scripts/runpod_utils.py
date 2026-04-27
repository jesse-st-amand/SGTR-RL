"""Helpers for building and launching one-shot RunPod jobs."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from sgtr_rl.config import TrainingConfig
from sgtr_rl.runtime_config import RuntimeConfig

RUNPOD_API_BASE = "https://rest.runpod.io/v1"


def infer_repo_url() -> str:
    """Return the current git remote URL."""
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def infer_repo_ref() -> str:
    """Return the current git commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_pod_name(experiment_name: str) -> str:
    """Build a timestamped RunPod name for the launched training job."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"sgtr-{experiment_name}-{timestamp}".replace("_", "-")


def resolve_runpod_env(runtime: RuntimeConfig) -> dict[str, str]:
    """Resolve pod environment variables from local env passthrough and literals."""
    env = dict(runtime.runpod.env)
    for name in runtime.runpod.env_passthrough:
        value = os.getenv(name)
        if value is None:
            raise ValueError(
                f"Missing environment variable {name!r} required by runpod.env_passthrough"
            )
        env[name] = value
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_remote_runtime_yaml(runtime: RuntimeConfig) -> str:
    """Serialize the runtime config embedded into the remote job."""
    return yaml.safe_dump(runtime.model_dump(mode="json"), sort_keys=False)


def build_train_command(
    *,
    experiment_config_path: str,
    remote_runtime_path: str,
    group: str | None,
    exists: str,
) -> str:
    parts = [
        "uv",
        "run",
        "--python",
        "3.12",
        "python",
        "-m",
        "sgtr_rl.scripts.train",
        "--config",
        experiment_config_path,
        "--runtime",
        remote_runtime_path,
    ]
    if group:
        parts.extend(["--group", group])
    if exists:
        parts.extend(["--exists", exists])
    return shlex.join(parts)


def build_startup_script(
    *,
    repo_url: str,
    repo_ref: str,
    runtime_yaml_text: str,
    experiment_config_yaml: str,
    editable_deps: list,
    group: str | None,
    exists: str,
    workspace_subdir: str,
    cache_dir: str | None,
) -> str:
    """Build the one-shot shell script that runs inside the launched pod."""
    remote_runtime_path = "/tmp/sgtr-runtime.yaml"
    remote_config_path = "/tmp/sgtr-experiment.yaml"
    train_command = build_train_command(
        experiment_config_path=remote_config_path,
        remote_runtime_path=remote_runtime_path,
        group=group,
        exists=exists,
    )

    lines = [
        "set -euo pipefail",
        "python -m pip install --upgrade pip uv",
        "uv python install 3.12",
        "mkdir -p /root/workspace",
        "cd /root/workspace",
        f"git clone {shlex.quote(repo_url)} {shlex.quote(workspace_subdir)}",
        f"cd {shlex.quote(workspace_subdir)}",
        f"git checkout {shlex.quote(repo_ref)}",
    ]

    # Clone editable dependencies into subdirectories of the workspace
    for dep in editable_deps:
        parent = str(Path(dep.path).parent)
        lines.append(f"mkdir -p {shlex.quote(parent)}")
        clone_cmd = f"git clone {shlex.quote(dep.repo_url)} {shlex.quote(dep.path)}"
        lines.append(clone_cmd)
        if dep.ref:
            lines.append(
                f"git -C {shlex.quote(dep.path)} checkout {shlex.quote(dep.ref)}"
            )

    lines.extend([
        f"cat > {shlex.quote(remote_runtime_path)} <<'YAML'",
        runtime_yaml_text.rstrip(),
        "YAML",
        f"cat > {shlex.quote(remote_config_path)} <<'EXPYAML'",
        experiment_config_yaml.rstrip(),
        "EXPYAML",
    ])
    if cache_dir:
        lines.extend(
            [
                f"mkdir -p {shlex.quote(cache_dir)}",
                f"export HF_HOME={shlex.quote(cache_dir)}",
                f"export TRANSFORMERS_CACHE={shlex.quote(cache_dir)}",
                f"export HF_HUB_CACHE={shlex.quote(cache_dir)}",
            ]
        )
    lines.extend(
        [
            "uv sync --frozen --python 3.12",
            train_command,
        ]
    )
    return "\n".join(lines)


def build_pod_request(
    *,
    training_config: TrainingConfig,
    runtime: RuntimeConfig,
    experiment_config_path: str,
    group: str | None,
    exists: str,
) -> dict[str, Any]:
    """Build the RunPod REST request payload."""
    repo_url = runtime.runpod.repo_url or infer_repo_url()
    repo_ref = runtime.runpod.repo_ref or infer_repo_ref()
    env = resolve_runpod_env(runtime)
    experiment_config_yaml = Path(experiment_config_path).read_text()
    startup_script = build_startup_script(
        repo_url=repo_url,
        repo_ref=repo_ref,
        runtime_yaml_text=build_remote_runtime_yaml(runtime),
        experiment_config_yaml=experiment_config_yaml,
        editable_deps=runtime.runpod.editable_deps,
        group=group,
        exists=exists,
        workspace_subdir=runtime.runpod.workspace_subdir,
        cache_dir=runtime.local.cache_dir,
    )

    # Resolve network volume ID: explicit config value, else $RUNPOD_NETWORK_VOLUME_ID
    network_volume_id = runtime.runpod.network_volume_id or os.getenv(
        "RUNPOD_NETWORK_VOLUME_ID"
    )

    payload: dict[str, Any] = {
        "name": make_pod_name(training_config.experiment_name),
        "cloudType": runtime.runpod.cloud_type,
        "computeType": "GPU",
        "gpuCount": runtime.runpod.gpu_count,
        "gpuTypeIds": runtime.runpod.gpu_type_ids,
        "gpuTypePriority": "custom" if runtime.runpod.gpu_type_ids else "availability",
        "imageName": runtime.runpod.image_name,
        "containerDiskInGb": runtime.runpod.container_disk_gb,
        "volumeMountPath": runtime.runpod.volume_mount_path,
        "ports": ["22/tcp"],
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [startup_script],
        "env": env,
    }
    if network_volume_id:
        payload["networkVolumeId"] = network_volume_id
    return payload


class RunPodClient:
    """Tiny REST client for the RunPod Pod lifecycle APIs."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{RUNPOD_API_BASE}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request) as response:
                if response.status == 204:
                    return None
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise RuntimeError(f"RunPod API {method} {path} failed: {detail}") from exc

    def create_pod(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/pods", payload=payload)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")

    def delete_pod(self, pod_id: str) -> None:
        self._request("DELETE", f"/pods/{pod_id}")

    def wait_for_exit(self, pod_id: str, *, poll_interval_seconds: int) -> dict[str, Any]:
        while True:
            pod = self.get_pod(pod_id)
            if pod.get("desiredStatus") in {"EXITED", "TERMINATED"}:
                return pod
            time.sleep(poll_interval_seconds)
