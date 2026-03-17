"""Launch a one-shot SGTR-RL training job on RunPod."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from sgtr_rl.scripts.runpod_utils import RunPodClient, build_pod_request
from sgtr_rl.config import load_training_config
from sgtr_rl.runtime_config import load_runtime_config

logger = logging.getLogger(__name__)


def _redact_payload(payload: dict) -> dict:
    sanitized = dict(payload)
    env = dict(sanitized.get("env", {}))
    sanitized["env"] = {key: "<redacted>" for key in env}
    return sanitized


def _validate_runtime(runtime, *, experiment_config_path: str) -> None:
    if runtime.backend != "local":
        raise ValueError("RunPod launcher currently supports the local backend only")
    if not runtime.runpod.gpu_type_ids:
        raise ValueError("runtime.runpod.gpu_type_ids must be set")
    if runtime.runpod.network_volume_id:
        mount_path = Path(runtime.runpod.volume_mount_path).resolve()
        results_root = Path(runtime.artifacts.root_dir).resolve()
        if not str(results_root).startswith(str(mount_path)):
            raise ValueError(
                "runtime.artifacts.root_dir must be under runtime.runpod.volume_mount_path "
                "when using a network volume"
            )
        if runtime.local.cache_dir:
            cache_dir = Path(runtime.local.cache_dir).resolve()
            if not str(cache_dir).startswith(str(mount_path)):
                raise ValueError(
                    "runtime.local.cache_dir should be under the mounted network volume "
                    "to persist model downloads between runs"
                )
    if not Path(experiment_config_path).exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_config_path}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Launch SGTR-RL training on RunPod")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--runtime", required=True, help="Path to runtime config YAML")
    parser.add_argument("--group", default=None, help="Optional run group subdirectory")
    parser.add_argument(
        "--exists",
        default="new",
        choices=["new", "error", "skip", "overwrite"],
        help="Existing-run policy passed through to scripts.train",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the RunPod request payload instead of launching the pod",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after pod creation instead of waiting for it to exit",
    )
    args = parser.parse_args()

    training_config = load_training_config(args.config)
    runtime = load_runtime_config(args.runtime)
    _validate_runtime(runtime, experiment_config_path=args.config)

    payload = build_pod_request(
        training_config=training_config,
        runtime=runtime,
        experiment_config_path=args.config,
        group=args.group,
        exists=args.exists,
    )

    if args.dry_run:
        print(json.dumps(_redact_payload(payload), indent=2, sort_keys=True))
        return

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        raise ValueError("RUNPOD_API_KEY must be set in the environment")

    client = RunPodClient(api_key)
    pod = client.create_pod(payload)
    pod_id = pod["id"]
    print(f"Created pod {pod_id}")

    if args.no_wait:
        return

    final_pod = client.wait_for_exit(
        pod_id,
        poll_interval_seconds=runtime.runpod.poll_interval_seconds,
    )
    print(f"Pod {pod_id} exited with desiredStatus={final_pod.get('desiredStatus')}")

    if runtime.runpod.terminate_on_exit:
        client.delete_pod(pod_id)
        print(f"Deleted pod {pod_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
