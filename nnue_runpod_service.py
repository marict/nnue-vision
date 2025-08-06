import argparse
import os
import re
import shlex
import subprocess
from typing import Optional

import requests
import runpod
from graphql.language.print_string import print_string

import wandb
from config import load_config

DEFAULT_GPU_TYPE = "NVIDIA RTX 2000 Ada Generation"
REPO_URL = "https://github.com/marict/nnue-vision.git"


class RunPodError(Exception):
    pass


def _bash_c_quote(script: str) -> str:
    """Escape script for GraphQL."""
    command = f"bash -c {shlex.quote(script)}"
    return print_string(command)[1:-1]


def _resolve_gpu_id(gpu_type: str) -> str:
    """Return GPU id for given type."""
    try:
        gpus = runpod.get_gpus()
        for gpu in gpus:
            if gpu_type in {gpu.get("id"), gpu.get("displayName")}:
                return gpu["id"]
        raise RunPodError(f"GPU type '{gpu_type}' not found")
    except Exception as exc:
        raise RunPodError(f"Failed to list GPUs: {exc}") from exc


def _extract_project_name_from_config(
    config_path: str, model_type: str = "nnue"
) -> str:
    """Extract project name from config file."""
    config = load_config(config_path)
    return config.project_name


def _check_git_status() -> None:
    """Check for uncommitted git changes and fail if any are found."""
    subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )

    if result.stdout.strip():
        print("❌ Uncommitted changes detected!")
        raise RunPodError("Uncommitted changes detected!")


def _open_browser(url: str) -> None:
    """Try to open URL in browser."""
    chrome_commands = [
        "google-chrome",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "chrome",
        "chromium",
    ]

    for chrome_cmd in chrome_commands:
        try:
            subprocess.run(
                [chrome_cmd, url], check=True, capture_output=True, timeout=5
            )
            print(f"Opened W&B URL in browser: {url}")
            return
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            continue

    print(f"Could not open browser automatically. Visit: {url}")


def _create_docker_script(training_command: str) -> str:
    """Create Docker startup script."""
    commands = [
        "echo '[RUNPOD] Starting setup...'",
        "rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true",
        "apt-get update -y && apt-get install -y git",
        "cd /workspace",
        f"( [ -d repo/.git ] && git -C repo pull || git clone {REPO_URL} repo )",
        f"bash /workspace/repo/container_setup.sh {training_command}",
    ]
    return " && ".join(commands)


def start_cloud_training(
    train_args: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    *,
    api_key: Optional[str] = None,
    note: Optional[str] = None,
    script_name: str = "train.py",
) -> str:
    """Launch RunPod GPU instance for NNUE-Vision training."""

    _check_git_status()

    if note and re.findall(r"[^A-Za-z0-9_-]", note):
        raise ValueError(
            "Note contains invalid characters. Only letters, numbers, hyphens, and underscores allowed."
        )

    if not os.getenv("WANDB_API_KEY"):
        raise RunPodError("WANDB_API_KEY environment variable must be set")

    runpod.api_key = api_key or os.getenv("RUNPOD_API_KEY")
    if not runpod.api_key:
        raise RunPodError(
            "RunPod API key required. Set RUNPOD_API_KEY or use --api-key"
        )

    args_list = train_args.split() if train_args else []
    config_path = "config/train_nnue_default.py"
    model_type = "nnue"

    if args_list:
        model_type = args_list[0]
        try:
            config_index = args_list.index("--config")
            if config_index + 1 < len(args_list):
                config_path = args_list[config_index + 1]
        except ValueError:
            config_path = f"config/train_{model_type}_default.py"

    pod_name = _extract_project_name_from_config(config_path, model_type)

    placeholder_name = f"pod-id-pending{'-' + note if note else ''}"
    run = wandb.init(
        project=pod_name,
        name=placeholder_name,
        tags=["runpod", "remote-training", "nnue-vision"],
        notes="Remote NNUE-Vision training on RunPod",
    )
    wandb_run_id = run.id

    wandb_url = run.url + "/logs"
    _open_browser(wandb_url)

    cmd = f"{script_name} {train_args}"
    if note:
        cmd += f" --note={note}"
    if wandb_run_id:
        cmd += f" --wandb-run-id={wandb_run_id}"

    docker_script = _create_docker_script(cmd)
    final_docker_args = _bash_c_quote(docker_script)
    gpu_type_id = _resolve_gpu_id(gpu_type)

    try:
        pod = runpod.create_pod(
            name=pod_name,
            image_name="runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04",
            gpu_type_id=gpu_type_id,
            gpu_count=1,
            min_vcpu_count=6,
            min_memory_in_gb=16,
            volume_in_gb=1000,
            container_disk_in_gb=1000,
            network_volume_id="h3tyejvqqb",
            env={
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                "RUNPOD_API_KEY": runpod.api_key,
            },
            start_ssh=False,
            docker_args=final_docker_args,
        )
    except runpod.error.QueryError as exc:
        print("RunPod API QueryError:", exc)
        raise

    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")

    final_name = pod_id if not note else f"{pod_id} - {note}"
    wandb.run.name = final_name
    print(f"W&B run renamed to: {final_name}")
    print(f"Remote training will resume W&B run: {wandb_run_id}")
    print(f"Starting training job '{pod_name}' (pod {pod_id}) on {gpu_type}")

    return pod_id


def stop_runpod(pod_id: Optional[str] = None, api_key: Optional[str] = None) -> bool:
    """Stop the active RunPod instance."""
    pod_id = pod_id or os.getenv("RUNPOD_POD_ID")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")

    if not pod_id:
        return False

    print(f"Attempting to stop RunPod instance {pod_id}...")

    if not api_key:
        raise ValueError("RUNPOD_API_KEY not set.")

    try:
        runpod.api_key = api_key
        if hasattr(runpod, "stop_pod"):
            runpod.stop_pod(pod_id)
            print("Successfully requested pod stop (SDK).")
            return True
    except Exception as exc:
        print(f"SDK method failed: {exc}. Falling back to REST call...")

    try:
        url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
        headers = {"Authorization": f"Bearer {api_key}"}
        wandb.finish()
        resp = requests.post(url, headers=headers, timeout=10)
        resp.raise_for_status()
        print("Successfully requested pod stop (REST).")
        return True
    except Exception as exc:
        print(f"Failed to stop pod: {exc}")
        raise exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RunPod helper for NNUE-Vision training"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_parser = sub.add_parser("train", help="Start NNUE-Vision training pod")
    train_parser.add_argument(
        "train_args",
        nargs="*",
        help="Training arguments (e.g., nnue --config config/train_nnue_default.py)",
    )
    train_parser.add_argument(
        "--gpu-type", default=DEFAULT_GPU_TYPE, help="GPU type name"
    )
    train_parser.add_argument("--api-key", help="RunPod API key")
    train_parser.add_argument("--note", help="Note to add to the W&B run")
    train_parser.add_argument(
        "--script", default="train.py", help="Training script to run"
    )

    args = parser.parse_args()
    if args.cmd == "train":
        train_args_str = " ".join(args.train_args) if args.train_args else ""
        start_cloud_training(
            train_args_str,
            gpu_type=args.gpu_type,
            api_key=args.api_key,
            note=args.note,
            script_name=args.script,
        )
