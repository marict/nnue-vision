import argparse
import os
import re
import shlex
import subprocess

import requests
import runpod

import wandb

# Available GPU types for RunPod:
# - NVIDIA A40, A100 80GB PCIe, A100-SXM4-80GB, B200
# - NVIDIA H100 PCIe, H100 80GB HBM3, H100 NVL, H200
# - NVIDIA L40, L40S, RTX 6000 Ada Generation, RTX A6000
# - NVIDIA RTX PRO 6000 Blackwell Workstation Edition

# DEFAULT_GPU_TYPE = "NVIDIA RTX 6000 Ada Generation"  # Available in WA network volume
DEFAULT_GPU_TYPE = "NVIDIA RTX 2000 Ada Generation"
# TODO: Update this to your actual NNUE-Vision repository URL
REPO_URL = "https://github.com/YOUR_USERNAME/nnue-vision.git"


class RunPodError(Exception):
    """Custom exception for RunPod operations."""


def _validate_note(note: str) -> None:
    """Validate note contains only safe characters."""
    if not note:
        return

    # Remove spaces from allowed characters since they cause GraphQL issues
    invalid_chars = re.findall(r"[^A-Za-z0-9_-]", note)
    if invalid_chars:
        unique_invalid = sorted(set(invalid_chars))
        raise ValueError(
            f"Note contains invalid characters: {unique_invalid}. "
            "Only letters, numbers, hyphens, and underscores are allowed."
        )


def _resolve_gpu_id(gpu_type: str) -> str:
    """Return the GPU id for ``gpu_type`` which may be a name or id."""
    try:
        gpus = runpod.get_gpus()
    except Exception as exc:  # pragma: no cover - network errors
        raise RunPodError(f"Failed to list GPUs: {exc}") from exc

    for gpu in gpus:
        if gpu_type in {gpu.get("id"), gpu.get("displayName")}:
            return gpu["id"]
    raise RunPodError(f"GPU type '{gpu_type}' not found")


def _extract_project_name(train_args: str) -> str:
    """Extract the project name from training arguments or use default."""
    # Look for --project_name in arguments
    args_list = train_args.split()
    try:
        project_idx = args_list.index("--project_name")
        if project_idx + 1 < len(args_list):
            return args_list[project_idx + 1]
    except ValueError:
        pass
    return "nnue-vision-train"


def _build_training_command(
    train_args: str,
    keep_alive: bool,
    note: str | None,
    wandb_run_id: str | None,
    script_name: str = "train.py",
) -> str:
    """Build the complete training command with all flags."""
    cmd = f"{script_name} {train_args}"

    if keep_alive:
        cmd += " --keep-alive"

    if note:
        cmd += f" --note={note}"

    if wandb_run_id:
        cmd += f" --wandb-run-id={wandb_run_id}"

    return cmd


def _create_docker_script(training_command: str) -> str:
    """Create the Docker startup script for the RunPod instance."""
    return (
        f"apt-get update && apt-get install -y git && "
        f"cd /workspace && "
        f"( [ -d repo/.git ] && git -C repo pull || git clone {REPO_URL} repo ) && "
        f"bash /workspace/repo/container_setup.sh {training_command}"
    )


def start_cloud_training(
    train_args: str,
    gpu_type: str = DEFAULT_GPU_TYPE,
    *,
    api_key: str | None = None,
    keep_alive: bool = False,
    note: str | None = None,
    script_name: str = "train.py",
) -> str:
    """Launch a RunPod GPU instance and run NNUE-Vision training automatically."""

    # Validate inputs
    _validate_note(note)

    # Require WANDB_API_KEY so that a W&B run (and run_id) is always available
    if not os.getenv("WANDB_API_KEY"):
        raise RunPodError(
            "WANDB_API_KEY environment variable must be set when launching training through runpod_service."
        )

    # Set up RunPod API
    runpod.api_key = (
        api_key or os.getenv("RUNPOD_API_KEY") or getattr(runpod, "api_key", None)
    )
    if not runpod.api_key:
        raise RunPodError(
            "RunPod API key is required. Provide via --api-key or set RUNPOD_API_KEY"
        )

    # Extract project name from training arguments
    pod_name = _extract_project_name(train_args)
    project_name = pod_name

    # ------------------------------------------------------------------ #
    # 1. Create a local W&B run FIRST to obtain run_id
    # ------------------------------------------------------------------ #
    placeholder_name = f"pod-id-pending{'-' + note if note else ''}"
    wandb_result = init_local_wandb_and_open_browser(project_name, placeholder_name)
    wandb_run_id: str | None = None
    if wandb_result:
        _, wandb_run_id = wandb_result

    # Build training command (wandb_run_id guaranteed because WANDB_API_KEY is required)
    training_command = _build_training_command(
        train_args, keep_alive, note, wandb_run_id, script_name
    )
    docker_script = _create_docker_script(training_command)
    final_docker_args = f"bash -c {shlex.quote(docker_script)}"

    # ------------------------------------------------------------------ #
    # 2. Create RunPod instance
    # ------------------------------------------------------------------ #
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
            network_volume_id="h3tyejvqqb",  # Update this to your network volume ID
            env={
                "WANDB_API_KEY": os.getenv("WANDB_API_KEY", ""),
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                "RUNPOD_API_KEY": runpod.api_key,
            },
            start_ssh=False,
            docker_args=final_docker_args,
        )
    except runpod.error.QueryError as exc:  # type: ignore[attr-defined]
        print("RunPod API QueryError:", exc)
        raise

    pod_id = pod.get("id")
    if not pod_id:
        raise RunPodError("RunPod API did not return a pod id")

    if not wandb_run_id:
        raise RunPodError("W&B run id not found")

    # ------------------------------------------------------------------ #
    # 3. Rename local W&B run to final name (pod_id or pod_id-note)
    # ------------------------------------------------------------------ #
    try:
        final_name = pod_id if not note else f"{pod_id} - {note}"
        wandb.run.name = final_name
        print(f"W&B run renamed to: {final_name}")
        print(f"Remote training will resume W&B run: {wandb_run_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to rename W&B run: {exc}")

    print(
        f"Starting NNUE-Vision training job '{pod_name}' (pod {pod_id}) on {gpu_type}"
    )
    return pod_id


def init_local_wandb_and_open_browser(
    project_name: str, run_name: str
) -> tuple[str, str] | None:
    """Initialize W&B project locally and open the run URL in browser."""
    if not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not set, skipping local W&B initialization")
        return None

    try:
        # Initialize W&B run
        run = wandb.init(
            project=project_name,
            name=run_name,
            tags=["runpod", "remote-training", "nnue-vision"],
            notes=f"Remote NNUE-Vision training on RunPod instance {run_name}",
        )

        wandb_url = run.url + "/logs"
        wandb_run_id = run.id

        # Try to open in browser
        _open_browser(wandb_url)

        return (wandb_url, wandb_run_id)

    except Exception as e:
        print(f"Failed to initialize W&B locally: {e}")
        return None


def _open_browser(url: str) -> None:
    """Attempt to open URL in Chrome browser."""
    chrome_commands = [
        "google-chrome",  # Linux
        "google-chrome-stable",  # Linux alternative
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS
        "chrome",  # Windows/generic
        "chromium",  # Linux alternative
        "chromium-browser",  # Linux alternative
    ]

    for chrome_cmd in chrome_commands:
        try:
            subprocess.run(
                [chrome_cmd, url],
                check=True,
                capture_output=True,
                timeout=5,
            )
            print(f"Opened W&B URL in Chrome: {url}")
            return
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            continue

    print(f"Could not open Chrome. Please manually visit: {url}")


def stop_runpod(pod_id: str | None = None, api_key: str | None = None) -> bool:
    """Stop the active RunPod instance."""
    pod_id = pod_id or os.getenv("RUNPOD_POD_ID")
    api_key = api_key or os.getenv("RUNPOD_API_KEY")

    if not pod_id or not api_key:
        print("RUNPOD_POD_ID or RUNPOD_API_KEY not set – skipping pod stop.")
        return False

    # Try the Python SDK first
    try:
        runpod.api_key = api_key
        if hasattr(runpod, "stop_pod"):
            runpod.stop_pod(pod_id)
            print("Successfully requested pod stop (SDK).")
            return True
    except Exception as exc:  # noqa: BLE001
        print(f"SDK method failed: {exc}. Falling back to REST call...")

    # Fallback to direct REST API
    try:
        url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.post(url, headers=headers, timeout=10)
        resp.raise_for_status()
        print("Successfully requested pod stop (REST).")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to stop pod: {exc}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RunPod helper for NNUE-Vision training"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Start NNUE-Vision training pod")
    t.add_argument(
        "train_args",
        nargs="*",
        help="Training arguments (e.g., --max_epochs 100 --batch_size 64)",
    )
    t.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE, help="GPU type name")
    t.add_argument("--api-key", help="RunPod API key")
    t.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep pod alive after training completes (disables auto-stop)",
    )
    t.add_argument("--note", help="Note to add to the W&B run")
    t.add_argument(
        "--script",
        default="train.py",
        help="Training script to run",
    )

    args = parser.parse_args()
    if args.cmd == "train":
        train_args_str = " ".join(args.train_args) if args.train_args else ""
        start_cloud_training(
            train_args_str,
            gpu_type=args.gpu_type,
            api_key=args.api_key,
            keep_alive=args.keep_alive,
            note=args.note,
            script_name=args.script,
        )
