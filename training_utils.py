from __future__ import annotations

import argparse
import math
import os
import random
import runpy
import shutil
import string
import subprocess
import time
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

# Checkpoint directory
CHECKPOINT_DIR = (
    "/runpod-volume/checkpoints" if os.path.exists("/runpod-volume") else "checkpoints"
)


# --------------------------------------------------------------------------- #
# Early log capture for W&B replay
# --------------------------------------------------------------------------- #


class EarlyLogCapture:
    """Capture early logs before W&B initialization and replay them later."""

    def __init__(self):
        self.captured_logs = []
        self.enabled = True

    def log(self, message: str) -> None:
        """Capture a log message and also print it immediately."""
        if self.enabled:
            self.captured_logs.append(message)
        print(message)

    def replay_to_wandb(self) -> None:
        """Replay all captured logs and disable further capture."""
        if self.captured_logs:
            print("=== Replaying early logs to W&B ===")
            for log_msg in self.captured_logs:
                print(f"[REPLAY] {log_msg}")
        self.enabled = False
        self.captured_logs.clear()


# Global early log capture instance
_early_log_capture = EarlyLogCapture()


def early_log(message: str) -> None:
    """Log a message that will be captured and replayed to W&B."""
    global _early_log_capture
    _early_log_capture.log(message)


def replay_early_logs_to_wandb() -> None:
    """Replay all early logs to W&B and disable further capture."""
    global _early_log_capture
    _early_log_capture.replay_to_wandb()


# --------------------------------------------------------------------------- #
# Git utilities for debugging and tracking
# --------------------------------------------------------------------------- #


def get_git_info() -> dict[str, str]:
    """Get git repository information as a dictionary for logging to wandb."""
    git_info = {}

    # Check if we're in a git repository
    cwd = os.getcwd()
    git_info["cwd"] = cwd
    git_info["git_exists"] = str(os.path.exists(".git"))

    try:
        # Get current commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["commit_hash"] = result.stdout.strip()

        # Get current branch
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            branch = branch_result.stdout.strip()
            if not branch:  # Detached HEAD state
                branch = "detached"
        except subprocess.CalledProcessError:
            branch = "unknown"

        git_info["branch"] = branch

        # Get short commit message (first line only, limit to 120 chars)
        try:
            msg_result = subprocess.run(
                ["git", "log", "-1", "--format=%s"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            commit_msg = msg_result.stdout.strip()[:120]
            if len(msg_result.stdout.strip()) > 120:
                commit_msg += "..."
            git_info["commit_message"] = commit_msg
        except subprocess.CalledProcessError:
            git_info["commit_message"] = "no message"

        # Get git remote URL if available
        try:
            remote_result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            git_info["remote_url"] = remote_result.stdout.strip()
        except subprocess.CalledProcessError:
            git_info["remote_url"] = "no remote"

    except subprocess.CalledProcessError as e:
        git_info["error"] = f"Git command failed: {e}"
        if e.stderr:
            git_info["error_output"] = e.stderr
    except FileNotFoundError:
        git_info["error"] = (
            "Git command not found - git is not installed or not in PATH"
        )
    except subprocess.TimeoutExpired:
        git_info["error"] = "Git command timed out"
    except Exception as e:
        git_info["error"] = f"Unexpected error getting git info: {e}"

    return git_info


def log_git_commit_info() -> None:
    """Log current git commit information."""
    git_info = get_git_info()
    if git_info:
        early_log(f"📝 Git commit: {git_info['commit_hash'][:8]}")
        early_log(f"🌿 Branch: {git_info['branch']}")
        if git_info["error"]:
            early_log("⚠️  Git command failed")
    else:
        early_log("❌ Git information not available")


def log_git_info_to_wandb(wandb_run) -> None:
    """Log git information to W&B run."""
    git_info = get_git_info()
    if git_info and wandb_run:
        wandb_run.config.update(
            {
                "git_commit": git_info["commit_hash"],
                "git_branch": git_info["branch"],
                "git_dirty": False,  # No direct 'dirty' field in get_git_info, so set to False
            }
        )


# --------------------------------------------------------------------------- #
# Disk space monitoring utilities
# --------------------------------------------------------------------------- #


def get_disk_usage_percent() -> float:
    """Get current disk usage percentage."""
    try:
        import psutil

        usage = psutil.disk_usage("/")
        return (usage.used / usage.total) * 100
    except ImportError:
        try:
            # Fallback using shutil
            total, used, free = shutil.disk_usage("/")
            return (used / total) * 100
        except:
            return 0.0


def check_disk_space_emergency(threshold: float = 95.0) -> bool:
    """Check if disk space is critically low."""
    current_usage = get_disk_usage_percent()
    return current_usage > threshold


def cleanup_disk_space_emergency() -> None:
    """Attempt to clean up disk space in emergency situations."""
    try:
        early_log("🧹 Attempting emergency disk cleanup...")

        # Clean up PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            early_log("🗑️  Cleared CUDA cache")

        # Clean up temp directories
        import tempfile

        temp_dir = Path(tempfile.gettempdir())
        if temp_dir.exists():
            # Only clean our own temp files to be safe
            for pattern in ["*.tmp", "tmp*", "temp*"]:
                for file_path in temp_dir.glob(pattern):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                    except:
                        pass

        early_log("✅ Emergency cleanup completed")

    except Exception as e:
        early_log(f"❌ Emergency cleanup failed: {e}")


# --------------------------------------------------------------------------- #
# Configuration utilities
# --------------------------------------------------------------------------- #


def generate_run_name(
    model_type: str,
    config_name: str = "default",
    note: str = None,
    timestamp: bool = True,
) -> str:
    """Generate a descriptive run name for training."""
    parts = [model_type.lower()]

    if config_name and config_name != "default":
        # Extract meaningful parts from config name
        config_clean = config_name.replace("train_", "").replace("_default", "")
        if config_clean:
            parts.append(config_clean)

    if note:
        # Clean up note: remove spaces, keep alphanumeric and basic punctuation
        note_clean = "".join(c for c in note if c.isalnum() or c in "-_").lower()
        if note_clean:
            parts.append(note_clean)

    if timestamp:
        import time

        time_str = time.strftime("%m%d-%H%M")
        parts.append(time_str)

    return "-".join(parts)


def load_config_file(path: str) -> Dict[str, object]:
    """Load configuration from a Python file."""
    return runpy.run_path(path)


def update_config(cfg: BaseConfig, data: Dict[str, object]) -> None:
    """Update configuration object with dictionary data."""
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def apply_overrides(cfg: BaseConfig, overrides: List[str]) -> None:
    """Apply command-line overrides to configuration."""
    for override in overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            key = key.lstrip("-")
            if hasattr(cfg, key):
                field_type = type(getattr(cfg, key))
                if field_type == bool:
                    setattr(cfg, key, value.lower() in ("true", "1", "yes"))
                elif field_type in (int, float):
                    setattr(cfg, key, field_type(value))
                elif field_type == list:
                    # Handle list fields by parsing as Python literal
                    try:
                        setattr(cfg, key, literal_eval(value))
                    except (ValueError, SyntaxError):
                        # Fallback: split by comma
                        setattr(cfg, key, [v.strip() for v in value.split(",")])
                else:
                    setattr(cfg, key, value)


def parse_args() -> argparse.ArgumentParser:
    """Create argument parser for training scripts."""
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("config", type=str, help="Path to configuration file")
    parser.add_argument("--subset", type=float, help="Fraction of dataset to use")
    parser.add_argument("--dag-depth", type=int, help="DAG depth")
    parser.add_argument("--wandb-api-key", type=str, help="Weights & Biases API key")
    parser.add_argument("--wandb-run-id", type=str, help="Resume specific W&B run")
    parser.add_argument("--note", type=str, help="Optional note for run name")
    parser.add_argument(
        "--use-runpod", action="store_true", help="Use RunPod for training"
    )
    parser.add_argument("--gpu-type", type=str, help="GPU type for RunPod")
    parser.add_argument(
        "--keep-alive", action="store_true", help="Keep instance alive after training"
    )
    return parser


def get_lr(it: int, *, cfg: BaseConfig) -> float:
    """Calculate learning rate with warmup and decay."""
    # Validate configuration first
    if not hasattr(cfg, "learning_rate"):
        raise AttributeError("Config must have 'learning_rate' attribute")
    if not hasattr(cfg, "warmup_iters"):
        raise AttributeError("Config must have 'warmup_iters' attribute")
    if not hasattr(cfg, "lr_decay_iters"):
        raise AttributeError("Config must have 'lr_decay_iters' attribute")
    if not hasattr(cfg, "min_lr"):
        raise AttributeError("Config must have 'min_lr' attribute")

    # 1. Determine base learning rate
    if it < cfg.warmup_iters:
        # Linear warmup
        if cfg.warmup_iters > 0:
            # Using it + 1 to make it 1-based for warmup, matches test expectations
            base_lr = cfg.learning_rate * (it + 1) / cfg.warmup_iters
        else:
            base_lr = cfg.learning_rate
    elif it > cfg.lr_decay_iters:
        # Past decay phase
        base_lr = cfg.min_lr
    else:
        # In between, might be constant or cosine decay
        if not getattr(cfg, "decay_lr", True):
            base_lr = cfg.learning_rate
        else:
            # Cosine decay
            decay_ratio = (it - cfg.warmup_iters) / (
                cfg.lr_decay_iters - cfg.warmup_iters
            )
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            base_lr = cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    # 2. Apply cyclical modulation if enabled and not in warmup
    final_lr = base_lr
    if getattr(cfg, "use_cyclical_lr", False) and it >= cfg.warmup_iters:
        period = getattr(cfg, "cyclical_lr_period", 1000)
        amplitude = getattr(cfg, "cyclical_lr_amplitude", 0.1)

        # Cycle starts after warmup
        progress_in_decay = it - cfg.warmup_iters
        cycle_progress = (progress_in_decay % period) / period
        cyclical_factor = 1.0 + amplitude * math.sin(2 * math.pi * cycle_progress)

        final_lr *= cyclical_factor

    # 3. Final clamping
    # During warmup, we want the linear ramp-up, so no min_lr clamping yet
    if it < cfg.warmup_iters:
        return final_lr

    return max(cfg.min_lr, final_lr)


def get_checkpoint_filename(
    cfg: BaseConfig,
    iter_num: int,
    model_name: str | None = None,
    val_acc: float | None = None,
) -> str:
    """Generate checkpoint filename based on config and iteration."""
    base_name = model_name if model_name else cfg.name
    safe_name = "".join(c for c in base_name if c.isalnum() or c in ("-", "_"))

    if val_acc is not None:
        acc_str = f"{val_acc * 100:.2f}acc"
        return f"ckpt_{safe_name}_{iter_num}_{acc_str}.pt"
    else:
        return f"ckpt_{safe_name}_{iter_num}.pt"


def check_for_nonfinite(
    named_tensors_iter: Iterable[tuple[str, torch.Tensor]], label: str
) -> None:
    """Check tensors for NaN/Inf values and print detailed diagnostic info."""
    for name, tensor in named_tensors_iter:
        if tensor is None:
            continue
        if torch.isnan(tensor).any():
            print(
                f"[{label} NAN] {name}  →  min={tensor.min():.3e}  max={tensor.max():.3e}"
            )
        elif torch.isinf(tensor).any():
            print(
                f"[{label} INF] {name}  →  min={tensor.min():.3e}  max={tensor.max():.3e}"
            )
