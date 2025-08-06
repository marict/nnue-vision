#!/usr/bin/env python3
"""Checkpoint Management Module

Provides checkpoint management functionality for model training, including
WandB integration for artifact storage and model state persistence.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

import wandb
from training_utils import early_log


class CheckpointManager:
    """Simple checkpoint management for training."""

    def __init__(self, log_dir: str, run_name: str):
        """Initialize checkpoint manager."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = None
        self.best_metric = None
        self.run_name = run_name

    def save_best_model_to_wandb(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Any,
    ) -> None:
        """Save only the best model directly to wandb (no local storage)."""
        print(
            f"Saving best model to wandb (epoch {epoch}, F1: {metrics.get('val_f1', 0):.3f})..."
        )
        self.best_metric = metrics.get("val_f1", 0)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config_name": config.name,
        }

        with tempfile.NamedTemporaryFile(
            suffix=f"-best-f1-{epoch:02d}-{metrics.get('val_f1', 0):.3f}.ckpt",
            delete=False,
        ) as tmp_file:
            torch.save(checkpoint, tmp_file.name)
            tmp_path = tmp_file.name

        early_log(
            f"📤 Uploading BEST model to wandb (epoch {epoch}, F1: {metrics.get('val_f1', 0):.3f})..."
        )

        sanitized_run_name = re.sub(r"[^a-zA-Z0-9._-]", "_", self.run_name)
        artifact_name = f"{sanitized_run_name}-best"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="best_model",
            metadata={
                "epoch": epoch,
                "metrics": metrics,
                "config_name": config.name,
                "run_name": self.run_name,
            },
        )
        artifact.add_file(tmp_path)
        wandb.log_artifact(artifact)
        early_log(f"✅ Best model uploaded to wandb as {artifact_name}")

        os.unlink(tmp_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[int, Dict[str, float]]:
        """Load checkpoint and restore model/optimizer state."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["metrics"]

    def save_checkpoint_local(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Any,
        filename: Optional[str] = None,
    ) -> str:
        """Save checkpoint to local storage."""
        if filename is None:
            filename = (
                f"checkpoint-epoch-{epoch:03d}-f1-{metrics.get('val_f1', 0):.3f}.ckpt"
            )

        checkpoint_path = self.log_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config_name": config.name,
        }

        torch.save(checkpoint, checkpoint_path)
        early_log(f"💾 Checkpoint saved to {checkpoint_path}")

        return str(checkpoint_path)

    def get_best_metric(self) -> Optional[float]:
        """Get the best metric value seen so far."""
        return self.best_metric

    def should_save_checkpoint(self, current_metric: float) -> bool:
        """Check if current metric is better than the best seen so far."""
        if self.best_metric is None:
            return True
        return current_metric > self.best_metric
