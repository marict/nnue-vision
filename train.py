#!/usr/bin/env python3
"""Training Script

Train NNUE or EtinyNet models.

Usage Examples:
    # Train NNUE with default config
    python train.py nnue

    # Train EtinyNet with default config
    python train.py etinynet

    # Train with custom config
    python train.py nnue --config config/train_nnue_custom.py

    # Train EtinyNet with specific variant and dataset
    python train.py etinynet --etinynet_variant 1.0 --dataset_name cifar100

    # Override config parameters
    python train.py nnue --batch_size 64 --max_epochs 100 --learning_rate 0.001
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
from config import ConfigError, load_config
from data import create_data_loaders

# Create EtinyNet model
from nnue import NNUE, EtinyNet, GridFeatureSet
from nnue_runpod_service import stop_runpod
from training_utils import (
    early_log,
    replay_early_logs_to_wandb,
)


class CheckpointManager:
    """Simple checkpoint management for training."""

    def __init__(self, log_dir: str, run_name: str):
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
        # Update best tracking
        self.best_metric = metrics.get("val_f1", 0)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config_name": getattr(config, "name", "unknown"),
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

        # Sanitize run name for artifact naming (only alphanumeric, dashes, underscores, dots)
        sanitized_run_name = re.sub(r"[^a-zA-Z0-9._-]", "_", self.run_name)
        artifact_name = f"{sanitized_run_name}-best"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="best_model",
            metadata={
                "epoch": epoch,
                "metrics": metrics,
                "config_name": getattr(config, "name", "unknown"),
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
    ):
        """Load checkpoint and restore model/optimizer state."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint["epoch"], checkpoint["metrics"]


def compute_nnue_loss(model, batch):
    """Compute NNUE loss for training."""
    images, targets = batch

    # Forward pass
    logits = model(images)

    # For computer vision classification, always use CrossEntropyLoss
    loss = F.cross_entropy(logits, targets.long())

    return loss


def compute_metrics(outputs, targets, num_classes=1):
    """Compute training/validation metrics."""
    if num_classes > 1:
        # Multi-class classification metrics
        _, predicted = torch.max(outputs, 1)
        targets_int = targets.long()

        # Convert to numpy for sklearn
        pred_np = predicted.cpu().numpy()
        target_np = targets_int.cpu().numpy()

        accuracy = accuracy_score(target_np, pred_np)
        f1 = f1_score(target_np, pred_np, average="weighted", zero_division=0)
        precision = precision_score(
            target_np, pred_np, average="weighted", zero_division=0
        )
        recall = recall_score(target_np, pred_np, average="weighted", zero_division=0)

        return {"acc": accuracy, "f1": f1, "precision": precision, "recall": recall}
    else:
        # Single output regression/binary classification
        # For NNUE, just compute MSE-style accuracy
        predicted = torch.sigmoid(outputs)
        targets_binary = (targets > 0.5).float()
        accuracy = (predicted.round() == targets_binary).float().mean().item()

        return {
            "acc": accuracy,
            "f1": accuracy,  # Use accuracy as f1 for simplicity
            "precision": accuracy,
            "recall": accuracy,
        }


def train_model(
    config: Any,
    model_type: str,  # "nnue" or "etinynet"
    wandb_run_id: Optional[str] = None,
) -> int:
    """Unified training function for both NNUE and EtinyNet models"""
    # Common setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_log(f"🚀 Using device: {device}")

    wandb_config = {k: v for k, v in config.__dict__.items() if not k.startswith("__")}
    # Initialize WandB
    wandb.init(
        project=config.project_name,
        config=wandb_config,
        id=wandb_run_id,
        resume="allow",
    )
    early_log(f"📤 W&B run URL: {wandb.run.url}")
    replay_early_logs_to_wandb()

    checkpoint_manager = CheckpointManager(config.log_dir, wandb.run.name)

    # Common data loading
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_augmentation=config.use_augmentation,
        augmentation_strength=config.augmentation_strength,
        subset=config.subset,
        target_size=getattr(config, "target_size", None),
        binary_classification=getattr(config, "binary_classification", None),
    )

    # Model-specific initialization
    if model_type == "nnue":
        # NNUE-specific setup
        feature_set = GridFeatureSet(
            grid_size=getattr(config, "grid_size", 10),
            num_features_per_square=getattr(config, "num_features_per_square", 8),
        )
        model = NNUE(
            feature_set=feature_set,
            l1_size=getattr(config, "l1_size", 1024),
            l2_size=getattr(config, "l2_size", 15),
            l3_size=getattr(config, "l3_size", 32),
            num_classes=config.num_classes,
            input_size=config.input_size,
            weight_decay=config.weight_decay,
        )
        loss_fn = compute_nnue_loss
    elif model_type == "etinynet":
        # EtinyNet-specific setup
        model = EtinyNet(
            variant=config.etinynet_variant,
            num_classes=config.num_classes,
            input_size=config.input_size,
            weight_decay=config.weight_decay,
        )

        def etiny_loss_fn(model, batch):
            images, targets = batch
            logits = model(images)
            return torch.nn.functional.cross_entropy(logits, targets.long())

        loss_fn = etiny_loss_fn
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    early_log(
        f"🧠 Model type: {model_type}, Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Common optimizer setup
    optimizer = create_optimizer(model, config)

    # Training loop (common for both models)
    best_val_f1 = 0.0
    for epoch in range(config.max_epochs):
        # Training phase
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            batch = (images, labels)

            optimizer.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()

            if hasattr(config, "max_grad_norm") and config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()

            early_log(
                f"Epoch {epoch+1}/{config.max_epochs}, "
                f"Batch {batch_idx+1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )
            wandb.log(
                {"train/loss": loss.item()}, step=epoch * len(train_loader) + batch_idx
            )

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = evaluate_model(model, val_loader, loss_fn, device)

        early_log(
            f"Epoch {epoch+1}/{config.max_epochs} - "
            f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, "
            f"Val Acc: {val_metrics['acc']:.4f}"
        )
        wandb.log(
            {
                "val/loss": val_loss,
                "val/f1": val_metrics["f1"],
                "val/accuracy": val_metrics["acc"],
            },
            step=(epoch + 1) * len(train_loader) - 1,  # Log at the end of the epoch
        )

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            checkpoint_manager.save_best_model_to_wandb(
                model,
                optimizer,
                epoch,
                {"val_f1": val_metrics["f1"], "val_loss": val_loss},
                config,
            )

    # Final test evaluation (common)
    test_loss, test_metrics = evaluate_model(model, test_loader, loss_fn, device)
    wandb.log({"test/f1": test_metrics["f1"], "test/loss": test_loss})

    # Cleanup
    if not config.keep_alive:
        stop_runpod()

    return 0


# Helper functions
def create_optimizer(model, config):
    """Create optimizer based on config"""
    if config.optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:  # Default to Adam
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )


def evaluate_model(model, loader, loss_fn, device=None):
    """Common evaluation logic"""
    total_loss = 0
    all_outputs = []
    all_targets = []

    for batch in loader:
        # Move batch to device
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        batch = (images, labels)

        loss = loss_fn(model, batch)
        total_loss += loss.item()
        outputs = model(batch[0])
        all_outputs.append(outputs.cpu())
        all_targets.append(batch[1].cpu())

    outputs = torch.cat(all_outputs)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(outputs, targets, model.num_classes)
    return total_loss / len(loader), metrics


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for training."""
    parser = argparse.ArgumentParser(description="Train NNUE and EtinyNet models")

    # Model type
    parser.add_argument(
        "model_type", choices=["nnue", "etinynet"], help="Model type to train"
    )

    # Common arguments
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    parser.add_argument(
        "--batch_size", type=int, help="Override batch size from config"
    )
    parser.add_argument(
        "--max_epochs", type=int, help="Override max epochs from config"
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Override learning rate from config"
    )
    parser.add_argument("--note", type=str, help="Note to add to run name and config")
    parser.add_argument("--wandb_api_key", type=str, help="Wandb API key")
    parser.add_argument("--wandb-run-id", type=str, help="Resume specific W&B run")
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory for logs and checkpoints"
    )
    parser.add_argument(
        "--use_augmentation",
        type=lambda x: x.lower() == "true",
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--augmentation_strength",
        choices=["light", "medium", "heavy"],
        help="Data augmentation strength",
    )

    # Model-specific arguments
    parser.add_argument(
        "--etinynet_variant",
        type=str,
        choices=["0.75", "1.0", "0.98M"],
        help="EtinyNet variant",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["cifar10", "cifar100"],
        help="Dataset to use",
    )
    parser.add_argument("--num_classes", type=int, help="Number of classes (for NNUE)")

    return parser


def load_and_setup_config(args: argparse.Namespace, model_type: str) -> Any:
    """Load configuration and apply command-line overrides."""
    # Set default config path based on model type
    if args.config is None:
        args.config = f"config/train_{model_type}_default.py"

    try:
        early_log(f"⚙️  Loading configuration from: {args.config}")
        config = load_config(args.config)
        early_log(f"✅ Configuration loaded: {config.name}")
    except ConfigError as e:
        early_log(f"❌ Error loading configuration: {e}")
        raise

    # Apply common overrides
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.note is not None:
        config.note = args.note
    if args.use_augmentation is not None:
        config.use_augmentation = args.use_augmentation
    if args.augmentation_strength is not None:
        config.augmentation_strength = args.augmentation_strength
    if args.log_dir:
        config.log_dir = args.log_dir

    # Apply model-specific overrides
    if model_type == "etinynet":
        if args.etinynet_variant is not None:
            config.etinynet_variant = args.etinynet_variant
        if args.dataset_name is not None:
            config.dataset_name = args.dataset_name
    elif model_type == "nnue":
        if args.num_classes is not None:
            config.num_classes = args.num_classes

        if args.dataset_name is not None:
            config.dataset_name = args.dataset_name

    return config


def main():
    """Main entry point for training."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup wandb API key
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    if not os.getenv("WANDB_API_KEY"):
        early_log("❌ Error: WANDB_API_KEY not found. WandB logging is required.")
        early_log("💡 Set the environment variable or use --wandb_api_key argument.")
        return 1

    # Load configuration
    config = load_and_setup_config(args, args.model_type)

    # Run training based on model type
    return train_model(
        config, args.model_type, wandb_run_id=getattr(args, "wandb_run_id", None)
    )


if __name__ == "__main__":
    sys.exit(main())
