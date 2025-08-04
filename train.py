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
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
from config import ConfigError, load_config
from data import create_data_loaders
from nnue_runpod_service import stop_runpod
from training_utils import (
    check_disk_space_emergency,
    cleanup_disk_space_emergency,
    early_log,
    get_disk_usage_percent,
    log_git_commit_info,
    replay_early_logs_to_wandb,
)


class CheckpointManager:
    """Simple checkpoint management for training."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = None
        self.best_metric = None

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Any,
        is_best: bool = False,
    ) -> str:
        """Save checkpoint with model, optimizer, and metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config_name": getattr(config, "name", "unknown"),
        }

        # Create filename
        if is_best:
            filename = f"best-f1-{epoch:02d}-{metrics.get('val_f1', 0):.3f}.ckpt"
            self.best_checkpoint_path = self.log_dir / filename
            self.best_metric = metrics.get("val_f1", 0)
        else:
            filename = f"checkpoint-epoch-{epoch:02d}.ckpt"

        checkpoint_path = self.log_dir / filename
        torch.save(checkpoint, checkpoint_path)
        early_log(f"💾 Saved checkpoint: {checkpoint_path}")

        return str(checkpoint_path)

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


def adapt_batch_for_nnue(batch):
    """Adapt batch from dataset format to NNUE format."""
    images, labels = batch
    device = images.device

    # Convert labels to targets (keep as integers for CrossEntropyLoss)
    targets = labels.to(device)

    return images, targets


def compute_nnue_loss(model, batch):
    """Compute NNUE loss for training."""
    model._clip_weights()  # NNUE weight clipping

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


def train_nnue(config: Any, wandb_run_id: Optional[str] = None) -> int:
    """NNUE training loop."""

    # Early logging and system info
    early_log(f"🚀 Starting NNUE training...")
    early_log(f"📊 Disk usage: {get_disk_usage_percent():.1f}%")

    # Check for disk space emergencies early
    if check_disk_space_emergency():
        early_log("⚠️  Disk space is critically low!")
        cleaned_mb = cleanup_disk_space_emergency()
        early_log(f"🧹 Emergency cleanup freed {cleaned_mb:.1f} MB")

    # Log git information
    early_log("📝 Git repository information:")
    log_git_commit_info()

    try:
        # Set random seed for reproducibility
        torch.manual_seed(getattr(config, "seed", 42))
        np.random.seed(getattr(config, "seed", 42))

        # Create NNUE model
        from model import NNUE, GridFeatureSet, LossParams

        # Set up loss parameters from config
        loss_params = LossParams(
            start_lambda=getattr(config, "start_lambda", 1.0),
            end_lambda=getattr(config, "end_lambda", 1.0),
        )

        early_log("🏗️  Creating NNUE model...")

        # Create feature set
        feature_set = GridFeatureSet(
            grid_size=getattr(config, "grid_size", 10),
            num_features_per_square=getattr(config, "num_features_per_square", 8),
        )

        model = NNUE(
            feature_set=feature_set,
            l1_size=getattr(config, "l1_size", 1024),
            l2_size=getattr(config, "l2_size", 15),
            l3_size=getattr(config, "l3_size", 32),
            loss_params=loss_params,
            visual_threshold=getattr(config, "visual_threshold", 0.0),
            num_classes=getattr(config, "num_classes", 1),
        )
        early_log("✅ NNUE model created successfully")

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        early_log(f"📱 Using device: {device}")

        # Create data loaders
        early_log("📚 Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_name=getattr(config, "dataset_name", "cifar10"),
            batch_size=getattr(config, "batch_size", 32),
            use_augmentation=getattr(config, "use_augmentation", True),
            augmentation_strength=getattr(config, "augmentation_strength", "medium"),
            max_samples_per_split=getattr(config, "max_samples_per_split", None),
            subset=getattr(config, "subset", 1.0),
            binary_classification=getattr(config, "binary_classification", None),
        )
        early_log("✅ Data loaders created successfully")

        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=getattr(config, "learning_rate", 1e-3),
            weight_decay=getattr(config, "weight_decay", 5e-4),
        )

        # Setup wandb logger
        log_dir = getattr(config, "log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create wandb config with only serializable values
        wandb_config = {
            "learning_rate": getattr(config, "learning_rate", 1e-3),
            "batch_size": getattr(config, "batch_size", 32),
            "max_epochs": getattr(config, "max_epochs", 50),
            "model_type": "nnue",
            "num_classes": getattr(config, "num_classes", 1),
            "dataset_name": getattr(config, "dataset_name", "cifar10"),
        }

        wandb_kwargs = {
            "project": getattr(config, "project_name", "nnue_training"),
            "config": wandb_config,
            "dir": log_dir,
        }

        if wandb_run_id:
            early_log(f"🔄 Resuming W&B run: {wandb_run_id}")
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "allow"
        else:
            run_name = f"nnue-lr{config.learning_rate}-bs{config.batch_size}"
            if hasattr(config, "note") and config.note:
                run_name += f"-{config.note}"
            wandb_kwargs["name"] = run_name

        wandb.init(**wandb_kwargs)
        early_log(f"📤 W&B run URL: {wandb.run.url}")

        # Replay early logs to wandb
        replay_early_logs_to_wandb()

        # Setup checkpoint manager
        checkpoint_manager = CheckpointManager(log_dir)

        # Training loop
        max_epochs = getattr(config, "max_epochs", 50)
        log_interval = getattr(config, "log_interval", 50)
        best_val_f1 = 0.0

        early_log("🚀 Starting training loop...")

        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_metrics = {"acc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
            num_batches = 0
            train_batch_times = []

            for batch_idx, batch in enumerate(train_loader):
                batch_start_time = time.time()

                # Move batch to device
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                # Adapt batch for NNUE format
                nnue_batch = adapt_batch_for_nnue((images, labels))

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass and loss computation
                loss = compute_nnue_loss(model, nnue_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()

                # Track batch timing
                batch_time = time.time() - batch_start_time
                train_batch_times.append(batch_time)

                # Compute training metrics periodically
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        outputs = model(nnue_batch[0], nnue_batch[3])
                        batch_metrics = compute_metrics(
                            outputs, nnue_batch[1], model.num_classes
                        )
                        for key in train_metrics:
                            train_metrics[key] += batch_metrics[key]
                        num_batches += 1

                        # Log to wandb
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/acc": batch_metrics["acc"],
                                "train/f1": batch_metrics["f1"],
                                "epoch": epoch,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            commit=True,
                        )

                        if batch_idx % (log_interval * 10) == 0:
                            early_log(
                                f"Epoch {epoch}, Batch {batch_idx}: loss={loss.item():.4f}, acc={batch_metrics['acc']:.4f}, time={batch_time:.3f}s"
                            )

            # Average training metrics
            if num_batches > 0:
                for key in train_metrics:
                    train_metrics[key] /= num_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_targets = []
            val_batch_times = []

            with torch.no_grad():
                for batch in val_loader:
                    val_batch_start_time = time.time()

                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)

                    # Adapt batch for NNUE format
                    nnue_batch = adapt_batch_for_nnue((images, labels))

                    # Forward pass
                    loss = compute_nnue_loss(model, nnue_batch)
                    outputs = model(nnue_batch[0])

                    val_loss += loss.item()
                    val_outputs.append(outputs.cpu())
                    val_targets.append(nnue_batch[1].cpu())

                    # Track validation batch timing
                    val_batch_time = time.time() - val_batch_start_time
                    val_batch_times.append(val_batch_time)

            # Compute validation metrics
            val_outputs = torch.cat(val_outputs)
            val_targets = torch.cat(val_targets)
            val_metrics = compute_metrics(val_outputs, val_targets, model.num_classes)
            val_loss /= len(val_loader)

            # Calculate timing statistics
            avg_train_batch_time = (
                np.mean(train_batch_times) if train_batch_times else 0.0
            )
            avg_val_batch_time = np.mean(val_batch_times) if val_batch_times else 0.0

            # Log epoch results
            epoch_log = {
                "epoch": epoch,
                "train/epoch_loss": train_loss / len(train_loader),
                "train/epoch_acc": train_metrics["acc"],
                "train/epoch_f1": train_metrics["f1"],
                "train/avg_batch_time": avg_train_batch_time,
                "val/loss": val_loss,
                "val/f1": val_metrics["f1"],
                "val/avg_batch_time": avg_val_batch_time,
            }

            # Debug: Print what we're logging to wandb
            early_log(f"📊 Logging to wandb: val/f1={val_metrics['f1']:.4f}")
            wandb.log(epoch_log, commit=True)

            early_log(
                f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, "
                f"val_loss={val_loss:.4f}, val_f1={val_metrics['f1']:.4f}, "
                f"train_time={avg_train_batch_time:.3f}s/batch, val_time={avg_val_batch_time:.3f}s/batch"
            )

            # Save checkpoint if validation F1 improved
            is_best = val_metrics["f1"] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics["f1"]
                early_log(f"🎯 NEW BEST validation F1: {best_val_f1:.4f}")

            checkpoint_manager.save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_f1": val_metrics["f1"], "val_loss": val_loss},
                config,
                is_best=is_best,
            )

        # Test evaluation
        early_log("🧪 Running final test evaluation...")
        model.eval()
        test_loss = 0.0
        test_outputs = []
        test_targets = []

        with torch.no_grad():
            for batch in test_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                nnue_batch = adapt_batch_for_nnue((images, labels))

                loss = compute_nnue_loss(model, nnue_batch)
                outputs = model(nnue_batch[0])

                test_loss += loss.item()
                test_outputs.append(outputs.cpu())
                test_targets.append(nnue_batch[1].cpu())

        test_outputs = torch.cat(test_outputs)
        test_targets = torch.cat(test_targets)
        test_metrics = compute_metrics(test_outputs, test_targets, model.num_classes)
        test_loss /= len(test_loader)

        # Log final test results
        final_results = {
            "final/test_loss": test_loss,
            "final/test_acc": test_metrics["acc"],
            "final/test_f1": test_metrics["f1"],
            "final/test_precision": test_metrics["precision"],
            "final/test_recall": test_metrics["recall"],
            "final/disk_usage_percent": get_disk_usage_percent(),
            "final/training_completed": True,
        }

        wandb.log(final_results, commit=True)

        # Save final model as artifact
        if checkpoint_manager.best_checkpoint_path:
            artifact = wandb.Artifact("best_f1_model", type="model")
            artifact.add_file(str(checkpoint_manager.best_checkpoint_path))
            wandb.log_artifact(artifact)
            early_log(
                f"📦 Best model saved as artifact: {checkpoint_manager.best_checkpoint_path}"
            )

        early_log("✅ NNUE training completed successfully!")
        early_log(f"🎯 Best validation F1: {best_val_f1:.4f}")
        early_log(f"📊 Final test F1: {test_metrics['f1']:.4f}")

        wandb.finish()
        # Stop RunPod instance if keep_alive is False
        if not getattr(config, "keep_alive", False):
            stop_runpod()

        wandb.finish()
        return 0

    except Exception as e:
        error_msg = f"Fatal error in NNUE training: {str(e)}"
        early_log(f"❌ {error_msg}")
        traceback.print_exc()

        # Log error to wandb
        try:
            error_data = {
                "error/message": error_msg,
                "error/type": type(e).__name__,
                "error/traceback": traceback.format_exc(),
                "error/timestamp": time.time(),
            }
            wandb.log(error_data, commit=True)
        except Exception as wandb_error:
            early_log(f"⚠️  Failed to log error to wandb: {wandb_error}")

        wandb.finish()
        if not getattr(config, "keep_alive", False):
            stop_runpod()
        wandb.finish()


def train_etinynet(config: Any, wandb_run_id: Optional[str] = None) -> int:
    """EtinyNet training loop."""

    # Early logging and system info
    early_log(f"🚀 Starting EtinyNet training...")
    early_log(f"📊 Disk usage: {get_disk_usage_percent():.1f}%")

    # Check for disk space emergencies early
    if check_disk_space_emergency():
        early_log("⚠️  Disk space is critically low!")
        cleaned_mb = cleanup_disk_space_emergency()
        early_log(f"🧹 Emergency cleanup freed {cleaned_mb:.1f} MB")

    # Log git information
    early_log("📝 Git repository information:")
    log_git_commit_info()

    try:
        # Set random seed for reproducibility
        torch.manual_seed(getattr(config, "seed", 42))
        np.random.seed(getattr(config, "seed", 42))

        # Create EtinyNet model
        from model import EtinyNet

        variant = getattr(config, "etinynet_variant", "0.75")
        dataset_name = getattr(config, "dataset_name", "cifar10")

        # Determine number of classes based on dataset
        if dataset_name == "cifar10":
            num_classes = 10
        elif dataset_name == "cifar100":
            num_classes = 100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        early_log("🏗️  Creating EtinyNet model...")
        model = EtinyNet(
            variant=variant,
            num_classes=num_classes,
            input_size=32,  # CIFAR images are 32x32
        )
        early_log("✅ EtinyNet model created successfully")

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        early_log(f"📱 Using device: {device}")

        # Create data loaders
        early_log("📚 Creating data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_name=dataset_name,
            batch_size=getattr(config, "batch_size", 128),
            use_augmentation=getattr(config, "use_augmentation", True),
            augmentation_strength=getattr(config, "augmentation_strength", "medium"),
            max_samples_per_split=getattr(config, "max_samples_per_split", None),
            subset=getattr(config, "subset", 1.0),
            binary_classification=getattr(config, "binary_classification", None),
        )
        early_log("✅ Data loaders created successfully")

        # Setup optimizer (SGD for EtinyNet as per paper)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=getattr(config, "learning_rate", 0.1),
            momentum=0.9,
            weight_decay=getattr(config, "weight_decay", 1e-4),
        )

        # Setup learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=getattr(config, "max_epochs", 300)
        )

        # Setup wandb logger
        log_dir = getattr(config, "log_dir", "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Create wandb config with only serializable values
        wandb_config = {
            "learning_rate": getattr(config, "learning_rate", 0.1),
            "batch_size": getattr(config, "batch_size", 128),
            "max_epochs": getattr(config, "max_epochs", 300),
            "model_type": "etinynet",
            "etinynet_variant": variant,
            "dataset_name": dataset_name,
            "num_classes": num_classes,
        }

        wandb_kwargs = {
            "project": getattr(config, "project_name", "etinynet_training"),
            "config": wandb_config,
            "dir": log_dir,
        }

        if wandb_run_id:
            early_log(f"🔄 Resuming W&B run: {wandb_run_id}")
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "allow"
        else:
            run_name = f"etinynet-lr{config.learning_rate}-bs{config.batch_size}"
            if hasattr(config, "note") and config.note:
                run_name += f"-{config.note}"
            wandb_kwargs["name"] = run_name

        wandb.init(**wandb_kwargs)
        early_log(f"📤 W&B run URL: {wandb.run.url}")

        # Replay early logs to wandb
        replay_early_logs_to_wandb()

        # Setup checkpoint manager
        checkpoint_manager = CheckpointManager(log_dir)

        # Training loop
        max_epochs = getattr(config, "max_epochs", 300)
        log_interval = getattr(config, "log_interval", 50)
        best_val_f1 = 0.0

        early_log("🚀 Starting training loop...")

        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_metrics = {"acc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
            num_batches = 0
            train_batch_times = []

            for batch_idx, batch in enumerate(train_loader):
                batch_start_time = time.time()

                # Move batch to device
                images, targets = batch
                images, targets = images.to(device), targets.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = F.cross_entropy(outputs, targets)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()

                # Track batch timing
                batch_time = time.time() - batch_start_time
                train_batch_times.append(batch_time)

                # Compute training metrics periodically
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        batch_metrics = compute_metrics(
                            outputs, targets.float(), num_classes
                        )
                        for key in train_metrics:
                            train_metrics[key] += batch_metrics[key]
                        num_batches += 1

                        # Log to wandb
                        wandb.log(
                            {
                                "train/loss": loss.item(),
                                "train/acc": batch_metrics["acc"],
                                "train/f1": batch_metrics["f1"],
                                "epoch": epoch,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            commit=True,
                        )

                        if batch_idx % (log_interval * 10) == 0:
                            early_log(
                                f"Epoch {epoch}, Batch {batch_idx}: loss={loss.item():.4f}, acc={batch_metrics['acc']:.4f}, time={batch_time:.3f}s"
                            )

            # Update learning rate
            scheduler.step()

            # Average training metrics
            if num_batches > 0:
                for key in train_metrics:
                    train_metrics[key] /= num_batches

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_outputs = []
            val_targets = []
            val_batch_times = []

            with torch.no_grad():
                for batch in val_loader:
                    val_batch_start_time = time.time()

                    images, targets = batch
                    images, targets = images.to(device), targets.to(device)

                    outputs = model(images)
                    loss = F.cross_entropy(outputs, targets)

                    val_loss += loss.item()
                    val_outputs.append(outputs.cpu())
                    val_targets.append(targets.cpu().float())

                    # Track validation batch timing
                    val_batch_time = time.time() - val_batch_start_time
                    val_batch_times.append(val_batch_time)

            # Compute validation metrics
            val_outputs = torch.cat(val_outputs)
            val_targets = torch.cat(val_targets)
            val_metrics = compute_metrics(val_outputs, val_targets, num_classes)
            val_loss /= len(val_loader)

            # Calculate timing statistics
            avg_train_batch_time = (
                np.mean(train_batch_times) if train_batch_times else 0.0
            )
            avg_val_batch_time = np.mean(val_batch_times) if val_batch_times else 0.0

            # Log epoch results
            epoch_log = {
                "epoch": epoch,
                "train/epoch_loss": train_loss / len(train_loader),
                "train/epoch_acc": train_metrics["acc"],
                "train/epoch_f1": train_metrics["f1"],
                "train/avg_batch_time": avg_train_batch_time,
                "val/loss": val_loss,
                "val/f1": val_metrics["f1"],
                "val/avg_batch_time": avg_val_batch_time,
            }

            # Debug: Print what we're logging to wandb
            early_log(f"📊 Logging to wandb: val/f1={val_metrics['f1']:.4f}")
            wandb.log(epoch_log, commit=True)

            early_log(
                f"Epoch {epoch}: train_loss={train_loss/len(train_loader):.4f}, "
                f"val_loss={val_loss:.4f}, val_f1={val_metrics['f1']:.4f}, "
                f"train_time={avg_train_batch_time:.3f}s/batch, val_time={avg_val_batch_time:.3f}s/batch"
            )

            # Save checkpoint if validation F1 improved
            is_best = val_metrics["f1"] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics["f1"]
                early_log(f"🎯 NEW BEST validation F1: {best_val_f1:.4f}")

            checkpoint_manager.save_checkpoint(
                model,
                optimizer,
                epoch,
                {"val_f1": val_metrics["f1"], "val_loss": val_loss},
                config,
                is_best=is_best,
            )

        # Test evaluation
        early_log("🧪 Running final test evaluation...")
        model.eval()
        test_loss = 0.0
        test_outputs = []
        test_targets = []

        with torch.no_grad():
            for batch in test_loader:
                images, targets = batch
                images, targets = images.to(device), targets.to(device)

                outputs = model(images)
                loss = F.cross_entropy(outputs, targets)

                test_loss += loss.item()
                test_outputs.append(outputs.cpu())
                test_targets.append(targets.cpu().float())

        test_outputs = torch.cat(test_outputs)
        test_targets = torch.cat(test_targets)
        test_metrics = compute_metrics(test_outputs, test_targets, num_classes)
        test_loss /= len(test_loader)

        # Log final test results
        final_results = {
            "final/test_loss": test_loss,
            "final/test_acc": test_metrics["acc"],
            "final/test_f1": test_metrics["f1"],
            "final/test_precision": test_metrics["precision"],
            "final/test_recall": test_metrics["recall"],
            "final/disk_usage_percent": get_disk_usage_percent(),
            "final/training_completed": True,
        }

        wandb.log(final_results, commit=True)

        # Save final model as artifact
        if checkpoint_manager.best_checkpoint_path:
            artifact = wandb.Artifact("best_f1_model", type="model")
            artifact.add_file(str(checkpoint_manager.best_checkpoint_path))
            wandb.log_artifact(artifact)
            early_log(
                f"📦 Best model saved as artifact: {checkpoint_manager.best_checkpoint_path}"
            )

        early_log("✅ EtinyNet training completed successfully!")
        early_log(f"🎯 Best validation F1: {best_val_f1:.4f}")
        early_log(f"📊 Final test F1: {test_metrics['f1']:.4f}")

        wandb.finish()
        # Stop RunPod instance if keep_alive is False
        if not getattr(config, "keep_alive", False):
            stop_runpod()
        wandb.finish()

    except Exception as e:
        error_msg = f"Fatal error in EtinyNet training: {str(e)}"
        early_log(f"❌ {error_msg}")
        traceback.print_exc()

        # Log error to wandb
        try:
            error_data = {
                "error/message": error_msg,
                "error/type": type(e).__name__,
                "error/traceback": traceback.format_exc(),
                "error/timestamp": time.time(),
            }
            wandb.log(error_data, commit=True)
        except Exception as wandb_error:
            early_log(f"⚠️  Failed to log error to wandb: {wandb_error}")

        wandb.finish()
        if not getattr(config, "keep_alive", False):
            stop_runpod()
        wandb.finish()


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
    if args.model_type == "nnue":
        return train_nnue(config, wandb_run_id=getattr(args, "wandb_run_id", None))
    elif args.model_type == "etinynet":
        return train_etinynet(config, wandb_run_id=getattr(args, "wandb_run_id", None))
    else:
        early_log(f"❌ Unknown model type: {args.model_type}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
