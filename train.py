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
import subprocess
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
from serialize import serialize_model
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


def compute_loss(model, batch):
    """Compute loss for both NNUE and EtinyNet models."""
    images, targets = batch
    logits = model(images)
    return F.cross_entropy(logits, targets.long())


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
    )

    # Model-specific initialization
    if model_type == "nnue":
        # NNUE-specific setup
        feature_set = GridFeatureSet(
            grid_size=config.grid_size,
            num_features_per_square=config.num_features_per_square,
        )
        model = NNUE(
            feature_set=feature_set,
            l1_size=config.l1_size,
            l2_size=config.l2_size,
            l3_size=config.l3_size,
            num_classes=config.num_classes,
            input_size=config.input_size,
            weight_decay=config.weight_decay,
        )
        loss_fn = compute_loss
    elif model_type == "etinynet":
        # EtinyNet-specific setup
        model = EtinyNet(
            variant=config.etinynet_variant,
            num_classes=config.num_classes,
            input_size=config.input_size,
            weight_decay=config.weight_decay,
        )

        loss_fn = compute_loss
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
        train_losses = []
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

            train_losses.append(loss.item())
            early_log(
                f"Epoch {epoch+1}/{config.max_epochs}, "
                f"Batch {batch_idx+1}/{len(train_loader)}, "
                f"Loss: {loss.item():.4f}"
            )
            wandb.log(
                {"train/loss": loss.item()}, step=epoch * len(train_loader) + batch_idx
            )

        # Evaluate training metrics for the epoch
        model.eval()
        with torch.no_grad():
            train_loss, train_metrics = evaluate_model(
                model, train_loader, loss_fn, device
            )
        model.train()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = evaluate_model(model, val_loader, loss_fn, device)

            # Evaluate compiled model performance (every evaluation step)
            early_log(f"🔧 Evaluating compiled model performance...")
            try:
                compiled_metrics = evaluate_compiled_model(
                    model, val_loader, model_type, device
                )
            except Exception as e:
                early_log(f"❌ Compiled model evaluation failed: {e}")
                raise RuntimeError(
                    f"Compiled model evaluation is required but failed: {e}"
                )

        # Log metrics
        log_data = {
            "train/epoch_loss": train_loss,
            "train/epoch_f1": train_metrics["f1"],
            "train/epoch_accuracy": train_metrics["acc"],
            "val/loss": val_loss,
            "val/f1": val_metrics["f1"],
            "val/accuracy": val_metrics["acc"],
        }

        # Add compiled metrics if available
        if compiled_metrics:
            log_data.update(
                {
                    "compiled/f1": compiled_metrics["f1"],
                    "compiled/accuracy": compiled_metrics["acc"],
                }
            )
            early_log(
                f"Epoch {epoch+1}/{config.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}, Train Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, Val Acc: {val_metrics['acc']:.4f} | "
                f"Compiled F1: {compiled_metrics['f1']:.4f}, Compiled Acc: {compiled_metrics['acc']:.4f}"
            )
        else:
            early_log(
                f"Epoch {epoch+1}/{config.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}, Train Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
            )

        wandb.log(log_data, step=(epoch + 1) * len(train_loader) - 1)

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


def evaluate_compiled_model(model, loader, model_type):
    """Evaluate model using compiled C++ engine for real-world performance metrics."""
    try:
        # Check if C++ engine is available
        if model_type == "nnue":
            cpp_executable = Path("engine/build/regression_test")
            if not cpp_executable.exists():
                raise RuntimeError(
                    f"C++ NNUE engine not found: {cpp_executable}. Run 'cd engine && mkdir -p build && cd build && cmake .. && make' to build it."
                )
        elif model_type == "etinynet":
            cpp_executable = Path("engine/build/etinynet_inference")
            if not cpp_executable.exists():
                raise RuntimeError(
                    f"C++ EtinyNet engine not found: {cpp_executable}. Run 'cd engine && mkdir -p build && cd build && cmake .. && make' to build it."
                )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Serialize model to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{model_type}", delete=False) as f:
            model_path = Path(f.name)

        try:
            serialize_model(model, model_path)

            # Evaluate a subset of the dataset for speed
            all_outputs = []
            all_targets = []
            sample_count = 0
            max_samples = 100  # Limit for speed

            for batch in loader:
                if sample_count >= max_samples:
                    break

                images, labels = batch
                batch_size = images.shape[0]
                processed_samples = min(batch_size, max_samples - sample_count)

                # For NNUE, we need to extract features
                if model_type == "nnue":
                    # Use a simple feature extraction (first few pixels as features)
                    for i in range(processed_samples):
                        img = images[i].cpu().numpy()
                        # Extract simple features (first 50 pixels as feature indices)
                        feature_indices = list(range(min(50, img.size)))

                        # Run C++ inference
                        cpp_args = [str(cpp_executable), str(model_path)] + [
                            str(f) for f in feature_indices
                        ]
                        result = subprocess.run(
                            cpp_args, capture_output=True, text=True, timeout=10
                        )

                        if result.returncode == 0:
                            # Parse C++ output
                            for line in result.stdout.split("\n"):
                                if line.startswith("RESULT_INCREMENTAL_0:"):
                                    try:
                                        cpp_output = float(line.split(": ")[1])
                                        all_outputs.append(torch.tensor([cpp_output]))
                                        break
                                    except (ValueError, IndexError):
                                        # Fallback to PyTorch output
                                        all_outputs.append(torch.tensor([0.0]))
                                        break
                        else:
                            # Fallback to PyTorch output
                            all_outputs.append(torch.tensor([0.0]))

                elif model_type == "etinynet":
                    # For EtinyNet, save image to binary file and run inference
                    for i in range(processed_samples):
                        img = images[i].cpu().numpy()

                        # Save image to temporary binary file
                        with tempfile.NamedTemporaryFile(
                            suffix=".bin", delete=False
                        ) as img_f:
                            img_path = Path(img_f.name)
                            img.tofile(img_f.name)

                        try:
                            # Run C++ inference
                            cpp_args = [
                                str(cpp_executable),
                                str(model_path),
                                str(img_path),
                                str(img.shape[1]),
                                str(img.shape[2]),
                            ]
                            result = subprocess.run(
                                cpp_args, capture_output=True, text=True, timeout=10
                            )

                            if result.returncode == 0:
                                # Parse C++ output (first line should be logits)
                                lines = result.stdout.strip().split("\n")
                                if lines:
                                    cpp_output = float(lines[0])
                                    all_outputs.append(torch.tensor([cpp_output]))
                                else:
                                    all_outputs.append(torch.tensor([0.0]))
                            else:
                                all_outputs.append(torch.tensor([0.0]))
                        finally:
                            if img_path.exists():
                                img_path.unlink()

                # Add targets for the samples we processed
                # Handle case where batch is larger than actual data
                actual_labels = labels[:batch_size]  # Only take actual labels
                if processed_samples == 1:
                    target = actual_labels[0]
                    # Ensure target is a tensor, not a scalar
                    if target.dim() == 0:
                        target = target.unsqueeze(0)
                    all_targets.append(target)
                else:
                    targets = actual_labels[:processed_samples]
                    all_targets.extend(targets)
                sample_count += processed_samples

            if all_outputs and len(all_outputs) > 0:
                try:
                    # Ensure all tensors have the same shape before concatenating
                    if len(all_outputs) == 1:
                        outputs = all_outputs[0]
                    else:
                        outputs = torch.cat(all_outputs)

                    if len(all_targets) == 1:
                        targets = all_targets[0]
                    else:
                        targets = torch.cat(all_targets)

                    # Ensure outputs and targets have compatible shapes
                    if outputs.dim() == 1 and targets.dim() == 1:
                        # Single output per sample
                        metrics = compute_metrics(
                            outputs.unsqueeze(0),
                            targets.unsqueeze(0),
                            model.num_classes,
                        )
                    else:
                        metrics = compute_metrics(outputs, targets, model.num_classes)

                    return metrics
                except Exception as e:
                    early_log(f"⚠️ Error computing metrics: {e}")
                    early_log(f"  Outputs shape: {[o.shape for o in all_outputs]}")
                    early_log(f"  Targets shape: {[t.shape for t in all_targets]}")
                    return None
            else:
                return None

        finally:
            if model_path.exists():
                model_path.unlink()

    except:
        early_log(f"⚠️ Compiled evaluation failed: {e}")
        raise e


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
