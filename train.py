#!/usr/bin/env python3
"""Training Script for NNUE and EtinyNet models."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

import wandb
from checkpoint_manager import CheckpointManager
from config import ConfigError, load_config
from data import create_data_loaders
from evaluate import compute_metrics, evaluate_compiled_model, evaluate_model
from nnue import NNUE, EtinyNet, GridFeatureSet
from nnue_runpod_service import stop_runpod
from training_utils import (
    early_log,
    replay_early_logs_to_wandb,
)


def compile_cpp_engine(model_type: str) -> bool:
    """Compile C++ engine for the specified model type."""
    early_log(f"🔨 Compiling C++ engine for {model_type}...")

    engine_dir = Path("engine")
    build_dir = engine_dir / "build"

    # Create build directory if it doesn't exist
    build_dir.mkdir(exist_ok=True)

    try:
        # Run cmake
        early_log("  Running cmake...")
        cmake_result = subprocess.run(
            ["cmake", ".."],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if cmake_result.returncode != 0:
            early_log(f"❌ CMake failed: {cmake_result.stderr}")
            return False

        # Run make
        early_log("  Running make...")
        make_result = subprocess.run(
            ["make", "-j4"],
            cwd=build_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if make_result.returncode != 0:
            early_log(f"❌ Make failed: {make_result.stderr}")
            return False

        # Check if the expected executables were built
        if model_type == "nnue":
            executable = build_dir / "regression_test"
        elif model_type == "etinynet":
            executable = build_dir / "etinynet_inference"
        else:
            early_log(f"❌ Unknown model type: {model_type}")
            return False

        if executable.exists():
            early_log(f"✅ C++ engine compiled successfully: {executable}")
            return True
        else:
            early_log(f"❌ Expected executable not found: {executable}")
            return False

    except subprocess.TimeoutExpired:
        early_log("❌ Compilation timed out")
        return False
    except Exception as e:
        early_log(f"❌ Compilation failed: {e}")
        return False


def compute_loss(model, batch):
    """Compute loss for both NNUE and EtinyNet models."""
    images, targets = batch
    logits = model(images)
    return F.cross_entropy(logits, targets.long())


def train_model(
    config: Any,
    model_type: str,
    wandb_run_id: Optional[str] = None,
) -> int:
    """Unified training function for both NNUE and EtinyNet models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    early_log(f"🚀 Using device: {device}")

    wandb_config = {k: v for k, v in config.__dict__.items() if not k.startswith("__")}
    wandb.init(
        project=config.project_name,
        config=wandb_config,
        id=wandb_run_id,
        resume="allow",
    )
    early_log(f"📤 W&B run URL: {wandb.run.url}")
    replay_early_logs_to_wandb()

    checkpoint_manager = CheckpointManager(config.log_dir, wandb.run.name)

    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_augmentation=config.use_augmentation,
        augmentation_strength=config.augmentation_strength,
        subset=config.subset,
    )

    if model_type == "nnue":
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

    optimizer = create_optimizer(model, config)

    # Compile C++ engine early to catch issues
    early_log("🔨 Pre-compiling C++ engine to catch issues early...")
    if not compile_cpp_engine(model_type):
        early_log("❌ C++ engine compilation failed! Training will fail later.")
        early_log("   This is a critical error - please fix compilation issues.")
        return 1
    early_log("✅ C++ engine ready for compiled evaluation")

    best_val_f1 = 0.0
    for epoch in range(config.max_epochs):
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(train_loader):
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

        model.eval()
        with torch.no_grad():
            train_loss, train_metrics = evaluate_model(
                model, train_loader, loss_fn, device
            )
        model.train()

        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = evaluate_model(model, val_loader, loss_fn, device)

            early_log(f"🔧 Evaluating compiled model performance...")
            try:
                compiled_metrics = evaluate_compiled_model(
                    model, val_loader, model_type
                )
            except Exception as e:
                early_log(f"❌ Compiled model evaluation failed: {e}")
                raise RuntimeError(
                    f"Compiled model evaluation is required but failed: {e}"
                )

        log_data = {
            "train/epoch_loss": train_loss,
            "train/epoch_f1": train_metrics["f1"],
            "train/epoch_accuracy": train_metrics["acc"],
            "val/loss": val_loss,
            "val/f1": val_metrics["f1"],
            "val/accuracy": val_metrics["acc"],
        }

        if compiled_metrics:
            log_data.update(
                {
                    "compiled/f1": compiled_metrics["f1"],
                    "compiled/accuracy": compiled_metrics["acc"],
                    "compiled/ms_per_sample": compiled_metrics.get(
                        "ms_per_sample", 0.0
                    ),
                }
            )
            early_log(
                f"Epoch {epoch+1}/{config.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}, Train Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, Val Acc: {val_metrics['acc']:.4f} | "
                f"Compiled F1: {compiled_metrics['f1']:.4f}, Compiled Acc: {compiled_metrics['acc']:.4f}, Speed: {compiled_metrics.get('ms_per_sample', 0.0):.2f}ms/sample"
            )
        else:
            early_log(
                f"Epoch {epoch+1}/{config.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}, Train Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1']:.4f}, Val Acc: {val_metrics['acc']:.4f}"
            )

        wandb.log(log_data, step=(epoch + 1) * len(train_loader) - 1)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            checkpoint_manager.save_best_model_to_wandb(
                model,
                optimizer,
                epoch,
                {"val_f1": val_metrics["f1"], "val_loss": val_loss},
                config,
            )

    test_loss, test_metrics = evaluate_model(model, test_loader, loss_fn, device)
    wandb.log({"test/f1": test_metrics["f1"], "test/loss": test_loss})

    if not config.keep_alive:
        stop_runpod()

    return 0


def create_optimizer(model, config):
    """Create optimizer based on config"""
    if config.optimizer_type == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up the argument parser for training."""
    parser = argparse.ArgumentParser(description="Train NNUE and EtinyNet models")

    parser.add_argument(
        "model_type", choices=["nnue", "etinynet"], help="Model type to train"
    )

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
    if args.config is None:
        args.config = f"config/train_{model_type}_default.py"

    try:
        early_log(f"⚙️  Loading configuration from: {args.config}")
        config = load_config(args.config)
        early_log(f"✅ Configuration loaded: {config.name}")
    except ConfigError as e:
        early_log(f"❌ Error loading configuration: {e}")
        raise

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

    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    if not os.getenv("WANDB_API_KEY"):
        early_log("❌ Error: WANDB_API_KEY not found. WandB logging is required.")
        early_log("💡 Set the environment variable or use --wandb_api_key argument.")
        return 1

    config = load_and_setup_config(args, args.model_type)

    return train_model(config, args.model_type, wandb_run_id=args.wandb_run_id)


if __name__ == "__main__":
    sys.exit(main())
