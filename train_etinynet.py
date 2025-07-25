#!/usr/bin/env python3
"""
EtinyNet Training Script

Train EtinyNet models on standard computer vision datasets (CIFAR-10, CIFAR-100).
This script provides the same configuration system and logging as the main NNUE trainer
but optimized for standard CNN training workflows.

Usage:
    # Quick test
    python train_etinynet.py --config config/train_test.py

    # Local development
    python train_etinynet.py --config config/train_default.py

    # Custom parameters
    python train_etinynet.py --config config/train_default.py --max_epochs 100 --batch_size 64

Example configs for EtinyNet:
    batch_size = 64        # Larger batches work well for EtinyNet
    max_epochs = 200       # More epochs for better convergence
    learning_rate = 0.1    # Higher LR with SGD+momentum
"""

import argparse
import os
import traceback
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

import runpod_service
import wandb
from config import ConfigError, load_config
from data import create_data_loaders
from model import EtinyNet
from training_utils import (
    check_disk_space_emergency,
    cleanup_disk_space_emergency,
    early_log,
    get_disk_usage_percent,
    log_git_commit_info,
    log_git_info_to_wandb,
    replay_early_logs_to_wandb,
)


def main():
    # Early logging and system info
    early_log("🚀 Starting EtinyNet training...")
    early_log(f"📊 Disk usage: {get_disk_usage_percent():.1f}%")

    # Check for disk space emergencies early
    if check_disk_space_emergency():
        early_log("⚠️  Disk space is critically low!")
        cleaned_mb = cleanup_disk_space_emergency()
        early_log(f"🧹 Emergency cleanup freed {cleaned_mb:.1f} MB")

    # Log git information for debugging and tracking
    early_log("📝 Git repository information:")
    log_git_commit_info()

    parser = argparse.ArgumentParser(
        description="Train EtinyNet on computer vision datasets with config file support"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_etinynet_default.py",
        help="Path to the configuration file",
    )

    # Override options (optional - can override config file values)
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size from config"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Override max epochs from config"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["0.75", "1.0"],
        default=None,
        help="EtinyNet variant (0.75 or 1.0)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar10", "cifar100", "imagenet"],
        default="cifar10",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--note", type=str, default=None, help="Note to add to run name and config"
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="Wandb API key (or set WANDB_API_KEY env var)",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Resume specific W&B run",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for logs and checkpoints",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        early_log(f"⚙️  Loading configuration from: {args.config}")
        config = load_config(args.config)
        early_log(f"✅ Configuration loaded: {config.name}")
    except ConfigError as e:
        early_log(f"❌ Error loading configuration: {e}")
        return 1

    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.note is not None:
        config.note = args.note

    # EtinyNet-specific parameters
    variant = args.variant or getattr(config, "etinynet_variant", "0.75")
    dataset_name = args.dataset

    # Setup wandb API key
    wandb_api_key = args.wandb_api_key or getattr(config, "wandb_api_key", None)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    if not os.getenv("WANDB_API_KEY"):
        early_log("❌ Error: WANDB_API_KEY not found. WandB logging is required.")
        early_log("💡 Set the environment variable or use --wandb_api_key argument.")
        raise ValueError("WANDB_API_KEY is required for training")

    # Set random seed for reproducibility
    pl.seed_everything(getattr(config, "seed", 42))

    # Determine number of classes based on dataset
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create EtinyNet model
    model = EtinyNet(
        variant=variant,
        num_classes=num_classes,
        input_size=32,  # CIFAR datasets are 32x32
        use_asq=getattr(config, "use_asq", False),
        asq_bits=getattr(config, "asq_bits", 4),
        lr=getattr(config, "learning_rate", 0.1),
        max_epochs=getattr(config, "max_epochs", 200),
    )

    # Create data loaders for standard datasets (not NNUE-specific)
    early_log("📚 Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=dataset_name,
        batch_size=getattr(config, "batch_size", 64),
        num_workers=getattr(config, "num_workers", 4),
        max_samples_per_split=(
            None
            if getattr(config, "subset", 1.0) >= 1.0
            else int(50000 * getattr(config, "subset", 1.0))
        ),
    )

    # Set up logging
    log_dir = args.log_dir or getattr(config, "log_dir", "logs")
    loggers = []
    callbacks = []

    # Setup run name
    run_name = f"etinynet-{variant}_{dataset_name}"
    if hasattr(config, "note") and config.note:
        run_name += f"_{config.note}"

    # Wandb logger (always required)
    early_log("🔗 Setting up Wandb logging...")
    project_name = getattr(config, "project_name", f"etinynet_{dataset_name}")

    # Handle W&B run resumption if run ID is provided
    wandb_kwargs = {
        "project": project_name,
        "name": run_name,
        "save_dir": log_dir,
    }
    if getattr(args, "wandb_run_id", None):
        early_log(f"🔄 Resuming W&B run: {args.wandb_run_id}")
        wandb_kwargs["id"] = args.wandb_run_id
        wandb_kwargs["resume"] = "must"

    wandb_logger = WandbLogger(**wandb_kwargs)
    loggers = [wandb_logger]

    # Log configuration to wandb
    if hasattr(wandb_logger, "experiment"):
        wandb_logger.experiment.config.update(
            {
                "model_type": "EtinyNet",
                "variant": variant,
                "dataset": dataset_name,
                "num_classes": num_classes,
                "parameters": model.count_parameters(),
                "config_file": args.config,
                **{k: v for k, v in vars(config).items() if not k.startswith("_")},
            }
        )

    # Log git info and replay early logs
    log_git_info_to_wandb(wandb_logger.experiment)
    replay_early_logs_to_wandb(wandb_logger.experiment)

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename=f"etinynet-{variant}-" + "{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=getattr(config, "save_top_k", 3),
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=getattr(config, "patience", 15),
        verbose=True,
    )
    callbacks.append(early_stopping)

    # Progress bar
    if getattr(config, "enable_progress_bar", True):
        callbacks.append(TQDMProgressBar(refresh_rate=20))

    # Device setup
    devices = getattr(config, "devices", "auto")
    if devices == "auto":
        devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=getattr(config, "max_epochs", 200),
        accelerator=getattr(config, "accelerator", "auto"),
        devices=devices,
        logger=loggers if loggers else False,
        callbacks=callbacks,
        log_every_n_steps=getattr(config, "log_interval", 50),
        enable_checkpointing=getattr(config, "always_save_checkpoint", True),
        enable_progress_bar=getattr(config, "enable_progress_bar", True),
        deterministic=getattr(config, "deterministic", True),
        check_val_every_n_epoch=getattr(config, "check_val_every_n_epoch", 1),
    )

    # Log training start
    print(
        f"Starting EtinyNet training for {getattr(config, 'max_epochs', 200)} epochs..."
    )
    print(f"Configuration: {config.name}")
    print(f"Model: EtinyNet-{variant} ({model.count_parameters():,} parameters)")
    print(f"Dataset: {dataset_name.upper()} ({num_classes} classes)")
    print(f"Learning rate: {getattr(config, 'learning_rate', 0.1)}")
    print(f"Batch size: {getattr(config, 'batch_size', 64)}")
    # Display wandb information (always available)
    wandb_logger = next(
        (logger for logger in loggers if isinstance(logger, WandbLogger)), None
    )
    if wandb_logger:
        print(
            f"Wandb project: {getattr(config, 'project_name', f'etinynet_{dataset_name}')}"
        )
        print(f"Wandb run URL: {wandb_logger.experiment.url}")

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    print("Testing the model...")
    test_results = trainer.test(model, test_loader)

    # Log final test results to wandb (always required)
    if test_results:
        wandb.log(
            {
                "final/test_loss": test_results[0]["test_loss"],
                "final/test_acc": test_results[0]["test_acc"],
            }
        )

    # Save final model for serialization
    final_model_path = os.path.join(
        log_dir, f"etinynet_{variant}_{dataset_name}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    early_log(f"💾 Saved final model to: {final_model_path}")

    # Finish wandb run (always required)
    wandb.finish()

    # Stop RunPod instance if we're running on RunPod and keep-alive is not enabled
    if os.getenv("RUNPOD_POD_ID") and not getattr(config, "keep_alive", False):
        try:
            runpod_service.stop_runpod()
        except ImportError:
            pass  # runpod_service not available in this environment

    early_log("🎉 Training completed successfully!")
    print(f"\nTo serialize this model for C++ engine:")
    print(
        f"python serialize.py {final_model_path} etinynet_{variant}_{dataset_name}.etiny"
    )

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"Fatal error in EtinyNet training: {e}")
        traceback.print_exc()

        # Stop RunPod instance on error if we're running on RunPod
        if os.getenv("RUNPOD_POD_ID"):
            try:
                runpod_service.stop_runpod()
            except ImportError:
                pass  # runpod_service not available in this environment

        # Re-raise the exception to ensure proper exit code
        raise
