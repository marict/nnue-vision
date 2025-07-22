#!/usr/bin/env python3
"""
Minimal NNUE-Vision Training Script for Visual Wake Words Dataset

This script demonstrates how to train a simple CNN on the Visual Wake Words dataset.
It's been stripped down from the original chess NNUE code to focus on computer vision.

Usage:
    python train_minimal.py [--config CONFIG_PATH]

The script will:
1. Load configuration from a Python config file
2. Download the Visual Wake Words dataset automatically
3. Train a simple CNN for person/no-person classification
4. Save the trained model and logs
"""

import argparse
from pathlib import Path

import pytorch_lightning as pl

from config import ConfigError, load_config
from dataset import create_data_loaders
from model import ModelParams, SimpleCNN


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Minimal NNUE-Vision training with config support"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_minimal.py",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        print(f"Configuration loaded: {config.name}")
    except ConfigError as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Set random seed for reproducibility
    pl.seed_everything(getattr(config, "seed", 42))

    # Model configuration from config file
    model_params = ModelParams(
        input_size=getattr(config, "input_size", (96, 96)),
        num_classes=getattr(config, "num_classes", 2),
        learning_rate=getattr(config, "learning_rate", 1e-3),
    )

    # Create model
    print("Creating model...")
    model = SimpleCNN(model_params)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create data loaders (this will download the dataset if needed)
    print(f"\nLoading {getattr(config, 'dataset', 'Visual Wake Words')} dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=getattr(config, "batch_size", 32),
        num_workers=getattr(config, "num_workers", 2),
        target_size=getattr(config, "input_size", (96, 96)),
        subset=getattr(config, "subset", 1.0),
    )

    # Set up trainer with configuration
    trainer = pl.Trainer(
        max_epochs=getattr(config, "max_epochs", 10),
        accelerator=getattr(config, "accelerator", "auto"),
        devices=getattr(config, "devices", "auto"),
        log_every_n_steps=getattr(config, "log_interval", 50),
        enable_checkpointing=getattr(config, "always_save_checkpoint", True),
        enable_progress_bar=getattr(config, "enable_progress_bar", True),
        deterministic=getattr(config, "deterministic", True),
        check_val_every_n_epoch=getattr(config, "check_val_every_n_epoch", 1),
    )

    # Train the model
    print(f"\nStarting training...")
    print(f"Configuration: {config.name}")
    print(f"Training on {len(train_loader.dataset)} samples")
    print(f"Validating on {len(val_loader.dataset)} samples")

    trainer.fit(model, train_loader, val_loader)

    # Test the model
    print("\nTesting the final model...")
    test_results = trainer.test(model, test_loader)

    print(f"\nTraining completed!")
    print(f"Final test accuracy: {test_results[0]['test_acc']:.3f}")

    # Save the model
    import torch

    model_filename = f"{getattr(config, 'name', 'visual_wake_words_model')}.pt"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as '{model_filename}'")


if __name__ == "__main__":
    print("=" * 60)
    print("NNUE-Vision: Visual Wake Words Training")
    print("=" * 60)
    print("This is a minimal version of NNUE adapted for computer vision.")
    print("Original NNUE was designed for chess, this version trains on images.")
    print()

    main()

    print("\n" + "=" * 60)
    print("Training completed! You can now use the trained model for inference.")
    print("Check the logs/ directory for TensorBoard logs.")
    print("=" * 60)
