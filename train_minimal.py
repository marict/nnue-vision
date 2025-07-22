#!/usr/bin/env python3
"""
Minimal NNUE-Vision Training Script for Visual Wake Words Dataset

This script demonstrates how to train a simple CNN on the Visual Wake Words dataset.
It's been stripped down from the original chess NNUE code to focus on computer vision.

Usage:
    python train_minimal.py

The script will:
1. Download the Visual Wake Words dataset automatically
2. Train a simple CNN for person/no-person classification
3. Save the trained model and logs
"""

import pytorch_lightning as pl

from dataset import create_data_loaders
from model import ModelParams, SimpleCNN


def main():
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Model configuration
    model_params = ModelParams(
        input_size=(96, 96),  # Standard size for Visual Wake Words
        num_classes=2,  # person/no person
        learning_rate=1e-3,
    )

    # Create model
    print("Creating model...")
    model = SimpleCNN(model_params)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create data loaders (this will download the dataset if needed)
    print("\nLoading Visual Wake Words dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=32, num_workers=2, target_size=(96, 96)  # Reduced for stability
    )

    # Set up trainer with minimal configuration
    trainer = pl.Trainer(
        max_epochs=10,  # Small number for quick demo
        accelerator="auto",  # Use GPU if available, CPU otherwise
        devices="auto",  # Use all available devices
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=True,
    )

    # Train the model
    print(f"\nStarting training...")
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

    torch.save(model.state_dict(), "visual_wake_words_model.pt")
    print("Model saved as 'visual_wake_words_model.pt'")


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
