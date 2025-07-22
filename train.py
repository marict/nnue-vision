import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         TQDMProgressBar)

from dataset import create_data_loaders
from model import ModelParams, SimpleCNN


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN on Visual Wake Words dataset"
    )

    # Dataset arguments
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--image_size", type=int, default=96, help="Input image size (square)"
    )

    # Model arguments
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )

    # Training arguments
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--gpus", type=str, default=None, help="GPUs to use (e.g., '0,1' or 'auto')"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator to use ('cpu', 'gpu', 'auto')",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory for logs and checkpoints"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="visual_wake_words",
        help="Project name for logging",
    )
    parser.add_argument(
        "--save_top_k", type=int, default=3, help="Number of best models to save"
    )

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )

    args = parser.parse_args()

    # Set up model parameters
    model_params = ModelParams(
        input_size=(args.image_size, args.image_size),
        num_classes=2,  # person/no person
        learning_rate=args.learning_rate,
    )

    # Create model
    model = SimpleCNN(model_params)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(args.image_size, args.image_size),
    )

    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir, name=args.project_name)

    # Set up callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=50),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=args.save_top_k,
            filename="best-{epoch:02d}-{val_acc:.3f}",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_acc", mode="max", patience=args.patience, verbose=True
        ),
    ]

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.gpus.split(",") if args.gpus else "auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=True,
    )

    # Train the model
    print(f"Starting training for {args.max_epochs} epochs...")
    print(f"Model parameters: {model_params}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")

    trainer.fit(model, train_loader, val_loader)

    # Test the model
    print("Testing the model...")
    trainer.test(model, test_loader)

    # Save final model
    final_model_path = os.path.join(args.log_dir, args.project_name, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    print("Training completed!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    pl.seed_everything(42)

    main()
