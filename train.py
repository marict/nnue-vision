import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         TQDMProgressBar)
from pytorch_lightning.loggers import WandbLogger

from config import ConfigError, load_config
from dataset import create_data_loaders
from model import ModelParams, SimpleCNN


class WandbMetricsCallback(pl.Callback):
    """Custom callback for comprehensive wandb logging."""

    def __init__(self):
        super().__init__()
        self.train_start_time = None
        self.epoch_start_time = None
        self.step_start_time = None

    def on_train_start(self, trainer, pl_module):
        """Log initial setup information."""
        self.train_start_time = time.time()

        # Log model architecture details
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )

        wandb.log(
            {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/model_size_mb": total_params
                * 4
                / (1024 * 1024),  # Assuming float32
            }
        )

        # Log system information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (
                    1024**3
                )
                wandb.log(
                    {
                        f"system/gpu_{i}_name": gpu_name,
                        f"system/gpu_{i}_memory_gb": gpu_memory,
                    }
                )

    def on_train_epoch_start(self, trainer, pl_module):
        """Log epoch start time."""
        self.epoch_start_time = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Log batch start time and learning rate."""
        self.step_start_time = time.time()

        # Log learning rate from optimizer
        if trainer.optimizers:
            current_lr = trainer.optimizers[0].param_groups[0]["lr"]
            wandb.log(
                {
                    "train/learning_rate": current_lr,
                    "train/global_step": trainer.global_step,
                }
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log detailed training metrics after each batch."""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            wandb.log(
                {
                    "timing/step_time_ms": step_time * 1000,
                    "timing/steps_per_second": 1.0 / step_time,
                }
            )

        # Log gradient information
        grad_norm = 0.0
        param_norm = 0.0
        grad_max = 0.0

        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_norm += param.data.norm(2).item() ** 2
                grad_norm += param.grad.data.norm(2).item() ** 2
                grad_max = max(grad_max, param.grad.data.abs().max().item())

        grad_norm = grad_norm**0.5
        param_norm = param_norm**0.5

        wandb.log(
            {
                "gradients/grad_norm": grad_norm,
                "gradients/param_norm": param_norm,
                "gradients/grad_max": grad_max,
                "gradients/grad_param_ratio": grad_norm / (param_norm + 1e-8),
            }
        )

        # Log memory usage if CUDA is available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            wandb.log(
                {
                    "system/gpu_memory_allocated_gb": memory_allocated,
                    "system/gpu_memory_reserved_gb": memory_reserved,
                }
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch timing and training progress."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            wandb.log(
                {
                    "timing/epoch_time_minutes": epoch_time / 60,
                    "timing/total_training_hours": (time.time() - self.train_start_time)
                    / 3600,
                }
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation results and model predictions."""
        # Log sample predictions if we're in validation
        if hasattr(pl_module, "log_sample_predictions"):
            pl_module.log_sample_predictions()


def log_sample_predictions(model, val_loader, device, num_samples=8):
    """Log sample predictions to wandb."""
    model.eval()

    # Get a batch of validation data
    val_batch = next(iter(val_loader))
    images, labels = val_batch
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]

    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

    # Create wandb images with predictions
    class_names = ["No Person", "Person"]
    wandb_images = []

    for i in range(len(images)):
        img = images[i].cpu()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        confidence = probs[i][pred_label].item()

        # Convert tensor to wandb image format
        # Note: images are normalized, so we need to denormalize for visualization
        wandb_img = wandb.Image(
            img,
            caption=f"True: {class_names[true_label]}, Pred: {class_names[pred_label]} ({confidence:.3f})",
        )
        wandb_images.append(wandb_img)

    wandb.log({"validation/sample_predictions": wandb_images})


def setup_wandb_logger(config, model_params) -> WandbLogger:
    """Set up wandb logger with comprehensive configuration."""

    # Generate run name with timestamp and key parameters
    run_name = (
        f"cnn-lr{config.learning_rate}-bs{config.batch_size}-img{config.image_size}"
    )
    if hasattr(config, "note") and config.note:
        run_name += f"-{config.note}"

    # Create wandb config
    wandb_config = {
        # Model parameters
        "model/learning_rate": model_params.learning_rate,
        "model/input_size": model_params.input_size,
        "model/num_classes": model_params.num_classes,
        # Training parameters
        "train/batch_size": config.batch_size,
        "train/max_epochs": config.max_epochs,
        "train/image_size": config.image_size,
        "train/num_workers": config.num_workers,
        # System parameters
        "system/cuda_available": torch.cuda.is_available(),
        "system/torch_version": torch.__version__,
        "system/device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
        # Training configuration
        "config/accelerator": config.accelerator,
        "config/patience": config.patience,
        "config/save_top_k": config.save_top_k,
        "config/name": config.name,
    }

    if hasattr(config, "note") and config.note:
        wandb_config["experiment/note"] = config.note

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=run_name,
        config=wandb_config,
        save_dir=config.log_dir,
        log_model=True,  # Save model checkpoints to wandb
    )

    return wandb_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN on Visual Wake Words dataset with config file support"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_default.py",
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
        "--note", type=str, default=None, help="Note to add to run name and config"
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="Wandb API key (or set WANDB_API_KEY env var)",
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

    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.note is not None:
        config.note = args.note

    # Setup wandb API key
    wandb_api_key = args.wandb_api_key or getattr(config, "wandb_api_key", None)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key

    use_wandb = getattr(config, "use_wandb", True)
    if use_wandb and not os.getenv("WANDB_API_KEY"):
        print("Warning: WANDB_API_KEY not found. Wandb logging may not work properly.")
        print("Set the environment variable or use --wandb_api_key argument.")

    # Set random seed for reproducibility
    pl.seed_everything(getattr(config, "seed", 42))

    # Set up model parameters from config
    model_params = ModelParams(
        input_size=getattr(config, "input_size", (96, 96)),
        num_classes=getattr(config, "num_classes", 2),
        learning_rate=getattr(config, "learning_rate", 1e-3),
    )

    # Create model
    model = SimpleCNN(model_params)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=getattr(config, "batch_size", 32),
        num_workers=getattr(config, "num_workers", 4),
        target_size=getattr(config, "input_size", (96, 96)),
        subset=getattr(config, "subset", 1.0),
    )

    # Set up logging
    log_dir = getattr(config, "log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    loggers = []

    # Setup wandb logger if enabled
    if use_wandb:
        wandb_logger = setup_wandb_logger(config, model_params)
        loggers.append(wandb_logger)

    # Optional TensorBoard logger
    use_tensorboard = getattr(config, "use_tensorboard", False)
    if use_tensorboard:
        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir=log_dir, name=getattr(config, "project_name", "visual_wake_words")
        )
        loggers.append(tb_logger)

    # Set up callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=getattr(config, "log_interval", 50)),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=getattr(config, "save_top_k", 3),
            filename="best-{epoch:02d}-{val_acc:.3f}",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=getattr(config, "patience", 10),
            verbose=True,
        ),
    ]

    # Add wandb metrics callback only if using wandb
    if use_wandb:
        callbacks.insert(1, WandbMetricsCallback())

    # Set up trainer
    devices = getattr(config, "devices", "auto")
    if isinstance(devices, str) and "," in devices:
        devices = devices.split(",")

    trainer = pl.Trainer(
        max_epochs=getattr(config, "max_epochs", 50),
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
    print(f"Starting training for {getattr(config, 'max_epochs', 50)} epochs...")
    print(f"Configuration: {config.name}")
    print(f"Model parameters: {model_params}")
    print(f"Batch size: {getattr(config, 'batch_size', 32)}")
    print(
        f"Image size: {getattr(config, 'image_size', 96)}x{getattr(config, 'image_size', 96)}"
    )
    if use_wandb and loggers:
        wandb_logger = next(
            (logger for logger in loggers if isinstance(logger, WandbLogger)), None
        )
        if wandb_logger:
            print(
                f"Wandb project: {getattr(config, 'project_name', 'visual_wake_words')}"
            )
            print(f"Wandb run URL: {wandb_logger.experiment.url}")

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    print("Testing the model...")
    test_results = trainer.test(model, test_loader)

    # Log final test results to wandb if enabled
    if use_wandb and test_results:
        wandb.log(
            {
                "final/test_loss": test_results[0]["test_loss"],
                "final/test_acc": test_results[0]["test_acc"],
            }
        )

    # Log sample predictions on test set if wandb is enabled
    if use_wandb:
        print("Logging sample predictions...")
        device = next(model.parameters()).device
        log_sample_predictions(model, test_loader, device, num_samples=16)

    # Save final model
    project_name = getattr(config, "project_name", "visual_wake_words")
    final_model_path = Path(log_dir) / project_name / f"{config.name}.pt"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save model as wandb artifact if enabled
    if use_wandb:
        artifact = wandb.Artifact("final_model", type="model")
        artifact.add_file(str(final_model_path))
        wandb.log_artifact(artifact)

    print("Training completed!")

    # Finish wandb run if used
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
