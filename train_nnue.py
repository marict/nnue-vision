import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

import runpod_service
import wandb
from config import ConfigError, load_config
from data import create_data_loaders
from model import NNUE, LossParams
from training_utils import (
    check_disk_space_emergency,
    cleanup_disk_space_emergency,
    early_log,
    get_disk_usage_percent,
    log_git_commit_info,
    log_git_info_to_wandb,
    replay_early_logs_to_wandb,
)


def adapt_batch_for_nnue(batch, num_ls_buckets=8):
    """
    Adapt batch from dataset format (images, labels) to NNUE format.

    Args:
        batch: Tuple of (images, labels) from dataset
        num_ls_buckets: Number of layer stack buckets

    Returns:
        Tuple of (images, targets, scores, layer_stack_indices) for NNUE
    """
    images, labels = batch
    batch_size = images.shape[0]
    device = images.device  # Get device from images

    # Convert labels to targets (float format for loss computation)
    targets = labels.float().to(device)

    # Generate synthetic scores (in real NNUE training, these would be search evaluation scores)
    # For visual wake words, we'll use dummy scores
    scores = torch.zeros_like(targets, device=device)

    # Generate random layer stack indices (bucket selection) on the same device
    layer_stack_indices = torch.randint(0, num_ls_buckets, (batch_size,), device=device)

    return images, targets, scores, layer_stack_indices


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

        # Log disk usage periodically (every 10 steps to avoid overhead)
        if trainer.global_step % 10 == 0:
            disk_usage = get_disk_usage_percent()
            wandb.log({"system/disk_usage_percent": disk_usage})

            # Check for disk space emergency
            if check_disk_space_emergency(
                threshold=90.0
            ):  # Lower threshold for warnings
                wandb.log({"alerts/disk_space_warning": True})
                print("⚠️  Warning: Disk space is getting low!")

                if disk_usage > 95.0:  # Critical threshold
                    wandb.log({"alerts/disk_space_critical": True})
                    cleanup_disk_space_emergency()

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
        # Generate layer stack indices for NNUE (random bucket selection)
        batch_size = images.shape[0]
        # Get number of buckets from the model
        num_buckets = getattr(model, "num_ls_buckets", 8)
        layer_stack_indices = torch.randint(
            0, num_buckets, (batch_size,), device=device
        )

        logits = model(images, layer_stack_indices)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

    # Create wandb images with predictions
    # Determine number of classes from the data
    num_classes = logits.shape[1]
    if num_classes == 2:
        class_names = ["No Person", "Person"]
    else:
        # For other datasets like CIFAR-10, use generic class names
        class_names = [f"Class {i}" for i in range(num_classes)]

    wandb_images = []

    for i in range(len(images)):
        img = images[i].cpu()
        true_label = labels[i].item()
        pred_label = preds[i].item()
        confidence = probs[i][pred_label].item()

        # Ensure labels are within bounds
        true_label_name = (
            class_names[true_label]
            if true_label < len(class_names)
            else f"Unknown({true_label})"
        )
        pred_label_name = (
            class_names[pred_label]
            if pred_label < len(class_names)
            else f"Unknown({pred_label})"
        )

        # Convert tensor to wandb image format
        # Note: images are normalized, so we need to denormalize for visualization
        wandb_img = wandb.Image(
            img,
            caption=f"True: {true_label_name}, Pred: {pred_label_name} ({confidence:.3f})",
        )
        wandb_images.append(wandb_img)

    wandb.log({"validation/sample_predictions": wandb_images})


def setup_wandb_logger(config, wandb_run_id: str | None = None) -> WandbLogger:
    """Set up wandb logger with comprehensive configuration."""

    # Generate run name with timestamp and key parameters
    run_name = f"nnue-lr{config.learning_rate}-bs{config.batch_size}-img{getattr(config, 'input_size', (96, 96))[0]}"
    if hasattr(config, "note") and config.note:
        run_name += f"-{config.note}"

    # Create wandb config
    wandb_config = {
        # Model parameters
        "model/learning_rate": config.learning_rate,
        "model/input_size": getattr(config, "input_size", (96, 96)),
        "model/num_classes": getattr(config, "num_classes", 2),
        "model/num_ls_buckets": getattr(config, "num_ls_buckets", 8),
        "model/visual_threshold": getattr(config, "visual_threshold", 0.0),
        # Training parameters
        "train/batch_size": config.batch_size,
        "train/max_epochs": config.max_epochs,
        "train/image_size": getattr(config, "input_size", (96, 96))[0],
        "train/num_workers": getattr(config, "num_workers", 4),
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

    # Handle W&B run resumption if run ID is provided
    wandb_kwargs = {
        "project": config.project_name,
        "name": run_name,
        "config": wandb_config,
        "save_dir": config.log_dir,
        "log_model": True,  # Save model checkpoints to wandb
    }

    if wandb_run_id:
        early_log(f"🔄 Resuming W&B run: {wandb_run_id}")
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"

    # Initialize wandb logger
    wandb_logger = WandbLogger(**wandb_kwargs)

    return wandb_logger


class NNUEWrapper(pl.LightningModule):
    """
    Wrapper for NNUE model that adapts data format from standard (images, labels)
    to NNUE format (images, targets, scores, layer_stack_indices).
    """

    def __init__(self, nnue_model):
        super().__init__()
        self.nnue = nnue_model
        self.num_ls_buckets = nnue_model.num_ls_buckets

    def _compute_loss(self, batch, batch_idx):
        """Compute loss without logging (internal version of NNUE step_)"""
        # This replicates the NNUE step_ method but without logging

        # We clip weights at the start of each step. This means that after
        # the last step the weights might be outside of the desired range.
        # They should be also clipped accordingly in the serializer.
        self.nnue._clip_weights()

        (
            images,  # RGB images (B, 3, 96, 96)
            targets,  # Target labels
            scores,  # Search scores
            layer_stack_indices,  # Bucket indices
        ) = batch

        # Forward pass
        scorenet = self.nnue(images, layer_stack_indices) * self.nnue.nnue2score

        p = self.nnue.loss_params
        # convert the network and search scores to an estimate match result
        # based on the win_rate_model, with scalings and offsets optimized
        q = (scorenet - p.in_offset) / p.in_scaling
        qm = (-scorenet - p.in_offset) / p.in_scaling
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

        s = (scores - p.out_offset) / p.out_scaling
        sm = (-scores - p.out_offset) / p.out_scaling
        pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

        # blend that eval based score with the actual targets
        t = targets
        actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * (
            self.current_epoch / self.nnue.max_epoch
        )
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        # use a MSE-like loss function
        loss = torch.pow(torch.abs(pt - qf), p.pow_exp)
        if p.qp_asymmetry != 0.0:
            loss = loss * ((qf > pt) * p.qp_asymmetry + 1)
        loss = loss.mean()

        return loss

    def training_step(self, batch, batch_idx):
        adapted_batch = adapt_batch_for_nnue(batch, self.num_ls_buckets)
        # Compute loss without internal logging
        loss = self._compute_loss(adapted_batch, batch_idx)
        # Log using the wrapper's logging context
        if getattr(self, "_trainer", None) is not None:
            self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        adapted_batch = adapt_batch_for_nnue(batch, self.num_ls_buckets)
        # Compute loss without internal logging
        loss = self._compute_loss(adapted_batch, batch_idx)
        # Log using the wrapper's logging context
        if getattr(self, "_trainer", None) is not None:
            self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        adapted_batch = adapt_batch_for_nnue(batch, self.num_ls_buckets)
        # Compute loss without internal logging
        loss = self._compute_loss(adapted_batch, batch_idx)
        # Log using the wrapper's logging context
        if getattr(self, "_trainer", None) is not None:
            self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.nnue.configure_optimizers()

    def forward(self, images, layer_stack_indices):
        return self.nnue.forward(images, layer_stack_indices)


def main():
    # Early logging and system info
    early_log("🚀 Starting NNUE-Vision training...")
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
        description="Train CNN on Visual Wake Words dataset with config file support"
    )

    # Configuration file
    parser.add_argument(
        "--config",
        default="config/train_nnue_default.py",
        type=str,
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

    # Set up loss parameters from config
    loss_params = LossParams(
        start_lambda=getattr(config, "start_lambda", 1.0),
        end_lambda=getattr(config, "end_lambda", 1.0),
    )

    # Create NNUE model with config parameters
    nnue_model = NNUE(
        max_epoch=getattr(config, "max_epochs", 50),
        lr=getattr(config, "learning_rate", 1e-3),
        loss_params=loss_params,
        num_ls_buckets=getattr(config, "num_ls_buckets", 8),
        visual_threshold=getattr(config, "visual_threshold", 0.0),
    )

    # Wrap NNUE model to handle data format adaptation
    model = NNUEWrapper(nnue_model)

    # Create data loaders
    early_log("📚 Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=getattr(config, "batch_size", 32),
        num_workers=getattr(config, "num_workers", 4),
        target_size=getattr(config, "input_size", (96, 96)),
        subset=getattr(config, "subset", 1.0),
    )

    # Set up logging
    log_dir = args.log_dir or getattr(config, "log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Update config with log_dir for consistency
    config.log_dir = log_dir

    # Setup wandb logger (always required)
    wandb_logger = setup_wandb_logger(config, wandb_run_id=args.wandb_run_id)
    loggers = [wandb_logger]

    # Log git information to wandb for experiment tracking
    early_log("📤 Logging git information to W&B...")
    log_git_info_to_wandb(wandb_logger.experiment)

    # Replay early logs to wandb
    replay_early_logs_to_wandb()

    # Set up callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=getattr(config, "log_interval", 50)),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=getattr(config, "save_top_k", 3),
            filename="best-{epoch:02d}-{val_loss:.3f}",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=getattr(config, "patience", 10),
            verbose=True,
        ),
    ]

    # Add wandb metrics callback (always required)
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
    print(f"Model: NNUE with {getattr(config, 'num_ls_buckets', 8)} layer stacks")
    print(f"Learning rate: {getattr(config, 'learning_rate', 1e-3)}")
    print(f"Batch size: {getattr(config, 'batch_size', 32)}")
    print(
        f"Image size: {getattr(config, 'input_size', (96, 96))[0]}x{getattr(config, 'input_size', (96, 96))[1]}"
    )
    # Display wandb information (always available)
    wandb_logger = next(
        (logger for logger in loggers if isinstance(logger, WandbLogger)), None
    )
    if wandb_logger:
        print(f"Wandb project: {getattr(config, 'project_name', 'visual_wake_words')}")
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
                "final/test_loss": test_results[0].get("test_loss", 0.0),
                "final/test_acc": test_results[0].get("test_acc", 0.0),
            }
        )

    # Log sample predictions on test set (always required)
    print("Logging sample predictions...")
    device = next(model.parameters()).device
    log_sample_predictions(model, test_loader, device, num_samples=16)

    # Save final model
    project_name = getattr(config, "project_name", "visual_wake_words")
    final_model_path = Path(log_dir) / project_name / f"{config.name}.pt"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save model as wandb artifact (always required)
    artifact = wandb.Artifact("final_model", type="model")
    artifact.add_file(str(final_model_path))
    wandb.log_artifact(artifact)

    print("Training completed!")

    # Log final system information
    final_disk_usage = get_disk_usage_percent()
    print(f"📊 Final disk usage: {final_disk_usage:.1f}%")

    # Log final system stats to wandb (always required)
    wandb.log(
        {
            "final/disk_usage_percent": final_disk_usage,
            "final/training_completed": True,
        }
    )

    print("🎯 Training metrics and git info logged to W&B")

    # Finish wandb run (always required)
    wandb.finish()

    # Stop RunPod instance if we're running on RunPod and keep-alive is not enabled
    if os.getenv("RUNPOD_POD_ID") and not getattr(config, "keep_alive", False):
        try:
            runpod_service.stop_runpod()
        except ImportError:
            pass  # runpod_service not available in this environment


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error in training: {e}")
        traceback.print_exc()

        # Stop RunPod instance on error if we're running on RunPod
        if os.getenv("RUNPOD_POD_ID"):
            try:
                runpod_service.stop_runpod()
            except ImportError:
                pass  # runpod_service not available in this environment

        # Re-raise the exception to ensure proper exit code
        raise
