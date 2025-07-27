"""Unified Training Framework

This module provides a unified training system for both NNUE and EtinyNet models,
extracting common functionality and providing model-specific adapters.
"""

import argparse
import os
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import runpod_service_nnue
import wandb
from config import ConfigError, load_config
from data import create_data_loaders
from model import NNUE, EtinyNet, LossParams
from training_utils import (
    check_disk_space_emergency,
    cleanup_disk_space_emergency,
    early_log,
    get_disk_usage_percent,
    log_git_commit_info,
    log_git_info_to_wandb,
    replay_early_logs_to_wandb,
)


class ModelAdapter(ABC):
    """Abstract base class for model-specific training adapters."""

    @abstractmethod
    def get_model_type_name(self) -> str:
        """Return the name of the model type."""
        pass

    @abstractmethod
    def create_model(self, config: Any) -> pl.LightningModule:
        """Create and return the model instance."""
        pass

    @abstractmethod
    def create_data_loaders(self, config: Any) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """Create and return train, validation, and test data loaders."""
        pass

    @abstractmethod
    def get_callbacks(self, config: Any, log_dir: str) -> List[Callback]:
        """Return model-specific callbacks."""
        pass

    @abstractmethod
    def setup_wandb_config(self, config: Any) -> Dict[str, Any]:
        """Setup wandb configuration dictionary."""
        pass

    @abstractmethod
    def get_model_specific_args(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Add model-specific arguments to the parser."""
        pass

    @abstractmethod
    def apply_model_specific_overrides(
        self, config: Any, args: argparse.Namespace
    ) -> None:
        """Apply model-specific command-line overrides to config."""
        pass

    @abstractmethod
    def log_sample_predictions(
        self,
        model: pl.LightningModule,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """Log sample predictions to wandb."""
        pass

    @abstractmethod
    def save_final_model(
        self, model: pl.LightningModule, config: Any, log_dir: str
    ) -> str:
        """Save the final model and return the path."""
        pass

    def get_default_config_path(self) -> str:
        """Return the default configuration file path."""
        return f"config/train_{self.get_model_type_name().lower()}_default.py"

    def get_run_name(self, config: Any) -> str:
        """Generate a run name for wandb."""
        model_name = self.get_model_type_name().lower()
        run_name = f"{model_name}-lr{config.learning_rate}-bs{config.batch_size}"
        if hasattr(config, "note") and config.note:
            run_name += f"-{config.note}"
        return run_name


class CompactProgressLogger(Callback):
    """Custom callback for compact iteration-based logging similar to the desired format."""

    def __init__(self):
        super().__init__()
        self.step_start_time = None
        self.iter_count = 0
        self.running_metrics = {}
        self.val_metrics_history = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Record timing for each batch."""
        self.step_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log compact training metrics after each batch."""
        self.iter_count += 1

        # Calculate timing
        step_time_ms = 0.0
        if self.step_start_time is not None:
            step_time_ms = (time.time() - self.step_start_time) * 1000

        # Get loss from outputs
        loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()

        # Get learning rate
        lr = 0.0
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]

        # Calculate gradient norms
        grad_norm = 0.0
        param_norm = 0.0
        for param in pl_module.parameters():
            if param.grad is not None:
                param_norm += param.data.norm(2).item() ** 2
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm**0.5
        param_norm = param_norm**0.5

        # Get additional metrics from logged metrics
        logged_metrics = trainer.logged_metrics
        additional_metrics = []

        for key, value in logged_metrics.items():
            if (
                "train" in key
                and key != "train_loss"
                and isinstance(value, torch.Tensor)
            ):
                metric_name = key.replace("train_", "").replace("train/", "")
                additional_metrics.append(f"{metric_name} {value.item():.4f}")

        # Memory usage if available
        memory_info = ""
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / (1024**3)
            memory_info = f", mem {memory_gb:.2f}GB"

        # Format timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Create the compact log line (similar to desired format but with available metrics)
        metrics_str = ""
        if additional_metrics:
            metrics_str = ", " + ", ".join(additional_metrics)

        # Add gradient and parameter norms as pseudo-metrics
        grad_metrics = f", grad_norm {grad_norm:.4f}, param_norm {param_norm:.4f}"

        log_line = (
            f"{timestamp}\n"
            f"iter {self.iter_count}: loss {loss:.4f}{metrics_str}{grad_metrics}"
            f"{memory_info}, time {step_time_ms:.2f}ms"
        )

        print(log_line)

        # Update running averages for validation display
        self.running_metrics["loss"] = loss
        self.running_metrics["lr"] = lr
        self.running_metrics["grad_norm"] = grad_norm

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation results with additional metrics."""
        # Get validation metrics
        val_metrics = {}
        for key, value in trainer.logged_metrics.items():
            if "val" in key and isinstance(value, torch.Tensor):
                metric_name = key.replace("val_", "").replace("val/", "")
                val_metrics[metric_name] = value.item()

        # Store for history
        self.val_metrics_history.append(val_metrics.copy())

        # Print validation summary with more detail
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"{timestamp}")

        # Format validation metrics similar to training format
        val_loss = val_metrics.get("loss", 0.0)
        val_acc = val_metrics.get("acc", 0.0)
        val_f1 = val_metrics.get("f1", 0.0)
        val_precision = val_metrics.get("precision", 0.0)
        val_recall = val_metrics.get("recall", 0.0)

        validation_line = (
            f"Validation: loss {val_loss:.4f}, acc {val_acc:.4f}, "
            f"f1 {val_f1:.4f}, precision {val_precision:.4f}, recall {val_recall:.4f}"
        )

        # Add any other validation metrics
        other_metrics = []
        for key, value in val_metrics.items():
            if key not in ["loss", "acc", "f1", "precision", "recall"]:
                other_metrics.append(f"{key} {value:.4f}")

        if other_metrics:
            validation_line += ", " + ", ".join(other_metrics)

        print(validation_line)

        # Show improvement if we have history
        if len(self.val_metrics_history) > 1:
            prev_loss = self.val_metrics_history[-2].get("loss", val_loss)
            improvement = prev_loss - val_loss
            if improvement > 0:
                print(
                    f"Validation loss improved by {improvement:.4f}. New best: {val_loss:.4f}"
                )

    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch completion."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp}")
        print(f"Epoch {trainer.current_epoch + 1} completed")

    def on_train_end(self, trainer, pl_module):
        """Log training completion."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp}")
        print("Training completed!")


class BaseTrainer:
    """Base trainer class containing common training logic."""

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter

    def setup_argument_parser(self) -> argparse.ArgumentParser:
        """Set up the argument parser with common and model-specific arguments."""
        parser = argparse.ArgumentParser(
            description=f"Train {self.adapter.get_model_type_name()} model with config file support"
        )

        # Common arguments
        parser.add_argument(
            "--config",
            type=str,
            default=self.adapter.get_default_config_path(),
            help="Path to the configuration file",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=None,
            help="Override batch size from config",
        )
        parser.add_argument(
            "--max_epochs",
            type=int,
            default=None,
            help="Override max epochs from config",
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
            "--wandb-run-id", type=str, default=None, help="Resume specific W&B run"
        )
        parser.add_argument(
            "--log_dir",
            type=str,
            default="logs",
            help="Directory for logs and checkpoints",
        )
        parser.add_argument(
            "--use_augmentation",
            type=lambda x: x.lower() == "true",
            default=None,
            help="Enable data augmentation (true/false)",
        )
        parser.add_argument(
            "--augmentation_strength",
            type=str,
            choices=["light", "medium", "heavy"],
            default=None,
            help="Data augmentation strength level",
        )

        # Add model-specific arguments
        parser = self.adapter.get_model_specific_args(parser)
        return parser

    def load_and_setup_config(self, args: argparse.Namespace) -> Any:
        """Load configuration and apply command-line overrides."""
        # Load configuration
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
        if hasattr(args, "use_augmentation") and args.use_augmentation is not None:
            config.use_augmentation = args.use_augmentation
        if (
            hasattr(args, "augmentation_strength")
            and args.augmentation_strength is not None
        ):
            config.augmentation_strength = args.augmentation_strength

        # Apply model-specific overrides
        self.adapter.apply_model_specific_overrides(config, args)

        # Update config with log_dir for consistency
        config.log_dir = args.log_dir or getattr(config, "log_dir", "logs")

        return config

    def setup_wandb_logger(
        self, config: Any, wandb_run_id: Optional[str] = None
    ) -> WandbLogger:
        """Set up wandb logger with comprehensive configuration."""
        wandb_config = self.adapter.setup_wandb_config(config)

        # Handle W&B run resumption if run ID is provided
        wandb_kwargs = {
            "project": getattr(
                config,
                "project_name",
                f"{self.adapter.get_model_type_name().lower()}_training",
            ),
            "config": wandb_config,
            "save_dir": config.log_dir,
            "log_model": True,
        }

        if wandb_run_id:
            early_log(f"🔄 Resuming W&B run: {wandb_run_id}")
            wandb_kwargs["id"] = wandb_run_id
            # PyTorch Lightning's WandbLogger handles resumption automatically when given an id
            # Don't set a new name when resuming - keep the existing run name
        else:
            # Only set a name for new runs
            run_name = self.adapter.get_run_name(config)
            wandb_kwargs["name"] = run_name

        return WandbLogger(**wandb_kwargs)

    def setup_trainer(
        self, config: Any, loggers: List[pl_loggers.Logger], callbacks: List[Callback]
    ) -> pl.Trainer:
        """Set up the PyTorch Lightning trainer."""
        devices = getattr(config, "devices", "auto")
        if isinstance(devices, str) and "," in devices:
            devices = devices.split(",")

        return pl.Trainer(
            max_epochs=getattr(config, "max_epochs", 50),
            accelerator=getattr(config, "accelerator", "auto"),
            devices=devices,
            logger=loggers if loggers else False,
            callbacks=callbacks,
            log_every_n_steps=getattr(config, "log_interval", 50),
            enable_checkpointing=getattr(config, "always_save_checkpoint", True),
            enable_progress_bar=False,  # Disable default progress bar - using our custom logger
            deterministic=getattr(config, "deterministic", True),
            check_val_every_n_epoch=getattr(config, "check_val_every_n_epoch", 1),
        )

    def run_training(self) -> int:
        """Main training execution method."""
        # Early logging and system info
        early_log(f"🚀 Starting {self.adapter.get_model_type_name()} training...")
        early_log(f"📊 Disk usage: {get_disk_usage_percent():.1f}%")

        # Check for disk space emergencies early
        if check_disk_space_emergency():
            early_log("⚠️  Disk space is critically low!")
            cleaned_mb = cleanup_disk_space_emergency()
            early_log(f"🧹 Emergency cleanup freed {cleaned_mb:.1f} MB")

        # Log git information for debugging and tracking
        early_log("📝 Git repository information:")
        log_git_commit_info()

        # Initialize variables for cleanup
        emergency_wandb_logger = None

        try:
            # Parse arguments and load config
            parser = self.setup_argument_parser()
            args = parser.parse_args()
            config = self.load_and_setup_config(args)

            # Setup wandb API key
            wandb_api_key = args.wandb_api_key or getattr(config, "wandb_api_key", None)
            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key

            if not os.getenv("WANDB_API_KEY"):
                early_log(
                    "❌ Error: WANDB_API_KEY not found. WandB logging is required."
                )
                early_log(
                    "💡 Set the environment variable or use --wandb_api_key argument."
                )
                raise ValueError("WANDB_API_KEY is required for training")

            # Set random seed for reproducibility
            pl.seed_everything(getattr(config, "seed", 42))

            # Set up emergency W&B logger early for crash reporting
            try:
                emergency_wandb_logger = self.setup_emergency_wandb_logger(config, args)
                early_log("🚨 Emergency W&B logger ready for crash reporting")
            except Exception as e:
                early_log(f"⚠️  Failed to setup emergency W&B logger: {e}")
                emergency_wandb_logger = None

            # Create model and data loaders (these are crash-prone areas)
            try:
                early_log("🏗️  Creating model...")
                model = self.adapter.create_model(config)
                early_log("✅ Model created successfully")
            except Exception as e:
                error_msg = f"Fatal error during model creation: {str(e)}"
                early_log(f"❌ {error_msg}")
                if emergency_wandb_logger:
                    self._log_error_to_wandb(
                        emergency_wandb_logger, "model_creation", error_msg, e
                    )
                raise

            try:
                early_log("📚 Creating data loaders...")
                train_loader, val_loader, test_loader = (
                    self.adapter.create_data_loaders(config)
                )
                early_log("✅ Data loaders created successfully")
            except Exception as e:
                error_msg = f"Fatal error during data loader creation: {str(e)}"
                early_log(f"❌ {error_msg}")
                if emergency_wandb_logger:
                    self._log_error_to_wandb(
                        emergency_wandb_logger, "data_loading", error_msg, e
                    )
                raise

            # Set up logging directory
            log_dir = config.log_dir
            os.makedirs(log_dir, exist_ok=True)

            # Setup wandb logger
            wandb_logger = self.setup_wandb_logger(
                config, wandb_run_id=getattr(args, "wandb_run_id", None)
            )
            loggers = [wandb_logger]

            # Log git information to wandb for experiment tracking
            early_log("📤 Logging git information to W&B...")
            log_git_info_to_wandb(wandb_logger.experiment)

            # Replay early logs to wandb
            replay_early_logs_to_wandb()

            # Set up callbacks
            callbacks = [
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
                CompactProgressLogger(),
            ]

            # Add model-specific callbacks
            model_callbacks = self.adapter.get_callbacks(config, log_dir)
            callbacks.extend(model_callbacks)

            # Set up trainer
            try:
                trainer = self.setup_trainer(config, loggers, callbacks)
                early_log("✅ Trainer setup completed successfully")
            except Exception as e:
                error_msg = f"Fatal error during trainer setup: {str(e)}"
                early_log(f"❌ {error_msg}")
                if emergency_wandb_logger:
                    self._log_error_to_wandb(
                        emergency_wandb_logger, "trainer_setup", error_msg, e
                    )
                raise

            # Log training start information
            self._log_training_start(config, wandb_logger)

            # Train the model
            try:
                early_log("🚀 Starting training...")
                trainer.fit(model, train_loader, val_loader)
                early_log("✅ Training completed successfully")
            except Exception as e:
                error_msg = f"Fatal error during training: {str(e)}"
                early_log(f"❌ {error_msg}")
                # Log to both emergency logger and main logger
                if emergency_wandb_logger:
                    self._log_error_to_wandb(
                        emergency_wandb_logger, "training_execution", error_msg, e
                    )
                if wandb_logger:
                    self._log_error_to_wandb(
                        wandb_logger, "training_execution", error_msg, e
                    )
                raise

            # Test the model
            try:
                print("Testing the model...")
                test_results = trainer.test(model, test_loader)
            except Exception as e:
                error_msg = f"Error during testing: {str(e)}"
                early_log(f"⚠️  {error_msg}")
                # Non-fatal error, log but continue
                if wandb_logger:
                    self._log_error_to_wandb(wandb_logger, "testing", error_msg, e)
                test_results = []

            # Log final test results to wandb
            if test_results:
                try:
                    if hasattr(wandb_logger.experiment, "log"):
                        wandb_logger.experiment.log(
                            {
                                "final/test_loss": test_results[0].get(
                                    "test_loss", 0.0
                                ),
                                "final/test_acc": test_results[0].get("test_acc", 0.0),
                            }
                        )
                    else:
                        # Fallback for test environments
                        import wandb

                        wandb.log(
                            {
                                "final/test_loss": test_results[0].get(
                                    "test_loss", 0.0
                                ),
                                "final/test_acc": test_results[0].get("test_acc", 0.0),
                            }
                        )
                except Exception:
                    # If logging fails, continue gracefully
                    pass

            # Sample predictions logging removed - trusting F1 scores and metrics

            # Save final model
            final_model_path = self.adapter.save_final_model(model, config, log_dir)
            print(f"Final model saved to: {final_model_path}")

            # Save model as wandb artifact
            try:
                artifact = wandb.Artifact("final_model", type="model")
                artifact.add_file(str(final_model_path))
                wandb.log_artifact(artifact)
            except Exception:
                # If artifact logging fails (e.g., in test environments), continue gracefully
                pass

            print("Training completed!")

            # Log final system information
            final_disk_usage = get_disk_usage_percent()
            print(f"📊 Final disk usage: {final_disk_usage:.1f}%")

            # Log final system stats to wandb
            try:
                if hasattr(wandb_logger.experiment, "log"):
                    wandb_logger.experiment.log(
                        {
                            "final/disk_usage_percent": final_disk_usage,
                            "final/training_completed": True,
                        }
                    )
                else:
                    # Fallback for test environments
                    import wandb

                    wandb.log(
                        {
                            "final/disk_usage_percent": final_disk_usage,
                            "final/training_completed": True,
                        }
                    )
            except Exception:
                # If logging fails, continue gracefully
                pass

            print("🎯 Training metrics and git info logged to W&B")

            # Finish wandb run
            import wandb

            wandb.finish()

            # Stop RunPod instance if we're running on RunPod and keep-alive is not enabled
            if os.getenv("RUNPOD_POD_ID") and not getattr(config, "keep_alive", False):
                try:
                    runpod_service_nnue.stop_runpod()
                except ImportError:
                    pass  # runpod_service not available in this environment

            return 0

        except Exception as e:
            error_msg = f"Fatal error in training framework: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()

            # Try to log error to W&B with multiple fallbacks
            logged_to_wandb = False

            # Try main wandb logger first
            if "wandb_logger" in locals() and wandb_logger:
                try:
                    self._log_error_to_wandb(
                        wandb_logger, "framework_failure", error_msg, e
                    )
                    early_log("🚨 Error logged to main W&B logger")
                    logged_to_wandb = True
                except Exception as wandb_error:
                    early_log(f"⚠️  Failed to log error to main W&B: {wandb_error}")

            # Fallback to emergency W&B logger
            if not logged_to_wandb and emergency_wandb_logger:
                try:
                    self._log_error_to_wandb(
                        emergency_wandb_logger, "framework_failure", error_msg, e
                    )
                    early_log("🚨 Error logged to emergency W&B")
                    logged_to_wandb = True
                except Exception as wandb_error:
                    early_log(f"⚠️  Failed to log error to emergency W&B: {wandb_error}")

            # Last resort: try to create a minimal W&B run just for error logging
            if not logged_to_wandb:
                try:
                    import wandb

                    error_run = wandb.init(
                        project=getattr(config, "project_name", "nnue-vision-errors"),
                        name=f"error-{int(time.time())}",
                        config={
                            "error": True,
                            "model": self.adapter.get_model_type_name(),
                        },
                    )
                    wandb.log(
                        {
                            "error/message": error_msg,
                            "error/type": type(e).__name__,
                            "error/traceback": traceback.format_exc(),
                            "error/timestamp": time.time(),
                            "error/stage": "framework_failure",
                        }
                    )
                    wandb.finish(exit_code=1)
                    early_log("🚨 Error logged to emergency W&B run")
                except Exception as last_resort_error:
                    early_log(f"💥 All W&B error logging failed: {last_resort_error}")

            # Stop RunPod instance on error if we're running on RunPod
            if os.getenv("RUNPOD_POD_ID"):
                try:
                    runpod_service_nnue.stop_runpod()
                except ImportError:
                    pass  # runpod_service not available in this environment

            # Return error code instead of re-raising to ensure W&B sync
            return 1

    def _log_training_start(self, config: Any, wandb_logger: WandbLogger) -> None:
        """Log training start information."""
        print(
            f"Starting {self.adapter.get_model_type_name()} training for {getattr(config, 'max_epochs', 50)} epochs..."
        )
        print(f"Configuration: {config.name}")
        print(f"Learning rate: {getattr(config, 'learning_rate', 1e-3)}")
        print(f"Batch size: {getattr(config, 'batch_size', 32)}")
        print(f"Wandb project: {getattr(config, 'project_name', 'training')}")
        print(f"Wandb run URL: {wandb_logger.experiment.url}")

    def setup_emergency_wandb_logger(
        self, config: Any, args: argparse.Namespace
    ) -> WandbLogger:
        """Set up a minimal W&B logger for early crash reporting."""
        from pytorch_lightning.loggers import WandbLogger

        import wandb

        # Generate a simple run name for emergency logging
        run_name = (
            f"emergency_{self.adapter.get_model_type_name().lower()}_{config.name}"
        )

        # Minimal config for emergency logging
        emergency_config = {
            "emergency_run": True,
            "model_type": self.adapter.get_model_type_name(),
            "config_name": config.name,
            "crash_detection": True,
        }

        # Set up emergency W&B logger
        wandb_kwargs = {
            "project": getattr(
                config,
                "project_name",
                f"{self.adapter.get_model_type_name().lower()}_emergency",
            ),
            "name": run_name,
            "config": emergency_config,
            "save_dir": getattr(config, "log_dir", "logs"),
            "log_model": False,  # Don't log models for emergency runs
            "tags": [
                "emergency",
                "crash_detection",
                self.adapter.get_model_type_name().lower(),
            ],
        }

        return WandbLogger(**wandb_kwargs)

    def _log_error_to_wandb(
        self,
        wandb_logger: WandbLogger,
        error_stage: str,
        error_msg: str,
        exception: Exception,
    ) -> None:
        """Log error information to W&B for debugging."""
        import os
        import sys
        import time
        import traceback

        import wandb

        # Get full traceback
        tb_str = traceback.format_exc()

        # Collect system information
        system_info = {
            "python_version": sys.version,
            "torch_available": False,
            "cuda_available": False,
            "pwd": os.getcwd(),
            "env_vars": {
                k: v
                for k, v in os.environ.items()
                if "WANDB" in k or "CUDA" in k or "TORCH" in k
            },
        }

        try:
            import torch

            system_info["torch_available"] = True
            system_info["torch_version"] = torch.__version__
            system_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info["cuda_device_count"] = torch.cuda.device_count()
                system_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass

        # Log error details to W&B
        error_log = {
            f"error/{error_stage}/message": error_msg,
            f"error/{error_stage}/type": type(exception).__name__,
            f"error/{error_stage}/traceback": tb_str,
            f"error/{error_stage}/timestamp": time.time(),
            "system/python_version": system_info["python_version"],
            "system/torch_available": system_info["torch_available"],
            "system/cuda_available": system_info["cuda_available"],
            "system/working_directory": system_info["pwd"],
        }

        if system_info["torch_available"]:
            error_log["system/torch_version"] = system_info["torch_version"]
            if system_info["cuda_available"]:
                error_log["system/cuda_device_count"] = system_info["cuda_device_count"]
                error_log["system/cuda_device_name"] = system_info["cuda_device_name"]

        # Log to W&B
        wandb_logger.experiment.log(error_log)

        # Also log as a summary for easy access
        wandb_logger.experiment.summary[f"final_error_{error_stage}"] = error_msg
        wandb_logger.experiment.summary["final_status"] = "FAILED"
        wandb_logger.experiment.summary["error_stage"] = error_stage

        # Mark run as failed
        wandb_logger.experiment.mark_preempting()

        # Try to finish the run gracefully
        try:
            wandb.finish(exit_code=1)
        except:
            pass  # If finish fails, don't crash the crash handler
