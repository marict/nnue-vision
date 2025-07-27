"""EtinyNet Model Adapter

Provides EtinyNet-specific training functionality for the unified training framework.
"""

import argparse
import os
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from data import create_data_loaders
from model import EtinyNet
from training_framework import ModelAdapter
from training_utils import early_log, generate_run_name


class EtinyNetAdapter(ModelAdapter):
    """Model adapter for EtinyNet training."""

    def get_model_type_name(self) -> str:
        """Return the name of the model type."""
        return "EtinyNet"

    def create_model(self, config: Any) -> pl.LightningModule:
        """Create and return the EtinyNet model instance."""
        variant = getattr(config, "etinynet_variant", "0.75")

        # Determine number of classes based on dataset
        dataset_name = getattr(config, "dataset_name", "cifar10")
        if dataset_name == "cifar10":
            num_classes = 10
        elif dataset_name == "cifar100":
            num_classes = 100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Convert input_size from tuple to integer (EtinyNet expects square images)
        input_size_tuple = getattr(config, "input_size", (32, 32))
        input_size = (
            input_size_tuple[0]
            if isinstance(input_size_tuple, tuple)
            else input_size_tuple
        )

        return EtinyNet(
            variant=variant,
            num_classes=num_classes,
            input_size=input_size,
            use_asq=getattr(config, "use_asq", False),
            asq_bits=getattr(config, "asq_bits", 4),
            lr=getattr(config, "learning_rate", 0.1),
            max_epochs=getattr(config, "max_epochs", 200),
            weight_decay=getattr(config, "weight_decay", 1e-4),
        )

    def create_data_loaders(self, config: Any) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """Create and return train, validation, and test data loaders."""
        dataset_name = getattr(config, "dataset_name", "cifar10")

        return create_data_loaders(
            dataset_name=dataset_name,
            batch_size=getattr(config, "batch_size", 64),
            num_workers=getattr(config, "num_workers", 4),
            target_size=getattr(config, "input_size", (32, 32)),
            max_samples_per_split=(
                None
                if getattr(config, "subset", 1.0) >= 1.0
                else int(50000 * getattr(config, "subset", 1.0))
            ),
            use_augmentation=getattr(config, "use_augmentation", True),
            augmentation_strength=getattr(config, "augmentation_strength", "medium"),
        )

    def get_callbacks(self, config: Any, log_dir: str) -> List[Callback]:
        """Return EtinyNet-specific callbacks."""
        variant = getattr(config, "etinynet_variant", "0.75")
        dataset_name = getattr(config, "dataset_name", "cifar10")

        # Model checkpointing with accuracy monitoring
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename=f"etinynet-{variant}-" + "{epoch:02d}-{val_acc:.3f}",
            monitor="val_acc",
            mode="max",
            save_top_k=getattr(config, "save_top_k", 3),
            save_last=True,
            auto_insert_metric_name=False,
        )

        return [checkpoint_callback]

    def setup_wandb_config(self, config: Any) -> Dict[str, Any]:
        """Setup wandb configuration dictionary."""
        variant = getattr(config, "etinynet_variant", "0.75")
        dataset_name = getattr(config, "dataset_name", "cifar10")

        # Determine number of classes based on dataset
        if dataset_name == "cifar10":
            num_classes = 10
        elif dataset_name == "cifar100":
            num_classes = 100
        else:
            num_classes = 10  # Default fallback

        wandb_config = {
            "model_type": "EtinyNet",
            "variant": variant,
            "dataset": dataset_name,
            "num_classes": num_classes,
            "config_file": getattr(config, "_config_file", "unknown"),
            # Model parameters
            "model/learning_rate": config.learning_rate,
            "model/variant": variant,
            "model/use_asq": getattr(config, "use_asq", False),
            "model/asq_bits": getattr(config, "asq_bits", 4),
            # Training parameters
            "train/batch_size": config.batch_size,
            "train/max_epochs": config.max_epochs,
            "train/dataset": dataset_name,
            "train/num_workers": getattr(config, "num_workers", 4),
            # System parameters
            "system/cuda_available": torch.cuda.is_available(),
            "system/torch_version": torch.__version__,
            "system/device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
            # Training configuration
            "config/accelerator": getattr(config, "accelerator", "auto"),
            "config/patience": getattr(config, "patience", 15),
            "config/save_top_k": getattr(config, "save_top_k", 3),
            "config/name": config.name,
        }

        # Add other config attributes
        for k, v in vars(config).items():
            if not k.startswith("_") and k not in wandb_config:
                wandb_config[k] = v

        return wandb_config

    def get_model_specific_args(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Add EtinyNet-specific arguments to the parser."""
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
        return parser

    def apply_model_specific_overrides(
        self, config: Any, args: argparse.Namespace
    ) -> None:
        """Apply EtinyNet-specific command-line overrides to config."""
        # EtinyNet-specific parameters
        if hasattr(args, "variant") and args.variant is not None:
            config.etinynet_variant = args.variant
        elif not hasattr(config, "etinynet_variant"):
            config.etinynet_variant = "0.75"

        if hasattr(args, "dataset") and args.dataset is not None:
            config.dataset_name = args.dataset
        elif not hasattr(config, "dataset_name"):
            config.dataset_name = "cifar10"

    def get_run_name(self, config: Any) -> str:
        """Generate a run name for wandb using RunPod ID when available."""
        return generate_run_name("etinynet", config.name)

    def log_sample_predictions(
        self,
        model: pl.LightningModule,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """Log sample predictions to wandb."""
        # EtinyNet doesn't currently implement sample prediction logging
        # This could be implemented later if needed
        pass

    def save_final_model(
        self, model: pl.LightningModule, config: Any, log_dir: str
    ) -> str:
        """Save the final model and return the path."""
        variant = getattr(config, "etinynet_variant", "0.75")
        dataset_name = getattr(config, "dataset_name", "cifar10")

        final_model_path = os.path.join(
            log_dir, f"etinynet_{variant}_{dataset_name}_final.pt"
        )
        torch.save(model.state_dict(), final_model_path)
        early_log(f"💾 Saved final model to: {final_model_path}")

        # Log serialization instructions
        print(f"\nTo serialize this model for C++ engine:")
        print(
            f"python serialize.py {final_model_path} etinynet_{variant}_{dataset_name}.etiny"
        )

        return final_model_path
