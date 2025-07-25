"""NNUE Model Adapter

Provides NNUE-specific training functionality for the unified training framework.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

import wandb
from data import create_data_loaders
from model import NNUE, LossParams
from training_framework import ModelAdapter


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
        logits = self.nnue(images, layer_stack_indices)

        # Use different loss functions based on number of classes
        if self.nnue.num_classes > 1:
            # Multi-class classification: use CrossEntropyLoss
            loss = F.cross_entropy(logits, targets.long())
        else:
            # Original NNUE loss for single output (chess-style evaluation)
            scorenet = logits * self.nnue.nnue2score

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


class NNUEAdapter(ModelAdapter):
    """Model adapter for NNUE training."""

    def get_model_type_name(self) -> str:
        """Return the name of the model type."""
        return "NNUE"

    def create_model(self, config: Any) -> pl.LightningModule:
        """Create and return the NNUE model instance."""
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
            num_classes=getattr(config, "num_classes", 1),
        )

        # Wrap NNUE model to handle data format adaptation
        return NNUEWrapper(nnue_model)

    def create_data_loaders(self, config: Any) -> Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ]:
        """Create and return train, validation, and test data loaders."""
        return create_data_loaders(
            batch_size=getattr(config, "batch_size", 32),
            num_workers=getattr(config, "num_workers", 4),
            target_size=getattr(config, "input_size", (96, 96)),
            subset=getattr(config, "subset", 1.0),
            binary_classification=getattr(config, "binary_classification", None),
        )

    def get_callbacks(self, config: Any, log_dir: str) -> List[Callback]:
        """Return NNUE-specific callbacks."""
        return []  # NNUE uses standard callbacks from base framework

    def setup_wandb_config(self, config: Any) -> Dict[str, Any]:
        """Setup wandb configuration dictionary."""
        return {
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
            # Add note if present
            **(
                {"experiment/note": config.note}
                if hasattr(config, "note") and config.note
                else {}
            ),
        }

    def get_model_specific_args(
        self, parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        """Add NNUE-specific arguments to the parser."""
        # No NNUE-specific arguments currently needed
        return parser

    def apply_model_specific_overrides(
        self, config: Any, args: argparse.Namespace
    ) -> None:
        """Apply NNUE-specific command-line overrides to config."""
        # No NNUE-specific overrides currently needed
        pass

    def log_sample_predictions(
        self,
        model: pl.LightningModule,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> None:
        """Log sample predictions to wandb."""
        model.eval()
        num_samples = 8

        # Get a batch of test data
        test_batch = next(iter(test_loader))
        images, labels = test_batch
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
            # For other datasets, use generic class names
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

            # Denormalize image for proper visualization
            # ImageNet normalization parameters used in data pipeline
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            denorm_img = torch.clamp(img * std + mean, 0, 1)

            # Convert tensor to wandb image format
            wandb_img = wandb.Image(
                denorm_img,
                caption=f"True: {true_label_name}, Pred: {pred_label_name} ({confidence:.3f})",
            )
            wandb_images.append(wandb_img)

        wandb.log({"validation/sample_predictions": wandb_images})

    def save_final_model(
        self, model: pl.LightningModule, config: Any, log_dir: str
    ) -> str:
        """Save the final model and return the path."""
        project_name = getattr(config, "project_name", "visual_wake_words")
        final_model_path = Path(log_dir) / project_name / f"{config.name}.pt"
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        return str(final_model_path)
