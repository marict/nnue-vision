#!/usr/bin/env python3
"""
Quantized NNUE-Vision Training Script

This script trains a quantized NNUE model with proper quantization-aware training (QAT)
for minimal model size and maximum efficiency.

Usage:
    python train_quantized.py [--config CONFIG_PATH]

Features:
- Quantization-aware training (QAT)
- Model size optimization
- Efficient inference optimization
- Comprehensive quantization metrics
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.quantization as quant
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         TQDMProgressBar)
from pytorch_lightning.loggers import WandbLogger
from quantized_model import QuantizedModelParams, QuantizedNNUE

from config import ConfigError, load_config
from dataset import create_data_loaders


class QuantizationCallback(pl.Callback):
    """Custom callback for quantization-aware training."""

    def __init__(self, qat_start_epoch: int = 5):
        super().__init__()
        self.qat_start_epoch = qat_start_epoch
        self.qat_enabled = False

    def on_train_epoch_start(self, trainer, pl_module):
        """Enable QAT after specified epoch."""
        if trainer.current_epoch >= self.qat_start_epoch and not self.qat_enabled:
            print(
                f"\n🔧 Enabling Quantization-Aware Training at epoch {trainer.current_epoch}"
            )
            pl_module.prepare_for_quantization()
            self.qat_enabled = True

            if wandb.run is not None:
                wandb.log(
                    {
                        "qat/enabled_epoch": trainer.current_epoch,
                        "qat/status": "enabled",
                    }
                )


class QuantizedMetricsCallback(pl.Callback):
    """Callback for logging quantized model metrics."""

    def __init__(self):
        super().__init__()
        self.train_start_time = None

    def on_train_start(self, trainer, pl_module):
        """Log initial quantized model information."""
        self.train_start_time = time.time()

        # Get model size information
        size_info = pl_module.get_model_size_info()

        if wandb.run is not None:
            wandb.log(
                {
                    "model/total_parameters": size_info["total_parameters"],
                    "model/fp32_size_kb": size_info["fp32_size_kb"],
                    "model/estimated_int8_size_kb": size_info["int8_size_kb"],
                    "model/estimated_mixed_precision_size_kb": size_info[
                        "mixed_precision_size_kb"
                    ],
                    "model/compression_ratio_int8": size_info["compression_ratio_int8"],
                    "model/compression_ratio_mixed": size_info[
                        "compression_ratio_mixed"
                    ],
                }
            )

        print(f"\n📊 Quantized Model Information:")
        print(f"   Total Parameters: {size_info['total_parameters']:,}")
        print(f"   FP32 Size: {size_info['fp32_size_kb']:.1f} KB")
        print(f"   Estimated Int8 Size: {size_info['int8_size_kb']:.1f} KB")
        print(
            f"   Estimated Mixed Precision Size: {size_info['mixed_precision_size_kb']:.1f} KB"
        )
        print(
            f"   Compression Ratio (Int8): {size_info['compression_ratio_int8']:.1f}x"
        )
        print(
            f"   Compression Ratio (Mixed): {size_info['compression_ratio_mixed']:.1f}x"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        """Log epoch-end quantization metrics."""
        if wandb.run is not None:
            # Log learning rate
            current_lr = trainer.optimizers[0].param_groups[0]["lr"]
            wandb.log(
                {
                    "train/learning_rate": current_lr,
                    "train/epoch": trainer.current_epoch,
                }
            )


def setup_wandb_logger(config, model_params) -> WandbLogger:
    """Set up wandb logger for quantized training."""

    # Generate run name
    run_name = (
        f"qnnue-lr{config.learning_rate}-bs{config.batch_size}-"
        f"fd{config.feature_dim}-hd{config.hidden_dim}-{config.weight_bits}bit"
    )

    # Create wandb config
    wandb_config = {
        # Model parameters
        "model/learning_rate": model_params.learning_rate,
        "model/input_size": model_params.input_size,
        "model/num_classes": model_params.num_classes,
        "model/feature_dim": model_params.feature_dim,
        "model/hidden_dim": model_params.hidden_dim,
        "model/ft_scale": model_params.ft_scale,
        "model/hidden_scale": model_params.hidden_scale,
        "model/output_scale": model_params.output_scale,
        "model/weight_bits": model_params.weight_bits,
        "model/activation_bits": model_params.activation_bits,
        "model/hidden_bits": model_params.hidden_bits,
        # Training parameters
        "train/batch_size": config.batch_size,
        "train/max_epochs": config.max_epochs,
        "train/image_size": config.image_size,
        "train/num_workers": config.num_workers,
        "train/qat_start_epoch": getattr(config, "qat_start_epoch", 5),
        # Quantization parameters
        "quantization/backend": getattr(config, "quantization_backend", "fbgemm"),
        "quantization/observer_type": getattr(config, "observer_type", "minmax"),
        "quantization/qscheme": getattr(config, "qscheme", "per_tensor_symmetric"),
        # System parameters
        "system/cuda_available": torch.cuda.is_available(),
        "system/torch_version": torch.__version__,
        "system/device_count": (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        ),
    }

    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=run_name,
        config=wandb_config,
        save_dir=config.log_dir,
        log_model=True,
        tags=["quantized", "nnue", "efficient", "mobile-optimized"],
    )

    return wandb_logger


def save_quantized_models(model: QuantizedNNUE, config, log_dir: str):
    """Save quantized model in multiple formats."""

    project_name = getattr(config, "project_name", "visual_wake_words_quantized")
    models_dir = Path(log_dir) / project_name / "quantized_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving quantized models to {models_dir}")

    # 1. Save QAT model state dict
    qat_path = models_dir / f"{config.name}_qat.pt"
    torch.save(model.state_dict(), qat_path)
    print(f"   ✅ QAT model saved: {qat_path}")

    # 2. Convert to fully quantized model
    try:
        model.eval()
        quantized_model = model.convert_to_quantized()

        # Save quantized model
        quantized_path = models_dir / f"{config.name}_quantized.pt"
        torch.save(quantized_model.state_dict(), quantized_path)
        print(f"   ✅ Quantized model saved: {quantized_path}")

        # 3. Save as TorchScript (JIT)
        if getattr(config, "export_jit", True):
            try:
                # Create example input
                example_input = torch.randn(
                    1, 3, config.input_size[0], config.input_size[1]
                )
                traced_model = torch.jit.trace(quantized_model, example_input)

                jit_path = models_dir / f"{config.name}_quantized.jit"
                traced_model.save(str(jit_path))
                print(f"   ✅ JIT model saved: {jit_path}")

                # Check file sizes
                qat_size = qat_path.stat().st_size / 1024  # KB
                quantized_size = quantized_path.stat().st_size / 1024  # KB
                jit_size = jit_path.stat().st_size / 1024  # KB

                print(f"\n📏 Model Sizes:")
                print(f"   QAT Model: {qat_size:.1f} KB")
                print(f"   Quantized Model: {quantized_size:.1f} KB")
                print(f"   JIT Model: {jit_size:.1f} KB")
                print(f"   Compression Ratio: {qat_size / quantized_size:.1f}x")

                if wandb.run is not None:
                    wandb.log(
                        {
                            "final/qat_model_size_kb": qat_size,
                            "final/quantized_model_size_kb": quantized_size,
                            "final/jit_model_size_kb": jit_size,
                            "final/compression_ratio": qat_size / quantized_size,
                        }
                    )

                    # Save models as wandb artifacts
                    artifact = wandb.Artifact("quantized_models", type="model")
                    artifact.add_file(str(qat_path))
                    artifact.add_file(str(quantized_path))
                    artifact.add_file(str(jit_path))
                    wandb.log_artifact(artifact)

            except Exception as e:
                print(f"   ⚠️  JIT export failed: {e}")

        # 4. Export to ONNX (optional)
        if getattr(config, "export_onnx", False):
            try:
                import torch.onnx

                example_input = torch.randn(
                    1, 3, config.input_size[0], config.input_size[1]
                )
                onnx_path = models_dir / f"{config.name}_quantized.onnx"

                torch.onnx.export(
                    quantized_model,
                    example_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )
                print(f"   ✅ ONNX model saved: {onnx_path}")

            except Exception as e:
                print(f"   ⚠️  ONNX export failed: {e}")

    except Exception as e:
        print(f"   ⚠️  Quantized model conversion failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantized NNUE-Vision training with QAT support"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_quantized.py",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    try:
        print(f"📋 Loading configuration from: {args.config}")
        config = load_config(args.config)
        print(f"✅ Configuration loaded: {config.name}")
    except ConfigError as e:
        print(f"❌ Error loading configuration: {e}")
        return 1

    # Check wandb setup
    use_wandb = getattr(config, "use_wandb", False)
    if use_wandb and not os.getenv("WANDB_API_KEY"):
        print(
            "⚠️  Warning: WANDB_API_KEY not found. Wandb logging may not work properly."
        )
        print("Set the environment variable or use --wandb_api_key argument.")

    # Set random seed for reproducibility
    pl.seed_everything(getattr(config, "seed", 42))

    # Create quantized model parameters
    model_params = QuantizedModelParams(
        input_size=getattr(config, "input_size", (96, 96)),
        num_classes=getattr(config, "num_classes", 2),
        learning_rate=getattr(config, "learning_rate", 8e-4),
        feature_dim=getattr(config, "feature_dim", 128),
        hidden_dim=getattr(config, "hidden_dim", 64),
        ft_scale=getattr(config, "ft_scale", 127),
        hidden_scale=getattr(config, "hidden_scale", 127),
        output_scale=getattr(config, "output_scale", 361),
        weight_bits=getattr(config, "weight_bits", 8),
        activation_bits=getattr(config, "activation_bits", 8),
        hidden_bits=getattr(config, "hidden_bits", 16),
    )

    # Create quantized model
    print("🧠 Creating quantized NNUE model...")
    model = QuantizedNNUE(model_params)
    size_info = model.get_model_size_info()
    print(f"   Parameters: {size_info['total_parameters']:,}")
    print(f"   Estimated quantized size: {size_info['mixed_precision_size_kb']:.1f} KB")

    # Create data loaders
    print("📊 Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=getattr(config, "batch_size", 64),
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

    # Set up callbacks
    callbacks = [
        TQDMProgressBar(refresh_rate=getattr(config, "log_interval", 25)),
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
            patience=getattr(config, "patience", 15),
            verbose=True,
        ),
        QuantizationCallback(qat_start_epoch=getattr(config, "qat_start_epoch", 5)),
        QuantizedMetricsCallback(),
    ]

    # Set up trainer
    devices = getattr(config, "devices", "auto")
    if isinstance(devices, str) and "," in devices:
        devices = devices.split(",")

    trainer = pl.Trainer(
        max_epochs=getattr(config, "max_epochs", 60),
        accelerator=getattr(config, "accelerator", "auto"),
        devices=devices,
        logger=loggers if loggers else False,
        callbacks=callbacks,
        log_every_n_steps=getattr(config, "log_interval", 25),
        enable_checkpointing=getattr(config, "always_save_checkpoint", True),
        enable_progress_bar=getattr(config, "enable_progress_bar", True),
        deterministic=getattr(config, "deterministic", True),
        check_val_every_n_epoch=getattr(config, "check_val_every_n_epoch", 1),
    )

    # Log training start
    print(f"\n🚀 Starting quantized NNUE training...")
    print(f"   Configuration: {config.name}")
    print(f"   Max epochs: {getattr(config, 'max_epochs', 60)}")
    print(f"   Batch size: {getattr(config, 'batch_size', 64)}")
    print(f"   QAT start epoch: {getattr(config, 'qat_start_epoch', 5)}")
    print(f"   Target model size: {getattr(config, 'target_model_size_kb', 50)} KB")

    if use_wandb and loggers:
        wandb_logger = next(
            (logger for logger in loggers if isinstance(logger, WandbLogger)), None
        )
        if wandb_logger:
            print(
                f"   Wandb project: {getattr(config, 'project_name', 'visual_wake_words_quantized')}"
            )
            print(f"   Wandb run URL: {wandb_logger.experiment.url}")

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    print("\n🧪 Testing the quantized model...")
    test_results = trainer.test(model, test_loader)

    # Log final results
    final_acc = test_results[0]["test_acc"]
    print(f"\n✅ Training completed!")
    print(f"   Final test accuracy: {final_acc:.3f}")

    if use_wandb and test_results:
        wandb.log(
            {
                "final/test_loss": test_results[0]["test_loss"],
                "final/test_acc": test_results[0]["test_acc"],
                "final/test_precision": test_results[0]["test_precision"],
                "final/test_recall": test_results[0]["test_recall"],
                "final/test_f1": test_results[0]["test_f1"],
            }
        )

    # Save quantized models
    save_quantized_models(model, config, log_dir)

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    print("\n🎉 Quantized NNUE training completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
