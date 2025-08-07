#!/usr/bin/env python3
"""
Actual training tests for NNUE and EtinyNet models.
These tests run real training for a few epochs to verify the training pipeline works.
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, ".")

import torch
import torch.nn as nn

from config.config_loader import load_config
from data.loaders import create_data_loaders
from nnue import NNUE, EtinyNet, GridFeatureSet
from train import train_model


def test_nnue_actual_training():
    """Test actual NNUE training for a few epochs."""

    print("=== Testing NNUE Actual Training ===")

    # Create a proper config object for testing
    config = SimpleNamespace(
        name="nnue_test",
        project_name="nnue-test",
        log_dir="./logs",
        grid_size=4,
        num_features_per_square=2,
        l1_size=32,
        l2_size=16,
        l3_size=4,
        num_classes=10,
        input_size=32,
        etinynet_variant="micro",
        max_epochs=2,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0001,
        dataset_name="cifar10",
        max_samples_per_split=20,
        use_augmentation=False,
        augmentation_strength="medium",
        num_workers=0,
        subset=1.0,
        compiled_evaluation_enabled=True,
        compiled_evaluation_frequency=1,
        checkpointing_save_best=True,
        checkpointing_save_frequency=1,
        max_grad_norm=1.0,
        optimizer_type="adam",
        momentum=0.9,
    )

    # Mock wandb to avoid external dependencies
    with patch("train.wandb") as mock_wandb:
        mock_wandb.init.return_value = Mock()
        mock_wandb.run = Mock()
        mock_wandb.run.name = "test-run"
        mock_wandb.run.url = "https://test.wandb.ai/test"
        mock_wandb.log = Mock()
        mock_wandb.save = Mock()

        # Mock C++ compilation to avoid build issues
        with patch("train.compile_cpp_engine") as mock_compile:
            mock_compile.return_value = True
            with patch("train.test_cpp_engine_inference") as mock_test:
                mock_test.return_value = True

                # Run training
                result = train_model(config, "nnue")
                print(f"✅ NNUE training completed with result: {result}")
                assert result == 0, f"NNUE training failed with result {result}"


def test_etinynet_actual_training():
    """Test actual EtinyNet training for a few epochs."""

    print("\n=== Testing EtinyNet Actual Training ===")

    # Create a proper config object for testing
    config = SimpleNamespace(
        name="etinynet_test",
        project_name="etinynet-test",
        log_dir="./logs",
        grid_size=4,
        num_features_per_square=2,
        l1_size=32,
        l2_size=16,
        l3_size=4,
        num_classes=10,
        input_size=32,
        etinynet_variant="micro",
        max_epochs=2,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0001,
        dataset_name="cifar10",
        max_samples_per_split=20,
        use_augmentation=False,
        augmentation_strength="medium",
        num_workers=0,
        subset=1.0,
        compiled_evaluation_enabled=True,
        compiled_evaluation_frequency=1,
        checkpointing_save_best=True,
        checkpointing_save_frequency=1,
        max_grad_norm=1.0,
        optimizer_type="adam",
        momentum=0.9,
    )

    # Mock wandb to avoid external dependencies
    with patch("train.wandb") as mock_wandb:
        mock_wandb.init.return_value = Mock()
        mock_wandb.run = Mock()
        mock_wandb.run.name = "test-run"
        mock_wandb.run.url = "https://test.wandb.ai/test"
        mock_wandb.log = Mock()
        mock_wandb.save = Mock()

        # Mock C++ compilation to avoid build issues
        with patch("train.compile_cpp_engine") as mock_compile:
            mock_compile.return_value = True
            with patch("train.test_cpp_engine_inference") as mock_test:
                mock_test.return_value = True

                # Run training
                result = train_model(config, "etinynet")
                print(f"✅ EtinyNet training completed with result: {result}")
                assert result == 0, f"EtinyNet training failed with result {result}"


if __name__ == "__main__":
    print("🚀 Running actual training tests...")

    nnue_success = test_nnue_actual_training()
    etinynet_success = test_etinynet_actual_training()

    if nnue_success and etinynet_success:
        print("\n✅ All training tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some training tests failed!")
        sys.exit(1)
