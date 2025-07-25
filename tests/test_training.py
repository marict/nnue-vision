"""
Comprehensive tests for NNUE and EtinyNet training functionality.

This module tests:
- Training script imports and functionality
- Configuration loading and validation
- Model creation and training compatibility
- Data loader integration
- Full training pipeline integration
- Serialization after training
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import train
from config import ConfigError, load_config
from data import create_data_loaders
from etinynet_adapter import EtinyNetAdapter
from model import NNUE, EtinyNet, GridFeatureSet, LossParams
from nnue_adapter import NNUEWrapper, adapt_batch_for_nnue
from serialize import serialize_etinynet_model, serialize_model


class TestTrainingConfigurations:
    """Test configuration loading and validation."""

    def test_load_nnue_configs(self):
        """Test loading all NNUE training configurations."""
        # Include configs only if file exists
        possible_configs = [
            "config/train_nnue_default.py",
            "config/train_runpod.py",
        ]

        for config_path in possible_configs:
            if not Path(config_path).exists():
                # Skip this particular config but continue testing others
                continue

            config = load_config(config_path)
            assert hasattr(config, "name")
            assert hasattr(config, "batch_size")
            assert hasattr(config, "max_epochs")
            assert hasattr(config, "learning_rate")
            assert config.batch_size > 0
            assert config.max_epochs > 0
            assert config.learning_rate > 0

    def test_load_etinynet_config(self):
        """Test loading EtinyNet-specific configuration."""
        config = load_config("config/train_etinynet.py")

        assert config.name == "etinynet-cifar"
        assert hasattr(config, "etinynet_variant")
        assert config.etinynet_variant in ["0.75", "1.0", "0.98M"]
        assert hasattr(config, "use_asq")
        assert isinstance(config.use_asq, bool)
        assert config.batch_size == 64  # EtinyNet-specific setting
        assert config.max_epochs == 200  # EtinyNet-specific setting

    def test_invalid_config_handling(self):
        """Test error handling for invalid configurations."""
        with pytest.raises(ConfigError):
            load_config("nonexistent_config.py")


class TestModelCreation:
    """Test model creation for both NNUE and EtinyNet."""

    def test_nnue_model_creation(self):
        """Test NNUE model creation with various configurations."""
        # Test with minimal configuration
        feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
        model = NNUE(
            feature_set=feature_set,
            l1_size=64,
            l2_size=4,
            l3_size=8,
            num_ls_buckets=2,
            visual_threshold=0.5,
        )

        assert model.feature_set.num_features == 128  # 4*4*8
        assert model.l1_size == 64
        assert model.num_ls_buckets == 2

        # Test parameter counting
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_etinynet_model_creation(self):
        """Test EtinyNet model creation with all variants."""
        # Test EtinyNet-0.75
        model_075 = EtinyNet(
            variant="0.75", num_classes=10, input_size=32, use_asq=False
        )

        assert model_075.variant == "0.75"
        assert model_075.num_classes == 10
        assert model_075.input_size == 32
        param_count_075 = model_075.count_parameters()

        # Test EtinyNet-0.98M
        model_098 = EtinyNet(
            variant="0.98M", num_classes=10, input_size=32, use_asq=False
        )

        assert model_098.variant == "0.98M"
        param_count_098 = model_098.count_parameters()

        # Test EtinyNet-1.0
        model_10 = EtinyNet(variant="1.0", num_classes=10, input_size=32, use_asq=False)

        assert model_10.variant == "1.0"
        param_count_10 = model_10.count_parameters()

        # Parameter count ordering: 0.98M < 0.75 < 1.0
        # (0.98M is specifically designed to have ~980K params, which is less than 0.75's ~1M)
        assert param_count_098 < param_count_075 < param_count_10

    def test_etinynet_asq_functionality(self):
        """Test EtinyNet with Adaptive Scale Quantization."""
        model = EtinyNet(
            variant="0.75", num_classes=10, input_size=32, use_asq=True, asq_bits=4
        )

        assert model.use_asq is True
        assert hasattr(model, "asq")

        # Test that ASQ doesn't break model functionality
        test_input = torch.randn(2, 3, 32, 32)
        model.train()
        output = model(test_input)
        assert output.shape == (2, 10)
        assert not torch.isnan(output).any()


class TestDataLoaderIntegration:
    """Test data loader integration with models."""

    def test_nnue_data_loader_compatibility(self, fast_data_loaders):
        """Test NNUE model works with data loaders."""
        train_loader, val_loader, test_loader = fast_data_loaders

        # Test batch format adaptation
        batch = next(iter(train_loader))
        adapted_batch = adapt_batch_for_nnue(batch, num_ls_buckets=2)

        assert len(adapted_batch) == 4  # images, targets, scores, layer_stack_indices
        images, targets, scores, layer_stack_indices = adapted_batch

        assert images.shape[0] == targets.shape[0]
        assert scores.shape == targets.shape
        assert layer_stack_indices.shape[0] == images.shape[0]
        assert torch.all(layer_stack_indices >= 0)
        assert torch.all(layer_stack_indices < 2)

    def test_etinynet_data_loader_compatibility(self):
        """Test EtinyNet model works with standard data loaders."""
        # Create data loaders for CIFAR-10 style data with correct target size
        train_loader, val_loader, test_loader = create_data_loaders(
            dataset_name="cifar10",
            batch_size=4,
            num_workers=0,
            max_samples_per_split=8,
            target_size=(32, 32),  # Explicitly set CIFAR-10 size
        )

        batch = next(iter(train_loader))
        images, labels = batch

        assert images.shape[1:] == (3, 32, 32)  # CIFAR-10 format
        assert labels.max() < 10  # Valid CIFAR-10 classes


class TestTrainingScriptFunctionality:
    """Test training script functionality."""

    def test_nnue_wrapper_functionality(self, fast_data_loaders):
        """Test NNUEWrapper correctly adapts data format."""
        feature_set = GridFeatureSet(grid_size=4, num_features_per_square=4)
        nnue_model = NNUE(
            feature_set=feature_set,
            l1_size=32,
            l2_size=4,
            l3_size=8,
            num_ls_buckets=2,
            max_epoch=2,
            lr=1e-3,
            visual_threshold=0.5,
        )

        wrapper = NNUEWrapper(nnue_model)
        assert wrapper.num_ls_buckets == 2

        # Test training step with standard batch format
        train_loader, _, _ = fast_data_loaders
        batch = next(iter(train_loader))

        wrapper.train()
        loss = wrapper.training_step(
            batch, 0
        )  # NNUEWrapper handles adaptation internally

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_etinynet_training_step_functionality(self):
        """Test EtinyNet training step functionality."""
        model = EtinyNet(
            variant="0.75", num_classes=10, input_size=32, lr=0.1, max_epochs=2
        )

        # Create synthetic batch
        batch_size = 4
        images = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        batch = (images, labels)

        # Test training step
        model.train()
        loss = model.training_step(batch, 0)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Test validation step
        val_loss = model.validation_step(batch, 0)
        assert isinstance(val_loss, torch.Tensor)


class TestTrainingIntegration:
    """Test full training pipeline integration."""

    @pytest.mark.timeout(60)  # Prevent hanging tests
    def test_nnue_minimal_training_run(self, tmp_path):
        """Test minimal NNUE training run with mocked data."""
        with patch("nnue_adapter.create_data_loaders") as mock_loaders:
            # Mock tiny data loaders
            mock_dataset = torch.utils.data.TensorDataset(
                torch.randn(8, 3, 96, 96), torch.randint(0, 2, (8,))
            )
            mock_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=2)
            mock_loaders.return_value = (mock_loader, mock_loader, mock_loader)

            # Create minimal model
            feature_set = GridFeatureSet(grid_size=4, num_features_per_square=4)
            nnue_model = NNUE(
                feature_set=feature_set,
                l1_size=32,
                l2_size=4,
                l3_size=8,
                num_ls_buckets=2,
                max_epoch=1,  # Just 1 epoch
                lr=1e-3,
            )

            wrapper = NNUEWrapper(nnue_model)

            # Create minimal trainer
            trainer = pl.Trainer(
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            # Run training
            trainer.fit(wrapper, mock_loader, mock_loader)

            # Verify training completed
            assert trainer.state.finished
            assert not torch.isnan(
                torch.tensor([p.sum() for p in wrapper.parameters()])
            ).any()

    @pytest.mark.timeout(60)  # Prevent hanging tests
    def test_etinynet_minimal_training_run(self, tmp_path):
        """Test minimal EtinyNet training run with mocked data."""
        with patch("etinynet_adapter.create_data_loaders") as mock_loaders:
            # Mock tiny CIFAR-10 style data loaders
            mock_dataset = torch.utils.data.TensorDataset(
                torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
            )
            mock_loader = torch.utils.data.DataLoader(mock_dataset, batch_size=2)
            mock_loaders.return_value = (mock_loader, mock_loader, mock_loader)

            # Create minimal model
            model = EtinyNet(
                variant="0.75",
                num_classes=10,
                input_size=32,
                lr=0.1,
                max_epochs=1,  # Just 1 epoch
            )

            # Create minimal trainer
            trainer = pl.Trainer(
                max_epochs=1,
                accelerator="cpu",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            # Run training
            trainer.fit(model, mock_loader, mock_loader)

            # Verify training completed
            assert trainer.state.finished
            assert not torch.isnan(
                torch.tensor([p.sum() for p in model.parameters()])
            ).any()


class TestSerializationIntegration:
    """Test serialization after training."""

    def test_nnue_serialization_after_training(self, tmp_path):
        """Test NNUE model serialization after training."""
        # Create and train minimal model
        feature_set = GridFeatureSet(grid_size=4, num_features_per_square=4)
        model = NNUE(
            feature_set=feature_set, l1_size=32, l2_size=4, l3_size=8, num_ls_buckets=2
        )

        # Save model state
        model_path = tmp_path / "test_nnue.pt"
        torch.save(model.state_dict(), model_path)

        # Test serialization
        serialized_path = tmp_path / "test_model.nnue"
        serialize_model(model, serialized_path)

        assert serialized_path.exists()
        assert serialized_path.stat().st_size > 0

    def test_etinynet_serialization_after_training(self, tmp_path):
        """Test EtinyNet model serialization after training."""
        # Create minimal model
        model = EtinyNet(variant="0.75", num_classes=10, input_size=32, use_asq=False)

        # Save model state
        model_path = tmp_path / "test_etinynet.pt"
        torch.save(model.state_dict(), model_path)

        # Test serialization
        serialized_path = tmp_path / "test_model.etiny"
        serialize_etinynet_model(model, serialized_path)

        assert serialized_path.exists()
        assert serialized_path.stat().st_size > 0


class TestTrainingScriptImports:
    """Test that training scripts import correctly."""

    def test_train_imports(self):
        """Test that main NNUE training script imports without errors."""
        assert hasattr(train, "main")
        # Test adapter functionality is available through the new modules
        from nnue_adapter import NNUEWrapper, adapt_batch_for_nnue

        assert NNUEWrapper is not None
        assert adapt_batch_for_nnue is not None

    def test_train_etinynet_imports(self):
        """Test that EtinyNet training script imports without errors."""
        assert hasattr(train, "main")


class TestConfigurationValidation:
    """Test configuration validation and parameter compatibility."""

    def test_config_parameter_types(self):
        """Test that configuration parameters have correct types."""
        config = load_config("config/train_etinynet.py")

        assert isinstance(config.batch_size, int)
        assert isinstance(config.max_epochs, int)
        assert isinstance(config.learning_rate, (int, float))
        assert isinstance(config.use_asq, bool)

    def test_model_config_compatibility(self):
        """Test that model configurations are compatible with models."""
        config = load_config("config/train_etinynet.py")

        # Test that EtinyNet can be created with config parameters
        model = EtinyNet(
            variant=getattr(config, "etinynet_variant", "0.75"),
            num_classes=10,  # CIFAR-10
            input_size=32,
            use_asq=getattr(config, "use_asq", False),
            lr=config.learning_rate,
            max_epochs=config.max_epochs,
        )

        assert model.lr == config.learning_rate
        assert model.max_epochs == config.max_epochs
