"""
Tests for the training framework components.

This module tests the core training framework functionality including:
- WandbMetricsCallback
- BaseTrainer methods
- Utility functions
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch

from etinynet_adapter import EtinyNetAdapter
from nnue_adapter import NNUEAdapter, adapt_batch_for_nnue
from training_framework import BaseTrainer, WandbMetricsCallback


class TestWandbMetricsCallback:
    """Test the WandbMetricsCallback functionality."""

    def test_callback_creation(self):
        """Test that callback can be created."""
        callback = WandbMetricsCallback()
        assert callback.train_start_time is None
        assert callback.epoch_start_time is None
        assert callback.step_start_time is None

    def test_callback_timing_methods(self, monkeypatch):
        """Test callback timing methods."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("training_framework.wandb", mock_wandb)

        callback = WandbMetricsCallback()

        # Mock trainer and pl_module
        trainer = MagicMock()
        trainer.optimizers = [MagicMock()]
        trainer.optimizers[0].param_groups = [{"lr": 0.001}]
        trainer.global_step = 10

        pl_module = MagicMock()
        pl_module.parameters.return_value = [
            torch.tensor([1.0, 2.0], requires_grad=True),
            torch.tensor([3.0, 4.0], requires_grad=True),
        ]

        # Test train start
        callback.on_train_start(trainer, pl_module)
        assert callback.train_start_time is not None
        assert mock_wandb.log.called

        # Test epoch start
        callback.on_train_epoch_start(trainer, pl_module)
        assert callback.epoch_start_time is not None

        # Test batch start
        callback.on_train_batch_start(trainer, pl_module, None, 0)
        assert callback.step_start_time is not None

    def test_callback_gpu_logging(self, monkeypatch):
        """Test GPU information logging when CUDA is available."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("training_framework.wandb", mock_wandb)

        # Mock CUDA availability
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
        monkeypatch.setattr("torch.cuda.get_device_name", lambda i: "Test GPU")
        mock_props = MagicMock()
        mock_props.total_memory = 8 * 1024**3  # 8GB
        monkeypatch.setattr("torch.cuda.get_device_properties", lambda i: mock_props)

        callback = WandbMetricsCallback()
        trainer = MagicMock()
        pl_module = MagicMock()
        pl_module.parameters.return_value = []

        callback.on_train_start(trainer, pl_module)

        # Check that GPU info was logged
        log_calls = [call[0][0] for call in mock_wandb.log.call_args_list]
        gpu_logged = any("system/gpu_0_name" in call for call in log_calls)
        assert gpu_logged


class TestAdaptBatchForNNUE:
    """Test the adapt_batch_for_nnue function."""

    def test_adapt_batch_basic(self):
        """Test basic batch adaptation for NNUE."""
        # Create test batch
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 2, (4,))
        batch = (images, labels)

        # Adapt batch
        adapted_images, targets, scores, layer_stack_indices = adapt_batch_for_nnue(
            batch, num_ls_buckets=8
        )

        # Check outputs
        assert torch.equal(adapted_images, images)
        assert targets.dtype == torch.float32
        assert targets.shape == labels.shape
        assert scores.shape == targets.shape
        assert torch.all(scores == 0.0)  # Should be zeros for synthetic scores
        assert layer_stack_indices.shape == (4,)
        assert torch.all(layer_stack_indices >= 0)
        assert torch.all(layer_stack_indices < 8)

    def test_adapt_batch_different_bucket_count(self):
        """Test batch adaptation with different bucket counts."""
        images = torch.randn(2, 3, 32, 32)
        labels = torch.randint(0, 2, (2,))
        batch = (images, labels)

        for num_buckets in [1, 4, 16]:
            _, _, _, layer_stack_indices = adapt_batch_for_nnue(
                batch, num_ls_buckets=num_buckets
            )
            assert torch.all(layer_stack_indices >= 0)
            assert torch.all(layer_stack_indices < num_buckets)

    def test_adapt_batch_device_consistency(self):
        """Test that adapted batch maintains device consistency."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        images = torch.randn(2, 3, 32, 32, device=device)
        labels = torch.randint(0, 2, (2,), device=device)
        batch = (images, labels)

        adapted_images, targets, scores, layer_stack_indices = adapt_batch_for_nnue(
            batch
        )

        assert adapted_images.device == device
        assert targets.device == device
        assert scores.device == device
        assert layer_stack_indices.device == device


class TestBaseTrainerMethods:
    """Test BaseTrainer individual methods."""

    def test_setup_wandb_logger(self, monkeypatch):
        """Test wandb logger setup."""
        # Mock WandbLogger
        mock_logger = MagicMock()
        monkeypatch.setattr(
            "training_framework.WandbLogger", lambda **kwargs: mock_logger
        )

        adapter = NNUEAdapter()
        trainer = BaseTrainer(adapter)

        # Create mock config with all required attributes
        config = SimpleNamespace(
            learning_rate=0.001,
            batch_size=32,
            max_epochs=10,
            log_dir="test_logs",
            project_name="test_project",
            accelerator="cpu",
            patience=10,
            save_top_k=3,
            name="test_model",
        )

        logger = trainer.setup_wandb_logger(config)
        assert logger == mock_logger

    def test_setup_trainer(self):
        """Test PyTorch Lightning trainer setup."""
        adapter = NNUEAdapter()
        trainer = BaseTrainer(adapter)

        # Create mock config
        config = SimpleNamespace(
            max_epochs=10,
            accelerator="cpu",
            devices=1,
            log_interval=50,
            always_save_checkpoint=True,
            enable_progress_bar=True,
            deterministic=True,
            check_val_every_n_epoch=1,
        )

        loggers = []
        callbacks = []

        pl_trainer = trainer.setup_trainer(config, loggers, callbacks)
        assert isinstance(pl_trainer, pl.Trainer)
        assert pl_trainer.max_epochs == 10

    def test_load_and_setup_config(self, tmp_path, monkeypatch):
        """Test config loading and setup."""
        # Create temporary config file
        config_file = tmp_path / "test_config.py"
        config_file.write_text(
            """
name = "test_config"
batch_size = 16
max_epochs = 5
learning_rate = 0.002
"""
        )

        adapter = NNUEAdapter()
        trainer = BaseTrainer(adapter)

        # Create mock args
        args = SimpleNamespace(
            config=str(config_file),
            batch_size=32,  # Override
            max_epochs=None,
            learning_rate=None,
            note="test_note",
            log_dir="test_logs",
        )

        # Mock early_log to avoid output
        monkeypatch.setattr("training_framework.early_log", lambda x: None)

        config = trainer.load_and_setup_config(args)

        # Check that overrides were applied
        assert config.name == "test_config"
        assert config.batch_size == 32  # Overridden
        assert config.max_epochs == 5  # From file
        assert config.learning_rate == 0.002  # From file
        assert config.note == "test_note"
        assert config.log_dir == "test_logs"


class TestModelAdapterIntegration:
    """Test model adapter integration with training framework."""

    def test_nnue_adapter_with_base_trainer(self):
        """Test NNUE adapter integration with BaseTrainer."""
        adapter = NNUEAdapter()
        trainer = BaseTrainer(adapter)

        # Test argument parser setup
        parser = trainer.setup_argument_parser()
        assert "Train NNUE model" in parser.description

        # Test that parser has common arguments
        help_text = parser.format_help()
        assert "--config" in help_text
        assert "--batch_size" in help_text
        assert "--learning_rate" in help_text

    def test_etinynet_adapter_with_base_trainer(self):
        """Test EtinyNet adapter integration with BaseTrainer."""
        adapter = EtinyNetAdapter()
        trainer = BaseTrainer(adapter)

        # Test argument parser setup
        parser = trainer.setup_argument_parser()
        assert "Train EtinyNet model" in parser.description

        # Test that parser has both common and model-specific arguments
        help_text = parser.format_help()
        assert "--config" in help_text
        assert "--variant" in help_text
        assert "--dataset" in help_text

    def test_adapter_wandb_config_generation(self):
        """Test that adapters generate proper wandb configs."""
        # Test NNUE adapter
        nnue_adapter = NNUEAdapter()
        nnue_config = SimpleNamespace(
            learning_rate=0.001,
            batch_size=32,
            max_epochs=10,
            accelerator="cpu",
            patience=5,
            save_top_k=3,
            name="test_nnue",
        )
        wandb_config = nnue_adapter.setup_wandb_config(nnue_config)
        assert wandb_config["model/learning_rate"] == 0.001
        assert wandb_config["train/batch_size"] == 32

        # Test EtinyNet adapter
        etinynet_adapter = EtinyNetAdapter()
        etinynet_config = SimpleNamespace(
            learning_rate=0.1,
            batch_size=64,
            max_epochs=200,
            accelerator="cpu",
            patience=15,
            save_top_k=3,
            name="test_etinynet",
            etinynet_variant="1.0",
            dataset_name="cifar10",
        )
        wandb_config = etinynet_adapter.setup_wandb_config(etinynet_config)
        assert wandb_config["model/learning_rate"] == 0.1
        assert wandb_config["model/variant"] == "1.0"
        assert wandb_config["train/dataset"] == "cifar10"


class TestErrorHandling:
    """Test error handling in training framework."""

    def test_config_loading_error(self, tmp_path):
        """Test handling of config loading errors."""
        # Create invalid config file
        config_file = tmp_path / "invalid_config.py"
        config_file.write_text("invalid python syntax <<<")

        adapter = NNUEAdapter()
        trainer = BaseTrainer(adapter)

        args = SimpleNamespace(
            config=str(config_file),
            batch_size=None,
            max_epochs=None,
            learning_rate=None,
            note=None,
            log_dir="logs",
        )

        with pytest.raises(Exception):  # Should raise some kind of config error
            trainer.load_and_setup_config(args)

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        adapter = NNUEAdapter()
        trainer = BaseTrainer(adapter)

        args = SimpleNamespace(
            config="nonexistent_config.py",
            batch_size=None,
            max_epochs=None,
            learning_rate=None,
            note=None,
            log_dir="logs",
        )

        with pytest.raises(Exception):  # Should raise FileNotFoundError or similar
            trainer.load_and_setup_config(args)
