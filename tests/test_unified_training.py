"""
Tests for the unified training framework and model adapters.

These tests verify that the new unified training system works correctly
for both NNUE and EtinyNet models.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import train
from etinynet_adapter import EtinyNetAdapter
from nnue_adapter import NNUEAdapter
from training_framework import BaseTrainer


class DummyWandbLogger:
    """Minimal stub for Lightning's WandbLogger used in tests."""

    def __init__(self, *args, **kwargs):
        # Provide just the attributes accessed in the training script
        self.experiment = SimpleNamespace(
            config={},
            url="http://wandb.local/run",
        )
        self.save_dir = "."
        self.version = "test"

    # Minimal API surface used by Lightning
    def log_metrics(self, *args, **kwargs):
        pass

    def log_hyperparams(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        pass

    def log_graph(self, *args, **kwargs):
        pass

    def save(self):
        pass

    # Gracefully handle any other method/attribute requests
    def __getattr__(self, item):
        def _dummy(*args, **kwargs):
            return None

        return _dummy


@pytest.fixture
def setup_test_env(monkeypatch, tmp_path):
    """Set up test environment with wandb disabled and temp directories."""
    # Disable wandb
    monkeypatch.setenv("WANDB_API_KEY", "dummy_test_key")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    # Create test output directory
    test_outputs = tmp_path / "test_outputs"
    test_outputs.mkdir(exist_ok=True)

    return test_outputs


@pytest.fixture
def mock_wandb_components(monkeypatch):
    """Mock all wandb-related components to avoid network calls."""
    # Mock wandb module
    mock_wandb = SimpleNamespace(
        init=lambda *_, **__: SimpleNamespace(url="local", id="test_id"),
        log=lambda *_1, **_2: None,
        finish=lambda *_a, **_kw: None,
        Artifact=lambda *_, **__: SimpleNamespace(add_file=lambda x: None),
        log_artifact=lambda x: None,
        Image=lambda *_, **__: "mock_image",
    )

    return mock_wandb


def create_tiny_synthetic_loaders():
    """Create tiny synthetic data loaders for testing."""
    # NNUE-style data (Visual Wake Words - 96x96 RGB, binary classification)
    nnue_dataset = torch.utils.data.TensorDataset(
        torch.randn(4, 3, 32, 32),  # Using 32x32 for speed (config will handle)
        torch.randint(0, 2, (4,)),  # Binary classification
    )
    nnue_loader = torch.utils.data.DataLoader(nnue_dataset, batch_size=2)

    # EtinyNet-style data (CIFAR-10 - 32x32 RGB, 10 classes)
    etinynet_dataset = torch.utils.data.TensorDataset(
        torch.randn(4, 3, 32, 32),  # CIFAR-10 size
        torch.randint(0, 10, (4,)),  # 10 classes
    )
    etinynet_loader = torch.utils.data.DataLoader(etinynet_dataset, batch_size=2)

    return nnue_loader, etinynet_loader


class TestModelAdapters:
    """Test the model adapter classes."""

    def test_nnue_adapter_creation(self):
        """Test NNUE adapter can be created and has correct interface."""
        adapter = NNUEAdapter()
        assert adapter.get_model_type_name() == "NNUE"
        assert adapter.get_default_config_path() == "config/train_nnue_default.py"

    def test_etinynet_adapter_creation(self):
        """Test EtinyNet adapter can be created and has correct interface."""
        adapter = EtinyNetAdapter()
        assert adapter.get_model_type_name() == "EtinyNet"
        assert adapter.get_default_config_path() == "config/train_etinynet_default.py"

    def test_adapter_interface_completeness(self):
        """Test that both adapters implement the full ModelAdapter interface."""
        adapters = [NNUEAdapter(), EtinyNetAdapter()]

        required_methods = [
            "get_model_type_name",
            "create_model",
            "create_data_loaders",
            "get_callbacks",
            "setup_wandb_config",
            "get_model_specific_args",
            "apply_model_specific_overrides",
            "log_sample_predictions",
            "save_final_model",
            "get_default_config_path",
            "get_run_name",
        ]

        for adapter in adapters:
            for method in required_methods:
                assert hasattr(
                    adapter, method
                ), f"{adapter.__class__.__name__} missing {method}"
                assert callable(
                    getattr(adapter, method)
                ), f"{adapter.__class__.__name__}.{method} not callable"


class TestBaseTrainer:
    """Test the base trainer functionality."""

    def test_base_trainer_creation(self):
        """Test that BaseTrainer can be created with adapters."""
        nnue_adapter = NNUEAdapter()
        trainer = BaseTrainer(nnue_adapter)
        assert trainer.adapter == nnue_adapter

    def test_argument_parser_setup(self):
        """Test that argument parser is set up correctly for both models."""
        # Test NNUE adapter
        nnue_adapter = NNUEAdapter()
        nnue_trainer = BaseTrainer(nnue_adapter)
        nnue_parser = nnue_trainer.setup_argument_parser()

        # Test EtinyNet adapter
        etinynet_adapter = EtinyNetAdapter()
        etinynet_trainer = BaseTrainer(etinynet_adapter)
        etinynet_parser = etinynet_trainer.setup_argument_parser()

        # Both should have common arguments
        common_args = [
            "--config",
            "--batch_size",
            "--max_epochs",
            "--learning_rate",
            "--note",
        ]
        for parser in [nnue_parser, etinynet_parser]:
            help_text = parser.format_help()
            for arg in common_args:
                assert arg in help_text


class TestUnifiedTrainingScript:
    """Test the unified training script functionality."""

    @pytest.mark.timeout(30)  # Should complete in under 30 seconds
    def test_unified_nnue_training(
        self, setup_test_env, mock_wandb_components, monkeypatch
    ):
        """Test that unified script can train NNUE models."""
        test_outputs = setup_test_env

        # Mock external dependencies in training_framework
        import training_framework

        monkeypatch.setattr(training_framework, "WandbLogger", DummyWandbLogger)
        monkeypatch.setattr(training_framework, "wandb", mock_wandb_components)
        monkeypatch.setattr(
            training_framework, "replay_early_logs_to_wandb", lambda *_a, **_kw: None
        )
        monkeypatch.setattr(
            training_framework, "log_git_info_to_wandb", lambda *_a, **_kw: None
        )

        # Mock external dependencies in nnue_adapter
        import nnue_adapter

        monkeypatch.setattr(nnue_adapter, "wandb", mock_wandb_components)

        # Mock test method to avoid metric requirements
        from pytorch_lightning import Trainer as _PLTrainer

        monkeypatch.setattr(
            _PLTrainer,
            "test",
            lambda self, *a, **kw: [{"test_loss": 0.5, "test_acc": 0.8}],
        )

        # Create synthetic data loaders
        nnue_loader, _ = create_tiny_synthetic_loaders()

        with patch("nnue_adapter.create_data_loaders") as mock_loaders:
            mock_loaders.return_value = (nnue_loader, nnue_loader, nnue_loader)

            # Set up argv for unified script
            config_path = "config/train_nnue_default.py"
            argv = [
                "train.py",
                "nnue",
                "--config",
                config_path,
                "--log_dir",
                str(test_outputs),
            ]

            with patch.object(sys, "argv", argv):
                # This should complete without error (sys.exit(0) is expected)
                with pytest.raises(SystemExit) as exc_info:
                    train.main()
                assert (
                    exc_info.value.code == 0
                ), f"Expected successful exit, got: {exc_info.value.code}"

        # Verify some output was created
        assert len(list(test_outputs.rglob("*"))) > 0, "No output files were created"

    @pytest.mark.timeout(30)  # Should complete in under 30 seconds
    def test_unified_etinynet_training(
        self, setup_test_env, mock_wandb_components, monkeypatch
    ):
        """Test that unified script can train EtinyNet models."""
        test_outputs = setup_test_env

        # Mock external dependencies in training_framework
        import training_framework

        monkeypatch.setattr(training_framework, "WandbLogger", DummyWandbLogger)
        monkeypatch.setattr(training_framework, "wandb", mock_wandb_components)
        monkeypatch.setattr(
            training_framework, "replay_early_logs_to_wandb", lambda *_a, **_kw: None
        )
        monkeypatch.setattr(
            training_framework, "log_git_info_to_wandb", lambda *_a, **_kw: None
        )

        # Mock test method to avoid metric requirements
        from pytorch_lightning import Trainer as _PLTrainer

        monkeypatch.setattr(
            _PLTrainer,
            "test",
            lambda self, *a, **kw: [{"test_loss": 0.5, "test_acc": 0.8}],
        )

        # Create synthetic data loaders
        _, etinynet_loader = create_tiny_synthetic_loaders()

        with patch("etinynet_adapter.create_data_loaders") as mock_loaders:
            mock_loaders.return_value = (
                etinynet_loader,
                etinynet_loader,
                etinynet_loader,
            )

            # Set up argv for unified script
            config_path = "config/train_etinynet_default.py"
            argv = [
                "train.py",
                "etinynet",
                "--config",
                config_path,
                "--log_dir",
                str(test_outputs),
            ]

            with patch.object(sys, "argv", argv):
                # This should complete without error (sys.exit(0) is expected)
                with pytest.raises(SystemExit) as exc_info:
                    train.main()
                assert (
                    exc_info.value.code == 0
                ), f"Expected successful exit, got: {exc_info.value.code}"

        # Verify some output was created
        assert len(list(test_outputs.rglob("*"))) > 0, "No output files were created"

    def test_unified_script_error_handling(self, monkeypatch):
        """Test that unified script handles errors correctly."""
        # Test missing model type
        with patch.object(sys, "argv", ["train.py"]):
            with pytest.raises(SystemExit) as exc_info:
                train.main()
            assert exc_info.value.code == 1

        # Test invalid model type
        with patch.object(sys, "argv", ["train.py", "invalid_model"]):
            with pytest.raises(SystemExit) as exc_info:
                train.main()
            assert exc_info.value.code == 1

    def test_available_adapters(self):
        """Test that get_available_adapters returns expected adapters."""
        adapters = train.get_available_adapters()
        assert "nnue" in adapters
        assert "etinynet" in adapters
        assert isinstance(adapters["nnue"], NNUEAdapter)
        assert isinstance(adapters["etinynet"], EtinyNetAdapter)


class TestArgumentParsing:
    """Test argument parsing for both model types."""

    def test_nnue_argument_parsing(self, monkeypatch):
        """Test NNUE-specific argument parsing."""
        adapter = NNUEAdapter()
        trainer = BaseTrainer(adapter)

        # Mock sys.argv for testing
        test_args = [
            "train.py",
            "--config",
            "config/train_nnue_default.py",
            "--batch_size",
            "64",
            "--max_epochs",
            "10",
            "--learning_rate",
            "0.001",
        ]

        with patch.object(sys, "argv", test_args):
            parser = trainer.setup_argument_parser()
            args = parser.parse_args()

            assert args.config == "config/train_nnue_default.py"
            assert args.batch_size == 64
            assert args.max_epochs == 10
            assert args.learning_rate == 0.001

    def test_etinynet_argument_parsing(self, monkeypatch):
        """Test EtinyNet-specific argument parsing."""
        adapter = EtinyNetAdapter()
        trainer = BaseTrainer(adapter)

        # Mock sys.argv for testing
        test_args = [
            "train.py",
            "--config",
            "config/train_etinynet_default.py",
            "--batch_size",
            "32",
            "--variant",
            "1.0",
            "--dataset",
            "cifar100",
        ]

        with patch.object(sys, "argv", test_args):
            parser = trainer.setup_argument_parser()
            args = parser.parse_args()

            assert args.config == "config/train_etinynet_default.py"
            assert args.batch_size == 32
            assert args.variant == "1.0"
            assert args.dataset == "cifar100"
