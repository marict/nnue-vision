"""
Test the unified training script (train.py) with tiny configs.

These tests run the complete training script to ensure the unified interface works end-to-end,
using ultra-minimal configurations that complete in seconds.
"""

import os
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

import train

from .conftest import DummyWandbLogger


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
    mock_wandb = DummyWandbLogger()

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


class TestTrainingScriptExecution:
    """Test actual execution of training scripts with tiny configs."""

    @pytest.mark.timeout(30)  # Should complete in under 30 seconds [[memory:4098712]]
    def test_train_nnue_script_runs(
        self, setup_test_env, mock_wandb_components, monkeypatch
    ):
        """Test that unified train.py can run NNUE training with tiny config."""
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

    @pytest.mark.timeout(30)  # Should complete in under 30 seconds [[memory:4098712]]
    def test_train_etinynet_script_runs(
        self, setup_test_env, mock_wandb_components, monkeypatch
    ):
        """Test that unified train.py can run EtinyNet training with tiny config."""
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

    @pytest.mark.timeout(
        45
    )  # Allow a bit more time for subprocess calls [[memory:4098712]]
    def test_train_nnue_subprocess_execution(self, setup_test_env):
        """Test running train.py nnue as subprocess (closer to real usage)."""
        test_outputs = setup_test_env

        # Set environment variables for subprocess
        env = os.environ.copy()
        env["WANDB_API_KEY"] = "dummy_test_key"
        env["WANDB_MODE"] = "disabled"

        # Run as subprocess
        cmd = [
            sys.executable,
            "train.py",
            "nnue",
            "--config",
            "config/train_nnue_default.py",
            "--log_dir",
            str(test_outputs),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=40,  # Timeout for subprocess
        )

        # Check that it completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        assert (
            result.returncode == 0
        ), f"Script failed with return code {result.returncode}"
        assert (
            "Training completed!" in result.stdout
            or "Training completed!" in result.stderr
            or "Training completed successfully!" in result.stdout
            or "Training completed successfully!" in result.stderr
        )

    @pytest.mark.timeout(
        45
    )  # Allow a bit more time for subprocess calls [[memory:4098712]]
    def test_train_etinynet_subprocess_execution(self, setup_test_env):
        """Test running train.py etinynet as subprocess (closer to real usage)."""
        test_outputs = setup_test_env

        # Set environment variables for subprocess
        env = os.environ.copy()
        env["WANDB_API_KEY"] = "dummy_test_key"
        env["WANDB_MODE"] = "disabled"

        # Run as subprocess
        cmd = [
            sys.executable,
            "train.py",
            "etinynet",
            "--config",
            "config/train_etinynet_default.py",
            "--log_dir",
            str(test_outputs),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=40,  # Timeout for subprocess
        )

        # Check that it completed successfully
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        assert (
            result.returncode == 0
        ), f"Script failed with return code {result.returncode}"
        assert (
            "Training completed!" in result.stdout
            or "Training completed!" in result.stderr
            or "Training completed successfully!" in result.stdout
            or "Training completed successfully!" in result.stderr
        )


class TestConfigValidation:
    """Test that the default configs are valid and loadable."""

    def test_nnue_default_config_loads(self):
        """Test that the default NNUE config loads correctly."""
        from config import load_config

        config = load_config("config/train_nnue_default.py")

        # Verify key attributes
        assert config.name == "nnue_default"
        assert config.batch_size == 2
        assert config.max_epochs == 1
        assert config.subset == 0.001
        assert config.accelerator == "cpu"
        assert config.num_workers == 0

    def test_etinynet_default_config_loads(self):
        """Test that the default EtinyNet config loads correctly."""
        from config import load_config

        config = load_config("config/train_etinynet_default.py")

        # Verify key attributes
        assert config.name == "etinynet_default"
        assert config.batch_size == 2
        assert config.max_epochs == 1
        assert config.subset == 0.001
        assert config.accelerator == "cpu"
        assert config.num_workers == 0
        assert config.etinynet_variant == "0.75"


class TestScriptArgumentParsing:
    """Test that unified training script handles arguments correctly."""

    def test_unified_script_argument_parsing(self, monkeypatch):
        """Test unified train.py argument parsing and structure."""
        # Mock to prevent actual execution
        monkeypatch.setattr(train, "main", lambda: None)

        # Test that script imports and basic structure works
        assert hasattr(train, "main")
        assert hasattr(train, "get_available_adapters")

        # Test available adapters
        adapters = train.get_available_adapters()
        assert "nnue" in adapters
        assert "etinynet" in adapters

    def test_nnue_adapter_structure(self):
        """Test NNUE adapter has required methods."""
        from nnue_adapter import NNUEAdapter, NNUEWrapper, adapt_batch_for_nnue

        adapter = NNUEAdapter()
        assert hasattr(adapter, "create_model")
        assert hasattr(adapter, "create_data_loaders")
        assert callable(adapt_batch_for_nnue)
        assert NNUEWrapper is not None

    def test_etinynet_adapter_structure(self):
        """Test EtinyNet adapter has required methods."""
        from etinynet_adapter import EtinyNetAdapter

        adapter = EtinyNetAdapter()
        assert hasattr(adapter, "create_model")
        assert hasattr(adapter, "create_data_loaders")
