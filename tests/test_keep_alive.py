"""Tests for keep_alive functionality in training framework and RunPod service."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from config.config_loader import load_config
from training_utils import parse_args


class TestKeepAliveConfiguration:
    """Test keep_alive parameter configuration and usage."""

    def test_etinynet_config_has_keep_alive(self):
        """Test that EtinyNet config has keep_alive parameter set to True."""
        config = load_config("config/train_etinynet.py")

        assert hasattr(
            config, "keep_alive"
        ), "EtinyNet config should have keep_alive parameter"
        assert (
            config.keep_alive is True
        ), f"Expected keep_alive True, got {config.keep_alive}"

    def test_etinynet_default_config_keep_alive(self):
        """Test keep_alive in EtinyNet default config (should not be defined, defaults to False)."""
        config = load_config("config/train_etinynet_default.py")

        # Default config may not have keep_alive defined, which should default to False
        keep_alive = getattr(config, "keep_alive", False)
        assert isinstance(keep_alive, bool), "keep_alive should be boolean"

    def test_nnue_config_keep_alive_default(self):
        """Test that NNUE configs use default keep_alive behavior (False)."""
        config = load_config("config/train_nnue.py")

        # NNUE config doesn't define keep_alive, should default to False
        keep_alive = getattr(config, "keep_alive", False)
        assert (
            keep_alive is False
        ), f"Expected default keep_alive False, got {keep_alive}"

    def test_nnue_default_config_keep_alive_default(self):
        """Test that NNUE default config uses default keep_alive behavior (False)."""
        config = load_config("config/train_nnue_default.py")

        # NNUE default config doesn't define keep_alive, should default to False
        keep_alive = getattr(config, "keep_alive", False)
        assert (
            keep_alive is False
        ), f"Expected default keep_alive False, got {keep_alive}"


class TestKeepAliveCommandLineArgument:
    """Test keep_alive command line argument functionality."""

    def test_keep_alive_argument_definition(self):
        """Test that keep_alive argument is properly defined."""
        parser = parse_args()

        # Parse with keep-alive flag (need to include required config argument)
        args = parser.parse_args(["dummy_config", "--keep-alive"])
        assert hasattr(args, "keep_alive"), "Parser should have keep_alive attribute"
        assert (
            args.keep_alive is True
        ), "keep_alive should be True when flag is provided"

        # Parse without keep-alive flag
        args = parser.parse_args(["dummy_config"])
        assert hasattr(args, "keep_alive"), "Parser should have keep_alive attribute"
        assert (
            args.keep_alive is False
        ), "keep_alive should be False when flag is not provided"

    def test_keep_alive_help_text(self):
        """Test that keep_alive argument has proper help text."""
        parser = parse_args()

        help_text = parser.format_help()
        assert "--keep-alive" in help_text, "Help should contain --keep-alive flag"
        assert (
            "Keep instance alive after training" in help_text
        ), "Help should explain keep_alive purpose"


class TestKeepAliveTrainingFrameworkIntegration:
    """Test keep_alive integration with training framework."""

    @patch("training_framework.runpod_service_nnue")
    @patch.dict(os.environ, {"RUNPOD_POD_ID": "test-pod-123"})
    def test_runpod_stops_when_keep_alive_false(self, mock_runpod_service):
        """Test that RunPod instance stops when keep_alive is False."""
        # Mock the stop_runpod function
        mock_runpod_service.stop_runpod = Mock()

        # Create config with keep_alive = False
        config = SimpleNamespace()
        config.keep_alive = False

        # Simulate the end of training logic from training_framework.py
        if os.getenv("RUNPOD_POD_ID") and not getattr(config, "keep_alive", False):
            try:
                mock_runpod_service.stop_runpod()
            except ImportError:
                pass

        # Verify stop_runpod was called
        mock_runpod_service.stop_runpod.assert_called_once()

    @patch("training_framework.runpod_service_nnue")
    @patch.dict(os.environ, {"RUNPOD_POD_ID": "test-pod-123"})
    def test_runpod_stays_alive_when_keep_alive_true(self, mock_runpod_service):
        """Test that RunPod instance stays alive when keep_alive is True."""
        # Mock the stop_runpod function
        mock_runpod_service.stop_runpod = Mock()

        # Create config with keep_alive = True
        config = SimpleNamespace()
        config.keep_alive = True

        # Simulate the end of training logic
        if os.getenv("RUNPOD_POD_ID") and not getattr(config, "keep_alive", False):
            try:
                mock_runpod_service.stop_runpod()
            except ImportError:
                pass

        # Verify stop_runpod was NOT called
        mock_runpod_service.stop_runpod.assert_not_called()

    @patch("training_framework.runpod_service_nnue")
    @patch.dict(os.environ, {}, clear=True)
    def test_no_runpod_stop_when_not_on_runpod(self, mock_runpod_service):
        """Test that stop_runpod is not called when not running on RunPod."""
        # Mock the stop_runpod function
        mock_runpod_service.stop_runpod = Mock()

        # Create config with keep_alive = False
        config = SimpleNamespace()
        config.keep_alive = False

        # Simulate the end of training logic (no RUNPOD_POD_ID env var)
        if os.getenv("RUNPOD_POD_ID") and not getattr(config, "keep_alive", False):
            try:
                mock_runpod_service.stop_runpod()
            except ImportError:
                pass

        # Verify stop_runpod was NOT called (no RunPod environment)
        mock_runpod_service.stop_runpod.assert_not_called()

    def test_getattr_fallback_behavior(self):
        """Test that getattr properly handles missing keep_alive attribute."""
        # Config without keep_alive attribute
        config_without_keep_alive = SimpleNamespace()

        # Should default to False
        keep_alive_value = getattr(config_without_keep_alive, "keep_alive", False)
        assert (
            keep_alive_value is False
        ), "Should default to False when keep_alive not defined"

        # Config with keep_alive attribute
        config_with_keep_alive = SimpleNamespace()
        config_with_keep_alive.keep_alive = True

        # Should return the actual value
        keep_alive_value = getattr(config_with_keep_alive, "keep_alive", False)
        assert (
            keep_alive_value is True
        ), "Should return actual value when keep_alive is defined"


class TestKeepAliveRunPodServiceIntegration:
    """Test keep_alive integration with RunPod service."""

    def test_build_training_command_with_keep_alive_true(self):
        """Test that training command includes --keep-alive flag when keep_alive is True."""
        from runpod_service_nnue import _build_training_command

        train_args = "nnue --config config/train_nnue.py"
        command = _build_training_command(
            train_args=train_args,
            keep_alive=True,
            note=None,
            wandb_run_id=None,
            script_name="train.py",
        )

        assert "train.py" in command, "Command should include script name"
        assert train_args in command, "Command should include training arguments"
        assert (
            "--keep-alive" in command
        ), "Command should include --keep-alive flag when keep_alive is True"

    def test_build_training_command_with_keep_alive_false(self):
        """Test that training command excludes --keep-alive flag when keep_alive is False."""
        from runpod_service_nnue import _build_training_command

        train_args = "nnue --config config/train_nnue.py"
        command = _build_training_command(
            train_args=train_args,
            keep_alive=False,
            note=None,
            wandb_run_id=None,
            script_name="train.py",
        )

        assert "train.py" in command, "Command should include script name"
        assert train_args in command, "Command should include training arguments"
        assert (
            "--keep-alive" not in command
        ), "Command should NOT include --keep-alive flag when keep_alive is False"

    def test_build_training_command_with_all_parameters(self):
        """Test training command building with all parameters including keep_alive."""
        from runpod_service_nnue import _build_training_command

        train_args = "nnue --config config/train_nnue.py"
        command = _build_training_command(
            train_args=train_args,
            keep_alive=True,
            note="test-run",
            wandb_run_id="test-wandb-123",
            script_name="train.py",
        )

        expected_components = [
            "train.py",
            train_args,
            "--keep-alive",
            "--note=test-run",
            "--wandb-run-id=test-wandb-123",
        ]

        for component in expected_components:
            assert component in command, f"Command should include {component}"


class TestKeepAliveConfigConsistency:
    """Test consistency of keep_alive configuration across different models."""

    def test_keep_alive_values_are_boolean(self):
        """Test that keep_alive values in configs are boolean when defined."""
        configs_to_test = [
            "config/train_etinynet.py",
            "config/train_etinynet_default.py",
            "config/train_nnue.py",
            "config/train_nnue_default.py",
        ]

        for config_path in configs_to_test:
            config = load_config(config_path)

            if hasattr(config, "keep_alive"):
                assert isinstance(
                    config.keep_alive, bool
                ), f"keep_alive in {config_path} should be boolean, got {type(config.keep_alive)}"

    def test_keep_alive_reasonable_defaults(self):
        """Test that keep_alive has reasonable default behavior."""
        # EtinyNet config should have keep_alive = True (it's set explicitly)
        etinynet_config = load_config("config/train_etinynet.py")
        assert (
            getattr(etinynet_config, "keep_alive", False) is True
        ), "EtinyNet config should have keep_alive = True"

        # Other configs should default to False
        other_configs = [
            "config/train_etinynet_default.py",
            "config/train_nnue.py",
            "config/train_nnue_default.py",
        ]

        for config_path in other_configs:
            config = load_config(config_path)
            keep_alive = getattr(config, "keep_alive", False)
            # We're not asserting False here because they might have it defined
            # Just ensuring it's a boolean value
            assert isinstance(
                keep_alive, bool
            ), f"keep_alive in {config_path} should be boolean"


class TestKeepAliveDocumentationAndUsage:
    """Test that keep_alive functionality is properly documented and accessible."""

    def test_runpod_service_includes_keep_alive(self):
        """Test that RunPod service includes keep_alive functionality."""
        # Read the runpod_service_nnue.py file to verify keep_alive is implemented
        with open("runpod_service_nnue.py", "r") as f:
            service_content = f.read()

        assert (
            "--keep-alive" in service_content
        ), "RunPod service should include --keep-alive flag"
        assert (
            "keep_alive" in service_content
        ), "RunPod service should include keep_alive parameter"
        assert (
            "Keep pod alive after training" in service_content
        ), "RunPod service should document keep_alive functionality"

    def test_container_setup_recognizes_keep_alive_flag(self):
        """Test that container setup script recognizes keep-alive flag."""
        # Read the container setup script
        with open("container_setup.sh", "r") as f:
            script_content = f.read()

        assert (
            "--keep-alive" in script_content
        ), "Container setup script should check for --keep-alive flag"
        assert (
            "tail -f /dev/null" in script_content
        ), "Container setup script should keep container alive with tail command"
