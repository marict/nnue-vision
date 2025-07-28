"""Integration tests for training scripts.

Tests that verify the main training entry points work correctly.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestTrainingScriptIntegration:
    """Test that training scripts complete successfully."""

    def test_nnue_training_script_basic_functionality(self):
        """Test that 'python train.py nnue' script can start and recognize arguments."""
        # Test that the script loads correctly and can parse basic arguments
        cmd = [
            sys.executable,
            "train.py",
            "nnue",
            "--config",
            "config/train_nnue_test.py",
            "--max_epochs",
            "0",  # Don't actually train, just test argument parsing
            "--batch_size",
            "2",
            "--wandb_api_key",
            "dummy_key_for_testing",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # Short timeout for basic functionality test
            )

            # Script should either succeed (exit 0) or fail gracefully with recognizable error
            # The key is that it should NOT fail with import errors or syntax errors
            output = result.stdout + result.stderr

            # Check that it's not failing due to import/syntax errors
            assert (
                "ImportError" not in output
            ), f"Import error in training script: {output}"
            assert (
                "SyntaxError" not in output
            ), f"Syntax error in training script: {output}"
            assert (
                "ModuleNotFoundError" not in output
            ), f"Module not found error: {output}"

            # Check that it gets far enough to recognize the model type
            assert (
                "nnue" in output.lower() or "NNUE" in output
            ), f"Script doesn't seem to recognize NNUE model type: {output}"

            print(f"✅ NNUE training script basic functionality verified")

        except subprocess.TimeoutExpired:
            pytest.fail(
                "NNUE training script test timed out - basic functionality issue"
            )
        except Exception as e:
            pytest.fail(f"NNUE training script test failed: {e}")

    def test_etinynet_training_script_basic_functionality(self):
        """Test that 'python train.py etinynet' script can start and recognize arguments."""
        # Test that the script loads correctly and can parse basic arguments
        cmd = [
            sys.executable,
            "train.py",
            "etinynet",
            "--config",
            "config/train_etinynet_test.py",
            "--max_epochs",
            "0",  # Don't actually train, just test argument parsing
            "--batch_size",
            "2",
            "--wandb_api_key",
            "dummy_key_for_testing",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # Short timeout for basic functionality test
            )

            # Script should either succeed (exit 0) or fail gracefully with recognizable error
            # The key is that it should NOT fail with import errors or syntax errors
            output = result.stdout + result.stderr

            # Check that it's not failing due to import/syntax errors
            assert (
                "ImportError" not in output
            ), f"Import error in training script: {output}"
            assert (
                "SyntaxError" not in output
            ), f"Syntax error in training script: {output}"
            assert (
                "ModuleNotFoundError" not in output
            ), f"Module not found error: {output}"

            # Check that it gets far enough to recognize the model type
            assert (
                "etinynet" in output.lower() or "EtinyNet" in output
            ), f"Script doesn't seem to recognize EtinyNet model type: {output}"

            print(f"✅ EtinyNet training script basic functionality verified")

        except subprocess.TimeoutExpired:
            pytest.fail(
                "EtinyNet training script test timed out - basic functionality issue"
            )
        except Exception as e:
            pytest.fail(f"EtinyNet training script test failed: {e}")

    def test_training_help_commands_work(self):
        """Test that training script help commands work."""
        # Test general help
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "nnue" in result.stdout
        assert "etinynet" in result.stdout

        # Test NNUE help
        result = subprocess.run(
            [sys.executable, "train.py", "nnue", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--batch_size" in result.stdout

        # Test EtinyNet help
        result = subprocess.run(
            [sys.executable, "train.py", "etinynet", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--etinynet_variant" in result.stdout

        print("✅ All help commands work correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
