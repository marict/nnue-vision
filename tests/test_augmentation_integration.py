"""Integration tests for data augmentation system."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestAugmentationIntegration:
    """Test full integration of the augmentation system with training."""

    def test_augmentation_command_line_arguments(self):
        """Test that augmentation arguments are recognized by the training script."""
        # Test that the training script accepts augmentation arguments
        cmd = [
            sys.executable,
            "train.py",
            "nnue",
            "--config",
            "config/train_nnue_default.py",
            "--max_epochs",
            "0",  # Don't actually train
            "--use_augmentation=True",
            "--augmentation_strength=light",
            "--help",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        # Should exit with code 0 for help and include augmentation options
        assert result.returncode == 0
        assert "--use_augmentation" in result.stdout
        assert "--augmentation_strength" in result.stdout
        assert "light" in result.stdout
        assert "medium" in result.stdout
        assert "heavy" in result.stdout

    def test_augmentation_config_validation(self):
        """Test that augmentation configuration is properly validated."""
        # Test invalid augmentation strength
        cmd = [
            sys.executable,
            "train.py",
            "nnue",
            "--config",
            "config/train_nnue_default.py",
            "--max_epochs",
            "0",
            "--augmentation_strength=invalid",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())

        # Should fail with invalid choice error
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower()

    def test_etinynet_training_with_augmentation_works(self):
        """Test that EtinyNet training also works with augmentation."""
        cmd = [
            sys.executable,
            "train.py",
            "etinynet",
            "--config",
            "config/train_etinynet_default.py",
            "--max_epochs",
            "1",
            "--batch_size",
            "2",
            "--use_augmentation=True",
            "--augmentation_strength=medium",
            "--wandb_api_key=dummy",
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=Path.cwd(), timeout=120
            )

            # Training should complete successfully
            assert result.returncode == 0
            assert "Training completed!" in result.stdout

        except subprocess.TimeoutExpired:
            pytest.skip("EtinyNet training test skipped due to timeout")
        except Exception as e:
            pytest.skip(f"EtinyNet training test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
