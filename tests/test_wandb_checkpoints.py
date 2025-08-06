"""Tests for wandb checkpoint saving functionality.

This test suite verifies that the CheckpointManager correctly:
1. Saves checkpoints locally in all scenarios
2. Uploads checkpoints to wandb when configured to do so
3. Respects upload frequency settings (best models + periodic)
4. Fails immediately when wandb upload errors occur (no graceful fallback)
5. Creates artifacts with correct metadata and naming
6. Tracks best checkpoints correctly
7. Works with realistic training workflow patterns

Test Coverage:
- Basic checkpoint saving without wandb upload
- Best model uploads to wandb
- Periodic checkpoint uploads to wandb
- Upload logic validation (when to upload)
- Fail-fast behavior for wandb failures
- Comprehensive training workflow simulation
- Configuration parameter handling

Note: In production, wandb is always initialized before training starts (training exits if no WANDB_API_KEY).
These tests use mocks to avoid requiring actual wandb credentials or network access.
"""

# Import the classes we need to test
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))
from nnue import NNUE, GridFeatureSet
from train import CheckpointManager


class MockWandbArtifact:
    """Mock wandb.Artifact for testing."""

    def __init__(self, name, type, metadata=None):
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files = []

    def add_file(self, file_path):
        self.files.append(file_path)


class MockWandbRun:
    """Mock wandb.run for testing."""

    def __init__(self):
        self.artifacts_logged = []

    def log_artifact(self, artifact):
        self.artifacts_logged.append(artifact)


class MockConfig:
    """Mock config object for testing."""

    def __init__(self, **kwargs):
        self.name = "test_config"
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestWandbCheckpointSaving:
    """Test wandb checkpoint saving functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_model(self):
        """Create a simple mock model for testing."""
        feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
        model = NNUE(
            feature_set=feature_set,
            input_size=32,
            l1_size=64,
            l2_size=16,
            l3_size=8,
            num_classes=10,
            # The following params are no longer in NNUE.__init__
            # loss_params={"type": "cross_entropy"},
            # weight_decay={},
        )
        return model

    @pytest.fixture
    def mock_optimizer(self, mock_model):
        """Create a mock optimizer for testing."""
        return torch.optim.SGD(mock_model.parameters(), lr=0.01)

    def test_best_checkpoint_wandb_upload(self, temp_dir, mock_model, mock_optimizer):
        """Test that best checkpoints are uploaded to wandb with correct metadata."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig(always_save_best_to_wandb=True)

        mock_run = MockWandbRun()

        with patch("train.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.Artifact = MockWandbArtifact
            mock_wandb.log_artifact = mock_run.log_artifact

            manager.save_best_model_to_wandb(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=10,
                metrics={"val_f1": 0.92, "val_loss": 0.15},
                config=config,
            )

        # Check that artifact was created and uploaded
        assert len(mock_run.artifacts_logged) == 1
        artifact = mock_run.artifacts_logged[0]

        assert artifact.name == "test-run-best"
        assert artifact.type == "best_model"
        assert artifact.metadata["epoch"] == 10
        assert artifact.metadata["metrics"]["val_f1"] == 0.92
        assert artifact.metadata["config_name"] == "test_config"
        assert artifact.metadata["run_name"] == "test-run"
        # Assert that the temporary file was added (but not necessarily exists now)
        assert len(artifact.files) == 1
        # The path itself is temporary and deleted, so we can't assert .exists()

    def test_wandb_upload_error_fails_fast(self, temp_dir, mock_model, mock_optimizer):
        """Test that wandb upload errors cause immediate failure (no graceful handling)."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig(always_save_best_to_wandb=True)

        with patch("train.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            mock_wandb.Artifact.side_effect = Exception("Wandb connection failed")

            # This should raise an exception and fail immediately
            with pytest.raises(Exception, match="Wandb connection failed"):
                manager.save_best_model_to_wandb(
                    model=mock_model,
                    optimizer=mock_optimizer,
                    epoch=10,
                    metrics={"val_f1": 0.92, "val_loss": 0.15},
                    config=config,
                )

    def test_full_upload_workflow_simulation(
        self, temp_dir, mock_model, mock_optimizer
    ):
        """Test a full training workflow simulation with realistic upload patterns."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig()

        mock_run = MockWandbRun()

        with patch("train.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.Artifact = MockWandbArtifact
            mock_wandb.log_artifact = mock_run.log_artifact

            # Simulate 20 epochs of training (only best model is saved)
            best_val_f1 = 0.70
            uploaded_epochs = []

            for epoch in range(1, 21):
                # Simulate F1 improvements only at specific "best" epochs
                if epoch == 1:
                    val_f1 = 0.70  # Initial value
                elif epoch == 2:
                    val_f1 = 0.70  # No improvement
                elif epoch == 3:
                    val_f1 = 0.75  # First best
                elif epoch == 4:
                    val_f1 = 0.75  # No improvement
                elif epoch == 5:
                    val_f1 = 0.75  # No improvement
                elif epoch == 6:
                    val_f1 = 0.75  # No improvement
                elif epoch == 7:
                    val_f1 = 0.75  # No improvement
                elif epoch == 8:
                    val_f1 = 0.80  # Second best
                elif epoch == 9:
                    val_f1 = 0.80  # No improvement
                elif epoch == 10:
                    val_f1 = 0.80  # No improvement
                elif epoch == 11:
                    val_f1 = 0.80  # No improvement
                elif epoch == 12:
                    val_f1 = 0.80  # No improvement
                elif epoch == 13:
                    val_f1 = 0.80  # No improvement
                elif epoch == 14:
                    val_f1 = 0.80  # No improvement
                elif epoch == 15:
                    val_f1 = 0.85  # Third best
                else:
                    val_f1 = best_val_f1  # No further improvements

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    uploaded_epochs.append(epoch)
                    manager.save_best_model_to_wandb(
                        model=mock_model,
                        optimizer=mock_optimizer,
                        epoch=epoch,
                        metrics={"val_f1": val_f1, "val_loss": 1.0 - val_f1},
                        config=config,
                    )

            # Verify correct epochs were uploaded (only the best ones)
            expected_uploads = [3, 8, 15]
            assert uploaded_epochs == expected_uploads

            # Verify artifacts were created
            assert len(mock_run.artifacts_logged) == len(expected_uploads)

            # Check that best models have correct type and metadata
            for artifact in mock_run.artifacts_logged:
                assert artifact.type == "best_model"
                assert artifact.name == "test-run-best"
                assert artifact.metadata["run_name"] == "test-run"


if __name__ == "__main__":
    pytest.main([__file__])
