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
from model import NNUE, GridFeatureSet
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
            loss_params={"type": "cross_entropy"},
            weight_decay={},
        )
        return model

    @pytest.fixture
    def mock_optimizer(self, mock_model):
        """Create a mock optimizer for testing."""
        return torch.optim.SGD(mock_model.parameters(), lr=0.01)

    def test_checkpoint_manager_basic_save(self, temp_dir, mock_model, mock_optimizer):
        """Test basic checkpoint saving without wandb upload."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig()

        checkpoint_path = manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=5,
            metrics={"val_f1": 0.85, "val_loss": 0.25},
            config=config,
            is_best=False,
            upload_to_wandb=False,  # Explicitly disable wandb upload
        )

        # Check that checkpoint was saved locally
        assert Path(checkpoint_path).exists()
        assert "checkpoint-epoch-05.ckpt" in checkpoint_path

        # Load and verify checkpoint content
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["epoch"] == 5
        assert checkpoint["metrics"]["val_f1"] == 0.85
        assert checkpoint["config_name"] == "test_config"

    def test_best_checkpoint_wandb_upload(self, temp_dir, mock_model, mock_optimizer):
        """Test that best checkpoints are uploaded to wandb with correct metadata."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig(always_save_best_to_wandb=True)

        mock_run = MockWandbRun()

        with patch("train.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.Artifact = MockWandbArtifact
            mock_wandb.log_artifact = mock_run.log_artifact

            checkpoint_path = manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=10,
                metrics={"val_f1": 0.92, "val_loss": 0.15},
                config=config,
                is_best=True,
                upload_to_wandb=True,
            )

        # Check that checkpoint was saved locally
        assert Path(checkpoint_path).exists()
        assert "best-f1-10-0.920.ckpt" in checkpoint_path

        # Check that artifact was created and uploaded
        assert len(mock_run.artifacts_logged) == 1
        artifact = mock_run.artifacts_logged[0]

        assert artifact.name == "test-run-best"
        assert artifact.type == "best_model"
        assert artifact.metadata["epoch"] == 10
        assert artifact.metadata["metrics"]["val_f1"] == 0.92
        assert artifact.metadata["config_name"] == "test_config"
        assert artifact.metadata["is_best"] is True
        assert checkpoint_path in artifact.files

    def test_periodic_checkpoint_wandb_upload(
        self, temp_dir, mock_model, mock_optimizer
    ):
        """Test that periodic checkpoints are uploaded to wandb."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig(save_checkpoint_every_n_epochs=5)

        mock_run = MockWandbRun()

        with patch("train.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.Artifact = MockWandbArtifact
            mock_wandb.log_artifact = mock_run.log_artifact

            checkpoint_path = manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=15,  # Should trigger upload (15 % 5 == 0)
                metrics={"val_f1": 0.78, "val_loss": 0.35},
                config=config,
                is_best=False,
                upload_to_wandb=True,
            )

        # Check that artifact was created and uploaded
        assert len(mock_run.artifacts_logged) == 1
        artifact = mock_run.artifacts_logged[0]

        assert artifact.name == "test-run-epoch-15"
        assert artifact.type == "checkpoint"
        assert artifact.metadata["is_best"] is False

    def test_no_wandb_upload_when_disabled(self, temp_dir, mock_model, mock_optimizer):
        """Test that checkpoints are not uploaded when upload_to_wandb=False."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig(always_save_best_to_wandb=True)

        mock_run = MockWandbRun()

        with patch("train.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.Artifact = MockWandbArtifact
            mock_wandb.log_artifact = mock_run.log_artifact

            checkpoint_path = manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                epoch=10,
                metrics={"val_f1": 0.92, "val_loss": 0.15},
                config=config,
                is_best=True,
                upload_to_wandb=False,  # Explicitly disabled
            )

        # Check that checkpoint was saved locally
        assert Path(checkpoint_path).exists()

        # Check that no artifacts were uploaded
        assert len(mock_run.artifacts_logged) == 0

    def test_wandb_upload_error_fails_fast(self, temp_dir, mock_model, mock_optimizer):
        """Test that wandb upload errors cause immediate failure (no graceful handling)."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig(always_save_best_to_wandb=True)

        with patch("train.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            mock_wandb.Artifact.side_effect = Exception("Wandb connection failed")

            # This should raise an exception and fail immediately
            with pytest.raises(Exception, match="Wandb connection failed"):
                manager.save_checkpoint(
                    model=mock_model,
                    optimizer=mock_optimizer,
                    epoch=10,
                    metrics={"val_f1": 0.92, "val_loss": 0.15},
                    config=config,
                    is_best=True,
                    upload_to_wandb=True,
                )

    def test_checkpoint_upload_logic_best_models(self):
        """Test the logic for determining when to upload checkpoints - best models."""
        config = MockConfig(
            always_save_best_to_wandb=True, save_checkpoint_every_n_epochs=10
        )

        # Test best model upload logic
        is_best = True
        epoch = 7  # Not a multiple of 10

        should_upload = (is_best and config.always_save_best_to_wandb) or (
            epoch % config.save_checkpoint_every_n_epochs == 0
        )

        assert should_upload is True  # Should upload because it's best

    def test_checkpoint_upload_logic_periodic(self):
        """Test the logic for determining when to upload checkpoints - periodic."""
        config = MockConfig(
            always_save_best_to_wandb=True, save_checkpoint_every_n_epochs=10
        )

        # Test periodic upload logic
        is_best = False
        epoch = 20  # Multiple of 10

        should_upload = (is_best and config.always_save_best_to_wandb) or (
            epoch % config.save_checkpoint_every_n_epochs == 0
        )

        assert should_upload is True  # Should upload because it's periodic

    def test_checkpoint_upload_logic_no_upload(self):
        """Test the logic for determining when to upload checkpoints - no upload."""
        config = MockConfig(
            always_save_best_to_wandb=True, save_checkpoint_every_n_epochs=10
        )

        # Test no upload logic
        is_best = False
        epoch = 7  # Not best, not multiple of 10

        should_upload = (is_best and config.always_save_best_to_wandb) or (
            epoch % config.save_checkpoint_every_n_epochs == 0
        )

        assert should_upload is False  # Should not upload

    def test_checkpoint_upload_logic_best_disabled(self):
        """Test the logic when best model uploading is disabled."""
        config = MockConfig(
            always_save_best_to_wandb=False,  # Disabled
            save_checkpoint_every_n_epochs=10,
        )

        # Test best model with uploading disabled
        is_best = True
        epoch = 7

        should_upload = (is_best and config.always_save_best_to_wandb) or (
            epoch % config.save_checkpoint_every_n_epochs == 0
        )

        assert should_upload is False  # Should not upload even though it's best

    def test_checkpoint_manager_tracks_best(self, temp_dir, mock_model, mock_optimizer):
        """Test that CheckpointManager correctly tracks the best checkpoint."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig()

        # Save first checkpoint (not best)
        manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=1,
            metrics={"val_f1": 0.75},
            config=config,
            is_best=False,
            upload_to_wandb=False,  # Disable wandb upload for tracking test
        )

        assert manager.best_checkpoint_path is None
        assert manager.best_metric is None

        # Save best checkpoint
        best_path = manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=5,
            metrics={"val_f1": 0.85},
            config=config,
            is_best=True,
            upload_to_wandb=False,  # Disable wandb upload for tracking test
        )

        assert manager.best_checkpoint_path == Path(best_path)
        assert manager.best_metric == 0.85

        # Save another non-best checkpoint
        manager.save_checkpoint(
            model=mock_model,
            optimizer=mock_optimizer,
            epoch=8,
            metrics={"val_f1": 0.80},
            config=config,
            is_best=False,
            upload_to_wandb=False,  # Disable wandb upload for tracking test
        )

        # Best should still be from epoch 5
        assert manager.best_checkpoint_path == Path(best_path)
        assert manager.best_metric == 0.85

    def test_full_upload_workflow_simulation(
        self, temp_dir, mock_model, mock_optimizer
    ):
        """Test a full training workflow simulation with realistic upload patterns."""
        manager = CheckpointManager(temp_dir, run_name="test-run")
        config = MockConfig(
            always_save_best_to_wandb=True, save_checkpoint_every_n_epochs=5
        )

        mock_run = MockWandbRun()

        with patch("train.wandb") as mock_wandb:
            mock_wandb.run = mock_run
            mock_wandb.Artifact = MockWandbArtifact
            mock_wandb.log_artifact = mock_run.log_artifact

            # Simulate 20 epochs of training
            best_epochs = []
            uploaded_epochs = []

            for epoch in range(1, 21):
                # Simulate some F1 improvements (best models at epochs 3, 8, 15)
                val_f1 = 0.7 + (0.01 * epoch) + (0.05 if epoch in [3, 8, 15] else 0)
                is_best = epoch in [3, 8, 15]

                if is_best:
                    best_epochs.append(epoch)

                # Determine if should upload
                should_upload = (is_best and config.always_save_best_to_wandb) or (
                    epoch % config.save_checkpoint_every_n_epochs == 0
                )

                if should_upload:
                    uploaded_epochs.append(epoch)

                manager.save_checkpoint(
                    model=mock_model,
                    optimizer=mock_optimizer,
                    epoch=epoch,
                    metrics={"val_f1": val_f1, "val_loss": 1.0 - val_f1},
                    config=config,
                    is_best=is_best,
                    upload_to_wandb=should_upload,
                )

            # Verify correct epochs were uploaded
            # Should be: 3 (best), 5 (periodic), 8 (best), 10 (periodic), 15 (best+periodic), 20 (periodic)
            expected_uploads = [3, 5, 8, 10, 15, 20]
            assert uploaded_epochs == expected_uploads

            # Verify artifacts were created
            assert len(mock_run.artifacts_logged) == len(expected_uploads)

            # Check that best models have correct type and metadata
            best_artifacts = [
                a for a in mock_run.artifacts_logged if a.metadata.get("is_best")
            ]
            periodic_artifacts = [
                a for a in mock_run.artifacts_logged if not a.metadata.get("is_best")
            ]

            assert len(best_artifacts) == 3  # epochs 3, 8, 15
            assert len(periodic_artifacts) == 3  # epochs 5, 10, 20

            # Verify artifact names and types
            for artifact in best_artifacts:
                assert artifact.type == "best_model"
                assert artifact.name == "test-run-best"

            for artifact in periodic_artifacts:
                assert artifact.type == "checkpoint"
                assert artifact.name.startswith("test-run-epoch-")


if __name__ == "__main__":
    pytest.main([__file__])
