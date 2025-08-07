import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from nnue_runpod_service import (
    RunPodError,
    _check_git_status,
    _extract_project_name_from_config,
    _resolve_gpu_id,
    start_cloud_training,
    stop_runpod,
)


class TestRunPodService:
    """Test RunPod service functionality."""

    def test_extract_project_name_from_config_success(self):
        """Test successful project name extraction from config."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("project_name = 'test-project'\n")
            config_path = f.name

        try:
            project_name = _extract_project_name_from_config(config_path)
            assert project_name == "test-project"
        finally:
            os.unlink(config_path)

    def test_extract_project_name_from_config_missing_file(self):
        """Test project name extraction with missing config file."""
        from config.config_loader import ConfigError

        with pytest.raises(ConfigError):
            _extract_project_name_from_config("nonexistent_config.py")

    def test_extract_project_name_from_config_missing_project_name(self):
        """Test project name extraction with config missing project_name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("learning_rate = 0.001\n")
            config_path = f.name

        try:
            with pytest.raises(AttributeError):
                _extract_project_name_from_config(config_path)
        finally:
            os.unlink(config_path)

    @patch("subprocess.run")
    def test_check_git_status_clean(self, mock_run):
        """Test git status check with clean repository."""
        # Mock successful git commands
        mock_run.side_effect = [
            Mock(returncode=0),  # git rev-parse
            Mock(returncode=0, stdout=""),  # git status --porcelain (clean)
        ]

        # Should not raise any exception
        _check_git_status()

    @patch("subprocess.run")
    def test_check_git_status_uncommitted_changes(self, mock_run):
        """Test git status check with uncommitted changes."""
        # Mock successful git commands but with uncommitted changes
        mock_run.side_effect = [
            Mock(returncode=0),  # git rev-parse
            Mock(
                returncode=0, stdout="M  modified_file.py"
            ),  # git status --porcelain (dirty)
        ]

        with pytest.raises(RunPodError, match="Uncommitted changes detected!"):
            _check_git_status()

    @patch("subprocess.run")
    def test_check_git_status_not_git_repo(self, mock_run):
        """Test git status check when not in a git repository."""
        # Mock git rev-parse failure
        from subprocess import CalledProcessError

        mock_run.side_effect = [
            CalledProcessError(
                128, ["git", "rev-parse", "--git-dir"]
            ),  # git rev-parse fails
        ]

        with pytest.raises(RunPodError, match="Failed to check git status"):
            _check_git_status()

    @patch("runpod.get_gpus")
    def test_resolve_gpu_id_success(self, mock_get_gpus):
        """Test successful GPU ID resolution."""
        mock_gpus = [
            {"id": "gpu-1", "displayName": "NVIDIA RTX 2000 Ada Generation"},
            {"id": "gpu-2", "displayName": "NVIDIA RTX 4000"},
        ]
        mock_get_gpus.return_value = mock_gpus

        gpu_id = _resolve_gpu_id("NVIDIA RTX 2000 Ada Generation")
        assert gpu_id == "gpu-1"

    @patch("runpod.get_gpus")
    def test_resolve_gpu_id_not_found(self, mock_get_gpus):
        """Test GPU ID resolution when GPU type not found."""
        mock_gpus = [
            {"id": "gpu-1", "displayName": "NVIDIA RTX 4000"},
        ]
        mock_get_gpus.return_value = mock_gpus

        with pytest.raises(
            RunPodError, match="GPU type 'NVIDIA RTX 2000 Ada Generation' not found"
        ):
            _resolve_gpu_id("NVIDIA RTX 2000 Ada Generation")

    @patch("runpod.get_gpus")
    def test_resolve_gpu_id_api_error(self, mock_get_gpus):
        """Test GPU ID resolution when API fails."""
        mock_get_gpus.side_effect = Exception("API Error")

        with pytest.raises(RunPodError, match="Failed to list GPUs"):
            _resolve_gpu_id("NVIDIA RTX 2000 Ada Generation")

    @patch("nnue_runpod_service._check_git_status")
    @patch("nnue_runpod_service._extract_project_name_from_config")
    @patch("nnue_runpod_service._resolve_gpu_id")
    @patch("nnue_runpod_service._open_browser")
    @patch("wandb.init")
    @patch("runpod.create_pod")
    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key", "RUNPOD_API_KEY": "test_key"})
    def test_start_cloud_training_success(
        self,
        mock_create_pod,
        mock_wandb_init,
        mock_open_browser,
        mock_resolve_gpu,
        mock_extract_project,
        mock_check_git,
    ):
        """Test successful cloud training start."""
        # Mock all the dependencies
        mock_check_git.return_value = None
        mock_extract_project.return_value = "test-project"
        mock_resolve_gpu.return_value = "gpu-1"
        mock_wandb_run = Mock()
        mock_wandb_run.id = "test-run-id"
        mock_wandb_run.url = "https://wandb.ai/test"
        mock_wandb_init.return_value = mock_wandb_run
        mock_pod = {"id": "pod-123"}
        mock_create_pod.return_value = mock_pod

        # Test the function
        pod_id = start_cloud_training("nnue --config config/train_nnue_default.py")

        assert pod_id == "pod-123"
        mock_check_git.assert_called_once()
        mock_extract_project.assert_called_once_with("config/train_nnue_default.py")
        mock_resolve_gpu.assert_called_once()
        mock_wandb_init.assert_called_once()
        mock_create_pod.assert_called_once()

    @patch("nnue_runpod_service._check_git_status")
    @patch.dict(os.environ, {}, clear=True)
    def test_start_cloud_training_missing_wandb_key(self, mock_check_git):
        """Test cloud training start with missing WANDB_API_KEY."""
        mock_check_git.return_value = None
        with pytest.raises(
            RunPodError, match="WANDB_API_KEY environment variable must be set"
        ):
            start_cloud_training("nnue --config config/train_nnue_default.py")

    @patch("nnue_runpod_service._check_git_status")
    @patch.dict(os.environ, {"WANDB_API_KEY": "test_key"}, clear=True)
    def test_start_cloud_training_missing_runpod_key(self, mock_check_git):
        """Test cloud training start with missing RUNPOD_API_KEY."""
        mock_check_git.return_value = None
        with pytest.raises(RunPodError, match="RunPod API key required"):
            start_cloud_training("nnue --config config/train_nnue_default.py")

    @patch("runpod.stop_pod")
    @patch.dict(os.environ, {"RUNPOD_API_KEY": "test_key", "RUNPOD_POD_ID": "pod-123"})
    def test_stop_runpod_success_sdk(self, mock_stop_pod):
        """Test successful RunPod stop using SDK."""
        result = stop_runpod()
        assert result is True
        mock_stop_pod.assert_called_once_with("pod-123")

    @patch("runpod.stop_pod")
    @patch("requests.post")
    @patch.dict(os.environ, {"RUNPOD_API_KEY": "test_key", "RUNPOD_POD_ID": "pod-123"})
    def test_stop_runpod_success_rest_fallback(self, mock_post, mock_stop_pod):
        """Test successful RunPod stop using REST fallback."""
        # Mock SDK failure
        mock_stop_pod.side_effect = Exception("SDK Error")
        # Mock successful REST call
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = stop_runpod()
        assert result is True
        mock_post.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_stop_runpod_no_pod_id(self):
        """Test RunPod stop with no pod ID."""
        result = stop_runpod()
        assert result is False

    @patch.dict(os.environ, {"RUNPOD_POD_ID": "pod-123"}, clear=True)
    def test_stop_runpod_no_api_key(self):
        """Test RunPod stop with no API key."""
        with pytest.raises(ValueError, match="RUNPOD_API_KEY not set"):
            stop_runpod()

    def test_runpod_error_exception(self):
        """Test RunPodError exception creation."""
        error = RunPodError("Test error message")
        assert str(error) == "Test error message"


if __name__ == "__main__":
    pytest.main([__file__])
