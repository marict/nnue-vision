"""Test configuration and fixtures for NNUE-Vision tests."""

import tempfile

# ---------------------------------------------------------------------------
# Global warning filters to keep test output clean (no functional impact)
# ---------------------------------------------------------------------------
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn
from PIL import Image

# Ignore GPU‐available warnings when CPU fallback is used (common on CI/M1)
warnings.filterwarnings(
    "ignore",
    message=r"GPU available .+ not used",
    category=UserWarning,
)

# Ignore Lightning suggestion about DataLoader workers
warnings.filterwarnings(
    "ignore",
    message=r"The '(val_|train_|test_)?dataloader' does not have many workers",
    category=UserWarning,
)

# Ignore GraphQL ByteString deprecation warnings (external library issue)
warnings.filterwarnings(
    "ignore",
    message=r"'typing\.ByteString' is deprecated",
    category=DeprecationWarning,
    module="graphql",
)

# Ignore checkpoint directory already exists notice
warnings.filterwarnings(
    "ignore",
    message=r"Checkpoint directory .+ exists and is not empty",
    category=UserWarning,
)

# Ignore small batch vs logging interval info
warnings.filterwarnings(
    "ignore",
    message=r"The number of training batches \(\d+\) is smaller than the logging interval",
    category=UserWarning,
)

from data import GenericVisionDataset, create_data_loaders
from model import NNUE, GridFeatureSet, LossParams


class DummyArtifact:
    """Mock wandb artifact for testing."""

    def __init__(self, name, type=None):
        self.name = name
        self.type = type

    def add_file(self, path):
        pass

    def __getattr__(self, item):
        def _dummy(*args, **kwargs):
            return None

        return _dummy


class DummyWandbLogger:
    """Minimal stub for Lightning's WandbLogger used in tests."""

    def __init__(self, *args, **kwargs):
        # Provide just the attributes accessed in the training script
        from types import SimpleNamespace

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

    def log(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def Artifact(self, name, type=None):
        return DummyArtifact(name, type)

    # Gracefully handle any other method/attribute requests
    def __getattr__(self, item):
        def _dummy(*args, **kwargs):
            return None

        return _dummy


class MockDataset(torch.utils.data.Dataset):
    """Ultra-fast mock dataset for testing - no downloads, pure synthetic."""

    def __init__(self, size=8, num_classes=10, image_size=(96, 96)):
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate deterministic but varied synthetic images
        torch.manual_seed(idx)  # Deterministic per index
        image = torch.randn(3, *self.image_size)
        label = idx % self.num_classes
        return image, label


@pytest.fixture
def fast_mock_dataset():
    """Ultra-fast mock dataset with 8 samples."""
    return MockDataset(size=8, num_classes=10)


@pytest.fixture
def tiny_mock_dataset():
    """Tiny mock dataset with just 4 samples for fastest tests."""
    return MockDataset(size=4, num_classes=2)  # Binary for integration tests


@pytest.fixture
def fast_data_loaders():
    """Fast data loaders using mock data."""
    dataset = MockDataset(size=8, num_classes=2)  # Binary for test models
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    return train_loader, val_loader, test_loader


@pytest.fixture
def device():
    """Return the best available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def grid_feature_set():
    """Return a standard GridFeatureSet for testing."""
    return GridFeatureSet(grid_size=8, num_features_per_square=12)


@pytest.fixture
def small_grid_feature_set():
    """Return a smaller GridFeatureSet for faster testing."""
    return GridFeatureSet(grid_size=4, num_features_per_square=6)


@pytest.fixture
def tiny_grid_feature_set():
    """Return a very small GridFeatureSet for the fastest testing."""
    return GridFeatureSet(grid_size=4, num_features_per_square=8)


@pytest.fixture
def nnue_model():
    """Return an NNUE model for testing."""
    # Use smaller architecture for faster tests, consistent with new 0.98M defaults
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
    return NNUE(
        feature_set=feature_set,
        l1_size=128,  # Much smaller than new default 1024
        l2_size=8,  # Smaller than default 15
        l3_size=16,  # Smaller than default 32
        num_ls_buckets=2,  # Smaller for testing
    )


@pytest.fixture
def small_nnue_model():
    """Return a smaller NNUE model for faster testing."""
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=6)
    return NNUE(
        feature_set=feature_set,
        l1_size=64,  # Very small
        l2_size=4,  # Tiny
        l3_size=8,  # Tiny
        num_ls_buckets=2,
    )


@pytest.fixture
def tiny_nnue_model():
    """Return a tiny NNUE model for the fastest testing."""
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
    return NNUE(
        feature_set=feature_set,
        l1_size=32,  # Minimal
        l2_size=4,  # Minimal
        l3_size=4,  # Minimal
        num_ls_buckets=2,
    )


@pytest.fixture
def loss_params():
    """Return LossParams for testing."""
    return LossParams()


@pytest.fixture
def small_dataset():
    """Return a small dataset for testing."""
    return GenericVisionDataset(
        dataset_name="cifar10", split="train", target_size=(96, 96), max_samples=10
    )


@pytest.fixture
def sample_image_batch(device):
    """Generate a sample image batch for testing."""
    batch_size = 4

    # Generate synthetic batch data
    images = torch.randn(batch_size, 3, 96, 96, device=device)

    # Generate synthetic targets and scores
    targets = torch.rand(batch_size, 1, device=device)  # Between 0 and 1
    scores = torch.randn(batch_size, 1, device=device) * 100  # Search scores

    # Generate layer stack indices
    layer_stack_indices = torch.randint(0, 4, (batch_size,), device=device)

    return images, targets, scores, layer_stack_indices


@pytest.fixture
def small_image_batch(device):
    """Generate a smaller image batch for faster testing."""
    batch_size = 2

    images = torch.randn(batch_size, 3, 96, 96, device=device)

    targets = torch.rand(batch_size, 1, device=device)
    scores = torch.randn(batch_size, 1, device=device) * 50
    layer_stack_indices = torch.randint(0, 2, (batch_size,), device=device)

    return images, targets, scores, layer_stack_indices


@pytest.fixture
def tiny_image_batch(device):
    """Generate a tiny image batch for the fastest testing."""
    batch_size = 2

    images = torch.randn(batch_size, 3, 96, 96, device=device)

    targets = torch.rand(batch_size, 1, device=device)
    scores = torch.randn(batch_size, 1, device=device) * 20
    layer_stack_indices = torch.randint(0, 2, (batch_size,), device=device)

    return images, targets, scores, layer_stack_indices


@pytest.fixture
def sample_sparse_batch(grid_feature_set, device):
    """Generate a sample sparse batch for testing."""
    batch_size = 4
    max_features = 50  # Much smaller than full feature set

    # Generate random feature indices
    feature_indices = torch.randint(
        0, grid_feature_set.num_features, (batch_size, max_features), device=device
    )

    # Mask some features as inactive (-1)
    mask = torch.rand(batch_size, max_features, device=device) > 0.3
    feature_indices = feature_indices * mask.to(device) + (-1) * (~mask).to(device)

    # Feature values (mostly 1.0 for active features)
    feature_values = torch.ones(batch_size, max_features, device=device)

    return feature_indices, feature_values


@pytest.fixture
def small_sparse_batch(small_grid_feature_set, device):
    """Generate a smaller sparse batch for faster testing."""
    batch_size = 2
    max_features = 20

    feature_indices = torch.randint(
        0,
        small_grid_feature_set.num_features,
        (batch_size, max_features),
        device=device,
    )
    mask = torch.rand(batch_size, max_features, device=device) > 0.4
    feature_indices = feature_indices * mask.to(device) + (-1) * (~mask).to(device)

    feature_values = torch.ones(batch_size, max_features, device=device)

    return feature_indices, feature_values


@pytest.fixture
def tiny_sparse_batch(tiny_grid_feature_set, device):
    """Generate a tiny sparse batch for the fastest testing."""
    batch_size = 2
    max_features = 10

    feature_indices = torch.randint(
        0,
        tiny_grid_feature_set.num_features,
        (batch_size, max_features),
        device=device,
    )
    mask = torch.rand(batch_size, max_features, device=device) > 0.5
    feature_indices = feature_indices * mask.to(device) + (-1) * (~mask).to(device)

    feature_values = torch.ones(batch_size, max_features, device=device)

    return feature_indices, feature_values


@pytest.fixture
def trained_nnue_model(small_nnue_model, device):
    """Return a 'trained' NNUE model (actually just initialized for speed)."""
    small_nnue_model.to(device)
    # Skip actual training for speed - just return initialized model
    # Real training would take too long for unit tests
    small_nnue_model.eval()
    return small_nnue_model


@pytest.fixture
def trained_tiny_model(tiny_nnue_model, device):
    """Return a 'trained' tiny NNUE model."""
    tiny_nnue_model.to(device)
    tiny_nnue_model.eval()
    return tiny_nnue_model


@pytest.fixture
def temp_model_path():
    """Provide a temporary file path for model serialization testing."""
    with tempfile.NamedTemporaryFile(suffix=".nnue", delete=False) as f:
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


# Utility functions for tests
def assert_tensor_shape(tensor, expected_shape):
    """Assert that tensor has expected shape."""
    actual_shape = tuple(tensor.shape)
    assert (
        actual_shape == expected_shape
    ), f"Expected shape {expected_shape}, got {actual_shape}"


def assert_model_output_valid(output, batch_size):
    """Assert that model output is valid."""
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    assert (
        output.shape[0] == batch_size
    ), f"Expected batch size {batch_size}, got {output.shape[0]}"
