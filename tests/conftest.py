"""
Shared pytest fixtures and test utilities for NNUE-Vision tests.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from dataset import SyntheticVisualWakeWordsDataset, create_data_loaders
from model import NNUE, GridFeatureSet, LossParams


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
def nnue_model():
    """Return an NNUE model for testing."""
    return NNUE(
        max_epoch=10,
        num_batches_per_epoch=100,
        lr=1e-3,
        num_ls_buckets=4,  # Smaller for testing
    )


@pytest.fixture
def small_nnue_model():
    """Return a smaller NNUE model for faster testing."""
    return NNUE(
        max_epoch=10,
        num_batches_per_epoch=100,
        lr=1e-3,
        num_ls_buckets=2,  # Very small for testing
    )


@pytest.fixture
def loss_params():
    """Return standard loss parameters for testing."""
    return LossParams()


@pytest.fixture
def sample_image_batch(device):
    """Return a sample batch of image data for NNUE model."""
    batch_size = 4

    # Generate random 96x96 RGB images
    images = torch.randn(batch_size, 3, 96, 96, device=device)

    # Random targets and scores for loss computation
    targets = torch.rand(batch_size, 1, device=device)  # Between 0 and 1
    scores = torch.randn(batch_size, 1, device=device) * 100  # Search scores

    # Random layer stack indices (bucket selection)
    layer_stack_indices = torch.randint(0, 4, (batch_size,), device=device)

    return (images, targets, scores, layer_stack_indices)


@pytest.fixture
def small_image_batch(device):
    """Return a smaller sample batch for faster testing."""
    batch_size = 2

    # Generate random 96x96 RGB images
    images = torch.randn(batch_size, 3, 96, 96, device=device)

    targets = torch.rand(batch_size, 1, device=device)
    scores = torch.randn(batch_size, 1, device=device) * 50
    layer_stack_indices = torch.randint(0, 2, (batch_size,), device=device)

    return (images, targets, scores, layer_stack_indices)


@pytest.fixture
def sample_sparse_batch(grid_feature_set, device):
    """Return a sample batch of sparse feature data for testing internal components."""
    batch_size = 4
    max_features = 32  # Maximum number of active features per sample

    # Generate random feature indices (some samples might have fewer features)
    feature_indices = torch.randint(
        0, grid_feature_set.num_features, (batch_size, max_features), device=device
    )

    # Set some indices to -1 to simulate variable-length sparse features
    mask = torch.rand(batch_size, max_features) < 0.7  # 70% of features are active
    feature_indices = feature_indices * mask.to(device) + (-1) * (~mask).to(device)

    # Feature values (typically 1.0 for active features)
    feature_values = torch.ones(batch_size, max_features, device=device)

    return (feature_indices, feature_values)


@pytest.fixture
def small_sparse_batch(small_grid_feature_set, device):
    """Return a smaller sample batch for faster testing of internal components."""
    batch_size = 2
    max_features = 16

    feature_indices = torch.randint(
        0,
        small_grid_feature_set.num_features,
        (batch_size, max_features),
        device=device,
    )

    mask = torch.rand(batch_size, max_features) < 0.8
    feature_indices = feature_indices * mask.to(device) + (-1) * (~mask).to(device)

    feature_values = torch.ones(batch_size, max_features, device=device)

    return (feature_indices, feature_values)


@pytest.fixture
def trained_nnue_model(small_nnue_model, device):
    """Return an NNUE model that has been trained for a few steps."""
    small_nnue_model.to(device)
    small_nnue_model.train()

    # Create synthetic training data
    batch_size = 4

    optimizer = torch.optim.Adam(small_nnue_model.parameters(), lr=1e-3)

    # Train for a few steps
    for step in range(3):
        # Generate synthetic image batch
        images = torch.randn(batch_size, 3, 96, 96, device=device)
        targets = torch.rand(batch_size, 1, device=device)
        scores = torch.randn(batch_size, 1, device=device) * 50
        layer_stack_indices = torch.randint(0, 2, (batch_size,), device=device)

        batch = (images, targets, scores, layer_stack_indices)

        # Training step
        loss = small_nnue_model.training_step(batch, step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return small_nnue_model


@pytest.fixture
def temp_model_path(tmp_path):
    """Return a temporary path for saving models."""
    return tmp_path / "test_nnue_model.pt"


# Simple CNN model for dataset testing (not a legacy shim)
class SimpleTestCNN(nn.Module):
    """Minimal CNN for testing dataset integration only."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# Fixtures for dataset testing
@pytest.fixture
def simple_test_model():
    """Return a simple CNN model for dataset testing."""
    return SimpleTestCNN()


@pytest.fixture
def small_dataset():
    """Return a small synthetic dataset for testing."""
    return SyntheticVisualWakeWordsDataset(
        split="train", target_size=(96, 96), num_samples=10
    )


@pytest.fixture
def data_loaders():
    """Return small data loaders for testing."""
    return create_data_loaders(
        batch_size=4,
        num_workers=0,  # No multiprocessing for tests
        target_size=(96, 96),
    )


# Test utilities
def assert_tensor_shape(tensor, expected_shape):
    """Assert that a tensor has the expected shape."""
    assert (
        tensor.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {tensor.shape}"


def assert_tensor_range(tensor, min_val, max_val):
    """Assert that tensor values are within expected range."""
    assert tensor.min() >= min_val, f"Tensor minimum {tensor.min()} < {min_val}"
    assert tensor.max() <= max_val, f"Tensor maximum {tensor.max()} > {max_val}"


def assert_gradients_exist(model):
    """Assert that model parameters have gradients."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def assert_gradients_nonzero(model, tolerance=1e-8):
    """Assert that model has non-zero gradients."""
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2

    total_grad_norm = total_grad_norm**0.5
    assert (
        total_grad_norm > tolerance
    ), f"Total gradient norm {total_grad_norm} is too small"


def assert_sparse_features_valid(feature_indices, feature_values, max_features):
    """Assert that sparse features are in valid format."""
    batch_size = feature_indices.shape[0]
    max_active_features = feature_indices.shape[1]

    # Check shapes match
    assert feature_indices.shape == feature_values.shape
    assert feature_indices.shape == (batch_size, max_active_features)

    # Check feature indices are either valid or -1 (padding)
    valid_mask = feature_indices >= 0
    invalid_mask = feature_indices == -1
    assert torch.all(valid_mask | invalid_mask), "Feature indices must be >= 0 or -1"
    assert torch.all(
        feature_indices[valid_mask] < max_features
    ), "Feature indices exceed max_features"

    # Check feature values are positive where indices are valid
    assert torch.all(
        feature_values[valid_mask] > 0
    ), "Feature values must be positive for valid indices"


def assert_quantized_weights_valid(quantized_data):
    """Assert that quantized model data is valid for C++ export."""
    assert "feature_transformer" in quantized_data
    assert "conv_layer" in quantized_data
    assert "metadata" in quantized_data

    ft_data = quantized_data["feature_transformer"]
    assert ft_data["weight"].dtype == torch.int16
    assert ft_data["bias"].dtype == torch.int32
    assert ft_data["scale"] > 0

    # Check conv layer
    conv_data = quantized_data["conv_layer"]
    assert conv_data["weight"].dtype == torch.int8
    assert conv_data["bias"].dtype == torch.int32
    assert conv_data["scale"] > 0

    # Check layer stacks
    layer_stack_keys = [
        k for k in quantized_data.keys() if k.startswith("layer_stack_")
    ]
    assert len(layer_stack_keys) > 0

    for key in layer_stack_keys:
        ls_data = quantized_data[key]
        assert ls_data["l1_weight"].dtype == torch.int8
        assert ls_data["l2_weight"].dtype == torch.int8
        assert ls_data["output_weight"].dtype == torch.int8
        assert all(scale > 0 for scale in ls_data["scales"].values())
