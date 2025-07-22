"""
Shared pytest fixtures and test utilities for NNUE-Vision tests.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from dataset import SyntheticVisualWakeWordsDataset, create_data_loaders
from model import ModelParams, SimpleCNN


@pytest.fixture
def device():
    """Return the best available device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model_params():
    """Return standard model parameters for testing."""
    return ModelParams(input_size=(96, 96), num_classes=2, learning_rate=1e-3)


@pytest.fixture
def simple_model(model_params):
    """Return a SimpleCNN model for testing."""
    return SimpleCNN(model_params)


@pytest.fixture
def small_dataset():
    """Return a small synthetic dataset for testing."""
    return SyntheticVisualWakeWordsDataset(
        split="train", target_size=(96, 96), num_samples=10
    )


@pytest.fixture
def sample_batch(device):
    """Return a sample batch of synthetic image data."""
    batch_size = 4
    images = torch.randn(batch_size, 3, 96, 96, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)
    return images, labels


@pytest.fixture
def data_loaders():
    """Return small data loaders for testing."""
    return create_data_loaders(
        batch_size=4,
        num_workers=0,  # No multiprocessing for tests
        target_size=(96, 96),
    )


@pytest.fixture
def trained_model(simple_model, data_loaders, device):
    """Return a model that has been trained for a few steps."""
    simple_model.to(device)
    simple_model.train()

    train_loader, _, _ = data_loaders
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

    # Train for a few steps
    for i, (images, labels) in enumerate(train_loader):
        if i >= 2:  # Only train for 2 batches
            break

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = simple_model(images)
        loss = simple_model.loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

    return simple_model


@pytest.fixture
def temp_model_path(tmp_path):
    """Return a temporary path for saving models."""
    return tmp_path / "test_model.pt"


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
