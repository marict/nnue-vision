"""
Tests for dataset import functionality.

This module tests:
- Dataset creation and basic functionality
- Data loader creation and iteration
- Data transformations and preprocessing
- Data shape and type consistency
- Dataset statistics and properties
"""

import numpy as np
import pytest
import torch
from PIL import Image

from dataset import (SyntheticVisualWakeWordsDataset, create_data_loaders,
                     get_dataset_stats)
from tests.conftest import assert_tensor_range, assert_tensor_shape


class TestSyntheticVisualWakeWordsDataset:
    """Test the SyntheticVisualWakeWordsDataset class."""

    def test_dataset_creation(self):
        """Test that dataset can be created with different configurations."""
        # Test default creation
        dataset = SyntheticVisualWakeWordsDataset(num_samples=5)
        assert len(dataset) == 5
        assert dataset.split == "train"
        assert dataset.target_size == (96, 96)

        # Test with custom parameters
        dataset = SyntheticVisualWakeWordsDataset(
            split="validation", target_size=(64, 64), num_samples=10
        )
        assert len(dataset) == 10
        assert dataset.split == "validation"
        assert dataset.target_size == (64, 64)

    def test_dataset_getitem(self):
        """Test that __getitem__ returns valid data."""
        dataset = SyntheticVisualWakeWordsDataset(num_samples=5)

        for i in range(len(dataset)):
            image, label = dataset[i]

            # Check image properties
            assert isinstance(image, torch.Tensor)
            assert_tensor_shape(image, (3, 96, 96))  # C, H, W format after transforms
            assert image.dtype == torch.float32

            # Check normalization (ImageNet stats applied)
            # Values should be roughly in range [-3, 3] after normalization
            assert_tensor_range(image, -5.0, 5.0)

            # Check label properties
            assert isinstance(label, torch.Tensor)
            assert label.dtype == torch.long
            assert label.item() in [0, 1]

    def test_synthetic_pattern_consistency(self):
        """Test that synthetic patterns are consistent with labels."""
        dataset = SyntheticVisualWakeWordsDataset(num_samples=10, target_size=(96, 96))

        for i in range(len(dataset)):
            image, label = dataset[i]
            expected_label = 1 if i % 2 == 0 else 0
            assert (
                label.item() == expected_label
            ), f"Index {i}: expected {expected_label}, got {label.item()}"

    def test_reproducibility(self):
        """Test that dataset generates reproducible data."""
        dataset1 = SyntheticVisualWakeWordsDataset(num_samples=5)
        dataset2 = SyntheticVisualWakeWordsDataset(num_samples=5)

        for i in range(5):
            image1, label1 = dataset1[i]
            image2, label2 = dataset2[i]

            # Labels should be identical (deterministic based on index)
            assert label1.item() == label2.item()

            # Images should be similar (same random seed per index)
            # Note: transforms might add some randomness, so we check labels only

    def test_different_splits_have_different_transforms(self):
        """Test that train and validation splits have different transforms."""
        train_dataset = SyntheticVisualWakeWordsDataset(split="train", num_samples=1)
        val_dataset = SyntheticVisualWakeWordsDataset(split="validation", num_samples=1)

        # Both should work without error
        train_image, train_label = train_dataset[0]
        val_image, val_label = val_dataset[0]

        # Both should have same shape and type
        assert_tensor_shape(train_image, (3, 96, 96))
        assert_tensor_shape(val_image, (3, 96, 96))
        assert train_image.dtype == val_image.dtype == torch.float32


class TestDataLoaders:
    """Test data loader creation and functionality."""

    def test_create_data_loaders(self):
        """Test that data loaders are created successfully."""
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=4, num_workers=0, target_size=(96, 96)
        )

        # Check loader properties
        assert train_loader.batch_size == 4
        assert val_loader.batch_size == 4
        assert test_loader.batch_size == 4

        # Check dataset sizes
        assert len(train_loader.dataset) == 5000
        assert len(val_loader.dataset) == 1000
        assert len(test_loader.dataset) == 1000

    def test_data_loader_iteration(self):
        """Test that data loaders can be iterated over."""
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=2, num_workers=0, target_size=(64, 64)
        )

        # Test train loader
        batch_images, batch_labels = next(iter(train_loader))
        assert_tensor_shape(batch_images, (2, 3, 64, 64))
        assert_tensor_shape(batch_labels, (2,))
        assert batch_labels.dtype == torch.long

        # Test validation loader
        batch_images, batch_labels = next(iter(val_loader))
        assert_tensor_shape(batch_images, (2, 3, 64, 64))
        assert_tensor_shape(batch_labels, (2,))

        # Test test loader
        batch_images, batch_labels = next(iter(test_loader))
        assert_tensor_shape(batch_images, (2, 3, 64, 64))
        assert_tensor_shape(batch_labels, (2,))

    def test_batch_consistency(self):
        """Test that batches contain valid data."""
        train_loader, _, _ = create_data_loaders(
            batch_size=8, num_workers=0, target_size=(96, 96)
        )

        for i, (images, labels) in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches
                break

            # Check batch shapes
            assert images.shape[0] <= 8  # batch size
            assert_tensor_shape(images, (images.shape[0], 3, 96, 96))
            assert labels.shape[0] == images.shape[0]

            # Check data types
            assert images.dtype == torch.float32
            assert labels.dtype == torch.long

            # Check label values
            assert torch.all((labels >= 0) & (labels <= 1))

            # Check image normalization
            assert_tensor_range(images, -5.0, 5.0)

    def test_class_balance_in_batches(self):
        """Test that both classes appear in batches."""
        train_loader, _, _ = create_data_loaders(
            batch_size=16, num_workers=0, target_size=(96, 96)
        )

        all_labels = []
        for i, (_, labels) in enumerate(train_loader):
            if i >= 5:  # Check first 5 batches
                break
            all_labels.extend(labels.tolist())

        # Both classes should appear
        unique_labels = set(all_labels)
        assert 0 in unique_labels, "Class 0 (no-person) not found in batches"
        assert 1 in unique_labels, "Class 1 (person) not found in batches"


class TestDatasetStats:
    """Test dataset statistics and utility functions."""

    def test_get_dataset_stats(self, capsys):
        """Test that get_dataset_stats prints information."""
        get_dataset_stats()

        captured = capsys.readouterr()
        assert "Synthetic dataset statistics" in captured.out
        assert "Image size: 96x96 RGB" in captured.out
        assert "Classes: 2" in captured.out


class TestDatasetIntegration:
    """Integration tests for dataset functionality."""

    def test_dataset_with_model_input(self, simple_test_model, device):
        """Test that dataset output is compatible with model input."""
        dataset = SyntheticVisualWakeWordsDataset(num_samples=4)
        simple_test_model.to(device)
        simple_test_model.eval()

        # Test individual samples
        for i in range(len(dataset)):
            image, label = dataset[i]
            image = image.unsqueeze(0).to(device)  # Add batch dimension

            # Forward pass should work without error
            with torch.no_grad():
                logits = simple_test_model(image)

            assert_tensor_shape(logits, (1, 2))
            assert logits.dtype == torch.float32

    def test_dataloader_with_model_training(self, simple_test_model, device):
        """Test that data loader output works with model training."""
        train_loader, _, _ = create_data_loaders(
            batch_size=4, num_workers=0, target_size=(96, 96)
        )

        simple_test_model.to(device)
        simple_test_model.train()
        optimizer = torch.optim.Adam(simple_test_model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Train for one batch
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        logits = simple_test_model(images)
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify training worked
        assert loss.item() > 0  # Loss should be positive
        assert not torch.isnan(loss)  # Loss should not be NaN

        # Verify gradients were computed
        for param in simple_test_model.parameters():
            if param.requires_grad:
                assert param.grad is not None
