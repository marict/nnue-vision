"""
Tests for dataset loading and processing functionality.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import (VWW_CLASS_NAMES, VisualWakeWordsDataset, create_data_loaders,
                  get_dataset_stats, print_dataset_stats)
from tests.conftest import assert_tensor_shape


class TestVisualWakeWordsDataset:
    """Test the VisualWakeWordsDataset class."""

    def test_dataset_creation(self):
        """Test that we can create a dataset instance."""
        dataset = VisualWakeWordsDataset(max_samples=5)
        assert len(dataset) == 5
        assert dataset.split == "train"
        assert dataset.target_size == (96, 96)

    def test_dataset_splits(self):
        """Test different dataset splits."""
        for split in ["train", "validation", "test"]:
            dataset = VisualWakeWordsDataset(
                split=split, target_size=(96, 96), max_samples=5
            )
            assert dataset.split == split
            assert len(dataset) == 5

    def test_dataset_indexing(self):
        """Test that dataset supports indexing."""
        dataset = VisualWakeWordsDataset(max_samples=5)

        # Test basic indexing
        image, label = dataset[0]
        assert_tensor_shape(image, (3, 96, 96))
        assert isinstance(label, int)
        assert label in [0, 1]

        # Test multiple indices
        for i in range(min(3, len(dataset))):
            image, label = dataset[i]
            assert_tensor_shape(image, (3, 96, 96))
            assert label in [0, 1]

    def test_dataset_target_size(self):
        """Test different target sizes."""
        dataset = VisualWakeWordsDataset(max_samples=5, target_size=(128, 128))
        image, label = dataset[0]
        assert_tensor_shape(image, (3, 128, 128))

    def test_class_names(self):
        """Test that class names are correctly defined."""
        assert len(VWW_CLASS_NAMES) == 2
        assert "person" in VWW_CLASS_NAMES
        assert "no_person" in VWW_CLASS_NAMES

    def test_data_types(self):
        """Test that dataset returns correct data types."""
        dataset = VisualWakeWordsDataset(max_samples=3)

        for i in range(len(dataset)):
            image, label = dataset[i]

            # Image should be float tensor
            assert isinstance(image, torch.Tensor)
            assert image.dtype == torch.float32

            # Label should be integer
            assert isinstance(label, int)
            assert label in [0, 1]


class TestDataLoaders:
    """Test data loader creation and functionality."""

    def test_create_data_loaders_basic(self):
        """Test basic data loader creation."""
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=4, max_samples_per_split=8, num_workers=0, use_synthetic=True
        )

        # Check that loaders were created
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Check batch sizes
        assert train_loader.batch_size == 4
        assert val_loader.batch_size == 4
        assert test_loader.batch_size == 4

    def test_data_loader_output_shapes(self):
        """Test that data loaders produce correct output shapes."""
        train_loader, _, _ = create_data_loaders(
            batch_size=4,
            target_size=(96, 96),
            max_samples_per_split=8,
            num_workers=0,
            use_synthetic=True,
        )

        # Get one batch
        images, labels = next(iter(train_loader))

        # Check shapes
        assert_tensor_shape(images, (4, 3, 96, 96))
        assert_tensor_shape(labels, (4,))

        # Check data types
        assert images.dtype == torch.float32
        assert labels.dtype == torch.long

    def test_data_loader_different_batch_sizes(self):
        """Test data loaders with different batch sizes."""
        batch_sizes = [1, 2, 8, 16]

        for batch_size in batch_sizes:
            train_loader, _, _ = create_data_loaders(
                batch_size=batch_size,
                max_samples_per_split=16,
                num_workers=0,
                use_synthetic=True,
            )

            images, labels = next(iter(train_loader))
            assert images.shape[0] == batch_size
            assert len(labels) == batch_size

    def test_data_loader_different_image_sizes(self):
        """Test data loaders with different target image sizes."""
        sizes = [(64, 64), (96, 96), (128, 128)]

        for size in sizes:
            train_loader, _, _ = create_data_loaders(
                batch_size=4,
                target_size=size,
                max_samples_per_split=8,
                num_workers=0,
                use_synthetic=True,
            )

            images, labels = next(iter(train_loader))
            expected_shape = (4, 3, size[0], size[1])
            assert_tensor_shape(images, expected_shape)


class TestDatasetStatistics:
    """Test dataset statistics and utility functions."""

    def test_get_dataset_stats(self):
        """Test dataset statistics retrieval."""
        stats = get_dataset_stats()

        # Check required fields
        assert "name" in stats
        assert "description" in stats
        assert "num_classes" in stats
        assert "class_names" in stats

        # Check values
        assert stats["num_classes"] == 2
        assert len(stats["class_names"]) == 2

    def test_print_dataset_stats(self, _capsys):
        """Test dataset statistics printing."""
        print_dataset_stats()

        # Capture output
        captured = _capsys.readouterr()
        output = captured.out

        # Check that key information is printed
        assert "Visual Wake Words" in output
        assert "person" in output
        assert "no_person" in output

    def test_data_loader_iteration(self):
        """Test iterating through data loaders."""
        train_loader, _, _ = create_data_loaders(
            batch_size=4, max_samples_per_split=12, num_workers=0, use_synthetic=True
        )

        batch_count = 0
        total_samples = 0

        for images, labels in train_loader:
            batch_count += 1
            total_samples += len(labels)

            # Check batch properties
            assert isinstance(images, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert len(images) == len(labels)

            # Limit iteration for testing
            if batch_count >= 3:
                break

        assert batch_count > 0
        assert total_samples > 0


class TestDatasetCompatibility:
    """Test dataset compatibility with PyTorch components."""

    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        dataset = VisualWakeWordsDataset(max_samples=8)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Should be able to iterate
        for images, labels in loader:
            assert_tensor_shape(images, (4, 3, 96, 96))
            assert_tensor_shape(labels, (4,))
            break  # Just test one batch

    def test_dataset_with_transforms(self):
        """Test that dataset handles transforms correctly."""
        dataset = VisualWakeWordsDataset(max_samples=5, target_size=(64, 64))

        # Get sample
        image, label = dataset[0]

        # Check that transforms were applied
        assert_tensor_shape(image, (3, 64, 64))
        assert torch.all(image >= -3) and torch.all(
            image <= 3
        )  # Reasonable range after normalization
        assert isinstance(label, int)

    def test_dataset_consistency(self):
        """Test that dataset returns consistent results."""
        dataset = VisualWakeWordsDataset(max_samples=5)

        # Get same sample twice
        image1, label1 = dataset[0]
        image2, label2 = dataset[0]

        # Should be identical (assuming no randomness in transforms for non-train split)
        if dataset.split != "train":
            assert torch.allclose(image1, image2)
            assert label1 == label2
