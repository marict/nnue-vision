"""Fast tests for dataset functionality using mock data only."""

import pytest
import torch
from torch.utils.data import DataLoader

from data import (AVAILABLE_DATASETS, get_dataset_info, get_dataset_stats,
                  print_dataset_stats)
from tests.conftest import MockDataset


class TestMockDataset:
    """Test the fast MockDataset class for unit testing."""

    def test_mock_dataset_basic(self, fast_mock_dataset):
        """Test basic mock dataset functionality."""
        assert len(fast_mock_dataset) == 8

        image, label = fast_mock_dataset[0]
        assert image.shape == (3, 96, 96)
        assert isinstance(label, int)
        assert 0 <= label < 10

    def test_mock_dataset_deterministic(self, fast_mock_dataset):
        """Test that mock dataset is deterministic."""
        image1, label1 = fast_mock_dataset[0]
        image2, label2 = fast_mock_dataset[0]

        assert torch.allclose(image1, image2)
        assert label1 == label2

    def test_mock_dataset_different_indices(self, fast_mock_dataset):
        """Test that different indices give different data."""
        image1, label1 = fast_mock_dataset[0]
        image2, label2 = fast_mock_dataset[1]

        # Images should be different (not identical)
        assert not torch.allclose(image1, image2)


class TestFastDataLoaders:
    """Test fast data loaders for unit testing."""

    def test_fast_data_loaders_basic(self, fast_data_loaders):
        """Test basic fast data loader functionality."""
        train_loader, val_loader, test_loader = fast_data_loaders

        # Check loaders exist
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

        # Test train loader
        batch = next(iter(train_loader))
        images, labels = batch

        assert images.shape == (2, 3, 96, 96)  # batch_size=2
        assert labels.shape == (2,)
        assert torch.all((labels == 0) | (labels == 1))  # Binary labels

    def test_fast_data_loaders_all_splits(self, fast_data_loaders):
        """Test that all splits work."""
        train_loader, val_loader, test_loader = fast_data_loaders

        # Test each loader can produce a batch
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            images, labels = batch
            assert images.shape[0] == 2  # batch_size
            assert len(labels) == 2


class TestDatasetInfo:
    """Test dataset information and statistics functions."""

    def test_get_dataset_info(self):
        """Test getting dataset information."""
        info = get_dataset_info("cifar10")

        assert info["name"] == "CIFAR-10"
        assert info["num_classes"] == 10
        assert len(info["classes"]) == 10
        assert "airplane" in info["classes"]
        assert info["input_size"] == (32, 32)
        assert info["channels"] == 3

    def test_get_dataset_info_invalid(self):
        """Test getting info for invalid dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_info("invalid_dataset")

    def test_get_dataset_stats(self):
        """Test getting dataset statistics."""
        stats = get_dataset_stats("cifar10")

        assert stats["name"] == "CIFAR-10"
        assert stats["num_classes"] == 10
        assert stats["task"] == "10-class classification"
        assert stats["data_source"] == "torchvision.datasets"

    def test_print_dataset_stats(self, capsys):
        """Test printing dataset statistics."""
        print_dataset_stats("cifar10")

        captured = capsys.readouterr()
        assert "CIFAR-10" in captured.out
        assert "10-class classification" in captured.out

    def test_available_datasets(self):
        """Test that available datasets are properly defined."""
        assert "cifar10" in AVAILABLE_DATASETS
        assert "cifar100" in AVAILABLE_DATASETS

        for dataset_name, info in AVAILABLE_DATASETS.items():
            assert "name" in info
            assert "classes" in info
            assert "num_classes" in info
            assert "input_size" in info
            assert "channels" in info
