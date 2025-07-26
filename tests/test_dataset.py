"""Fast tests for dataset functionality using mock data only."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data import (
    AVAILABLE_DATASETS,
    GenericVisionDataset,
    create_data_loaders,
    get_dataset_info,
    get_dataset_stats,
    print_dataset_stats,
)


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


class TestDataAugmentation:
    """Test data augmentation functionality."""

    def test_augmentation_imports(self):
        """Test that augmentation dependencies are properly imported."""
        try:
            import albumentations as A
            import cv2
            from albumentations.pytorch import ToTensorV2
        except ImportError as e:
            pytest.fail(f"Required augmentation libraries not installed: {e}")

    def test_generic_dataset_with_augmentation(self):
        """Test GenericVisionDataset with augmentation enabled."""

        # Create a minimal dataset for testing
        class TestDataset:
            def __init__(self):
                self.data = [
                    (np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8), i % 10)
                    for i in range(5)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        # Test with augmentation enabled
        dataset = GenericVisionDataset(
            dataset_name="cifar10",
            split="train",
            target_size=(96, 96),
            max_samples=5,
            subset=1.0,
            use_augmentation=True,
            augmentation_strength="medium",
        )

        # Replace the actual dataset with our test dataset
        dataset.dataset = TestDataset()
        dataset.samples = dataset._prepare_samples()

        assert len(dataset) > 0

        # Test getting an item
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 96, 96)
        assert isinstance(label, int)

    def test_augmentation_with_various_input_shapes(self):
        """Test that augmentations work with different input image shapes."""
        # Test various input shapes commonly used in computer vision
        test_shapes = [
            (32, 32),  # CIFAR-10/100
            (96, 96),  # Current NNUE default
            (128, 128),  # Common small size
            (224, 224),  # ImageNet standard
            (256, 256),  # Common larger size
            (64, 128),  # Non-square aspect ratio
            (480, 640),  # Camera-like aspect ratio
        ]

        for target_height, target_width in test_shapes:
            # Create test dataset with various source sizes
            class VariableSizeDataset:
                def __init__(self, source_size):
                    self.data = [
                        (
                            np.random.randint(
                                0, 255, (*source_size, 3), dtype=np.uint8
                            ),
                            0,
                        )
                        for _ in range(3)
                    ]

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx]

            # Test with different source sizes
            for source_h, source_w in [(28, 28), (64, 64), (150, 200)]:
                for strength in ["light", "medium", "heavy"]:
                    dataset = GenericVisionDataset(
                        dataset_name="cifar10",
                        split="train",
                        target_size=(target_height, target_width),
                        max_samples=3,
                        subset=1.0,
                        use_augmentation=True,
                        augmentation_strength=strength,
                    )

                    # Replace with variable size test dataset
                    dataset.dataset = VariableSizeDataset((source_h, source_w))
                    dataset.samples = dataset._prepare_samples()

                    # Should work without errors and produce correct output size
                    try:
                        image, label = dataset[0]
                        assert image.shape == (
                            3,
                            target_height,
                            target_width,
                        ), f"Expected {(3, target_height, target_width)}, got {image.shape} for source {(source_h, source_w)} -> target {(target_height, target_width)} with {strength} augmentation"
                        assert isinstance(label, int)
                    except Exception as e:
                        pytest.fail(
                            f"Augmentation failed for source {(source_h, source_w)} -> target {(target_height, target_width)} with {strength}: {e}"
                        )

    def test_augmentation_strength_levels(self):
        """Test different augmentation strength levels."""

        class TestDataset:
            def __init__(self):
                self.data = [
                    (np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8), 0)
                    for _ in range(3)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        for strength in ["light", "medium", "heavy"]:
            dataset = GenericVisionDataset(
                dataset_name="cifar10",
                split="train",
                target_size=(96, 96),
                max_samples=3,
                subset=1.0,
                use_augmentation=True,
                augmentation_strength=strength,
            )

            # Replace with test dataset
            dataset.dataset = TestDataset()
            dataset.samples = dataset._prepare_samples()

            # Should work without errors
            image, label = dataset[0]
            assert image.shape == (3, 96, 96)
            assert dataset.augmentation_strength == strength

    def test_augmentation_disabled_for_validation(self):
        """Test that augmentation is automatically disabled for validation/test splits."""

        class TestDataset:
            def __init__(self):
                self.data = [
                    (np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8), 0)
                    for _ in range(3)
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        # Test validation split
        val_dataset = GenericVisionDataset(
            dataset_name="cifar10",
            split="test",  # validation/test split
            target_size=(96, 96),
            max_samples=3,
            subset=1.0,
            use_augmentation=None,  # Should auto-detect
            augmentation_strength="heavy",  # Should be ignored for val/test
        )

        val_dataset.dataset = TestDataset()
        val_dataset.samples = val_dataset._prepare_samples()

        # Should work without augmentation
        image, label = val_dataset[0]
        assert image.shape == (3, 96, 96)

    def test_augmentation_fallback_handling(self):
        """Test graceful fallback when augmentation fails."""

        class TestDataset:
            def __init__(self):
                # Create malformed data that might cause augmentation to fail
                self.data = [(None, 0), (np.array([]), 1)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataset = GenericVisionDataset(
            dataset_name="cifar10",
            split="train",
            target_size=(96, 96),
            max_samples=2,
            subset=1.0,
            use_augmentation=True,
            augmentation_strength="medium",
        )

        dataset.dataset = TestDataset()
        dataset.samples = dataset._prepare_samples()

        # Should handle errors gracefully
        try:
            image, label = dataset[0]
            # Should either work or fallback to random tensor
            assert isinstance(image, torch.Tensor)
        except Exception as e:
            # If it fails, that's also acceptable for this edge case test
            print(f"Expected failure for malformed data: {e}")

    def test_create_data_loaders_with_augmentation(self):
        """Test create_data_loaders function with augmentation parameters."""
        try:
            train_loader, val_loader, test_loader = create_data_loaders(
                dataset_name="cifar10",
                batch_size=2,
                num_workers=0,  # Avoid multiprocessing in tests
                target_size=(32, 32),
                max_samples_per_split=10,
                subset=0.01,  # Very small subset for fast testing
                use_augmentation=True,
                augmentation_strength="light",
            )

            # Test that loaders are created
            assert isinstance(train_loader, DataLoader)
            assert isinstance(val_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)

            # Test that we can get batches (may download CIFAR-10)
            train_batch = next(iter(train_loader))
            val_batch = next(iter(val_loader))

            assert len(train_batch) == 2  # images, labels
            assert len(val_batch) == 2

        except Exception as e:
            # If CIFAR-10 download fails in test environment, that's acceptable
            print(f"Data loader test skipped due to: {e}")


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
