"""Data loading and dataset utilities for NNUE-Vision.

This package provides:
- Generic computer vision dataset loading (CIFAR-10, CIFAR-100, etc.)
- Data preprocessing and transformations
- Dataset inspection and visualization tools

Example usage:
    from data import create_data_loaders, get_dataset_info

    # Create data loaders for CIFAR-10
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="cifar10",
        batch_size=32,
        max_samples_per_split=100
    )

    # Get dataset information
    info = get_dataset_info("cifar10")
    print(f"Classes: {info['classes']}")

    # Binary classification example (vehicles vs non-vehicles)
    binary_config = {
        'positive_classes': ['airplane', 'automobile', 'ship', 'truck']
    }
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="cifar10",
        binary_classification=binary_config
    )
"""

# Direct imports without circular dependency
from .datasets import AVAILABLE_DATASETS, GenericVisionDataset, get_dataset_info
from .loaders import create_data_loaders, get_dataset_stats, print_dataset_stats

__all__ = [
    # Dataset classes
    "GenericVisionDataset",
    # Data loaders
    "create_data_loaders",
    # Utilities
    "get_dataset_stats",
    "print_dataset_stats",
    "get_dataset_info",
    "AVAILABLE_DATASETS",
]
