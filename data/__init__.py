"""Data loading and dataset utilities for NNUE-Vision.

This package provides:
- Visual Wake Words dataset loading
- Data preprocessing and transformations
- Dataset inspection and visualization tools

Example usage:
    from data import create_data_loaders, DatasetInspector

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=32,
        max_samples_per_split=100
    )

    # Inspect dataset
    inspector = DatasetInspector()
    inspector.print_overview()
    inspector.show_sample_images()
"""

# Re‐export DatasetInspector for convenience
from .dataset_inspector import DatasetInspector
from .datasets import VWW_CLASS_NAMES, VisualWakeWordsDataset
from .loaders import (create_data_loaders, get_dataset_stats,
                      print_dataset_stats)

__all__ = [
    # Dataset classes
    "VisualWakeWordsDataset",
    # Data loaders
    "create_data_loaders",
    # Utilities
    "get_dataset_stats",
    "print_dataset_stats",
    "VWW_CLASS_NAMES",
    # Inspector
    "DatasetInspector",
]
