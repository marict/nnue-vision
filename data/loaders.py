"""Data loaders for Visual Wake Words dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .datasets import VWW_CLASS_NAMES, VisualWakeWordsDataset


def create_data_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (96, 96),
    max_samples_per_split: Optional[int] = None,
    streaming: bool = True,
    subset: float = 1.0,
    use_synthetic: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for Visual Wake Words dataset.

    Args:
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        target_size: Target image size (height, width)
        max_samples_per_split: Maximum samples per split (None for all)
        streaming: Whether to use streaming mode
        subset: Fraction of dataset to use (0.0 to 1.0)
        use_synthetic: Whether to use synthetic data (for tests)

    Returns:
        Tuple of (train_loader, validation_loader, test_loader)
    """
    print(f"🔄 Creating Visual Wake Words data loaders...")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Target size: {target_size}")
    print(f"  • Max samples per split: {max_samples_per_split or 'All'}")
    if use_synthetic:
        print(f"  • Using synthetic data for testing")
    else:
        print(f"  • Streaming: {streaming}")
        print(f"  • Subset: {subset}")

    # Calculate max_samples based on subset if not explicitly provided
    effective_max_samples = max_samples_per_split
    if subset < 1.0 and max_samples_per_split is None:
        # If subset is specified but max_samples is not, we'll let the dataset handle it
        # by passing subset as a parameter to control the fraction of data used
        pass

    # Create datasets for each split
    train_dataset = VisualWakeWordsDataset(
        split="train",
        target_size=target_size,
        max_samples=effective_max_samples,
        streaming=streaming,
        subset=subset,
        use_synthetic=use_synthetic,
    )

    val_dataset = VisualWakeWordsDataset(
        split="validation",
        target_size=target_size,
        max_samples=effective_max_samples,
        streaming=streaming,
        subset=subset,
        use_synthetic=use_synthetic,
    )

    test_dataset = VisualWakeWordsDataset(
        split="test",
        target_size=target_size,
        max_samples=effective_max_samples,
        streaming=streaming,
        subset=subset,
        use_synthetic=use_synthetic,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print("✅ Data loaders created successfully!")
    return train_loader, val_loader, test_loader


def get_dataset_stats() -> Dict[str, Any]:
    """Get basic statistics about the Visual Wake Words dataset."""
    return {
        "name": "Visual Wake Words",
        "description": "Visual Wake Words dataset for person detection (via HuggingFace)",
        "num_classes": len(VWW_CLASS_NAMES),
        "class_names": VWW_CLASS_NAMES,
        "task": "Binary classification (person detection)",
        "input_type": "RGB images",
        "data_source": "HuggingFace datasets",
    }


def print_dataset_stats() -> None:
    """Print comprehensive dataset statistics."""
    stats = get_dataset_stats()

    print("\n" + "=" * 60)
    print("📊 VISUAL WAKE WORDS DATASET STATISTICS")
    print("=" * 60)

    print(f"Dataset Name: {stats['name']}")
    print(f"Description: {stats['description']}")
    print(f"Task: {stats['task']}")
    print(f"Input Type: {stats['input_type']}")
    print(f"Data Source: {stats['data_source']}")
    print(f"Number of Classes: {stats['num_classes']}")
    print(f"Class Names: {', '.join(stats['class_names'])}")


def calculate_dataset_statistics(data_loader: DataLoader) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a data loader.

    Args:
        data_loader: PyTorch DataLoader to analyze

    Returns:
        Dictionary containing dataset statistics
    """
    print("🔍 Calculating dataset statistics...")

    # Get one batch to analyze
    images, labels = next(iter(data_loader))

    # Basic info
    batch_size = images.shape[0]
    image_shape = images.shape[1:]  # (C, H, W)

    # Image statistics
    img_mean = images.mean(dim=[0, 2, 3])  # Mean per channel
    img_std = images.std(dim=[0, 2, 3])  # Std per channel
    img_min = images.min()
    img_max = images.max()

    # Label statistics
    unique_labels, label_counts = torch.unique(labels, return_counts=True)
    label_distribution = {
        VWW_CLASS_NAMES[label.item()]: count.item()
        for label, count in zip(unique_labels, label_counts)
    }

    stats = {
        "batch_size": batch_size,
        "image_shape": tuple(image_shape),
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "image_mean_per_channel": img_mean.tolist(),
        "image_std_per_channel": img_std.tolist(),
        "image_min": img_min.item(),
        "image_max": img_max.item(),
        "unique_labels": unique_labels.tolist(),
        "label_distribution": label_distribution,
        "total_samples_in_batch": len(labels),
    }

    return stats


def print_loader_statistics(stats: Dict[str, Any]) -> None:
    """Print data loader statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("📈 DATA LOADER STATISTICS")
    print("=" * 60)

    print(f"Batch Size: {stats['batch_size']}")
    print(f"Image Shape: {stats['image_shape']} (C×H×W)")
    print(f"Image Data Type: {stats['image_dtype']}")
    print(f"Label Data Type: {stats['label_dtype']}")

    print(f"\n📊 Image Statistics:")
    print(f"  • Value Range: [{stats['image_min']:.3f}, {stats['image_max']:.3f}]")
    print(
        f"  • Mean per Channel: [R:{stats['image_mean_per_channel'][0]:.3f}, G:{stats['image_mean_per_channel'][1]:.3f}, B:{stats['image_mean_per_channel'][2]:.3f}]"
    )
    print(
        f"  • Std per Channel: [R:{stats['image_std_per_channel'][0]:.3f}, G:{stats['image_std_per_channel'][1]:.3f}, B:{stats['image_std_per_channel'][2]:.3f}]"
    )

    print(f"\n🏷️  Label Statistics:")
    print(f"  • Unique Labels: {stats['unique_labels']}")
    print(f"  • Class Distribution in Batch:")
    for class_name, count in stats["label_distribution"].items():
        percentage = (count / stats["total_samples_in_batch"]) * 100
        print(f"    - {class_name}: {count} samples ({percentage:.1f}%)")
