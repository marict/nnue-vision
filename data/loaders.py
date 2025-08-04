"""Data loader utilities for computer vision training."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .datasets import GenericVisionDataset, get_dataset_info


def create_data_loaders(
    dataset_name: str = "cifar10",
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Optional[Tuple[int, int]] = None,
    max_samples_per_split: Optional[int] = None,
    subset: float = 1.0,
    data_root: str = "./data/raw",
    binary_classification: Optional[dict] = None,
    use_augmentation: bool = True,
    augmentation_strength: str = "medium",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for computer vision datasets.

    Args:
        dataset_name: Name of dataset to load ('cifar10', 'cifar100')
        batch_size: Batch size for data loading
        num_workers: Number of worker processes for data loading
        target_size: Target image size (height, width)
        max_samples_per_split: Maximum samples per split (None for all)
        subset: Fraction of dataset to use (0.0 to 1.0)
        data_root: Root directory for dataset storage
        binary_classification: Dict with 'positive_classes' for binary tasks

    Returns:
        Tuple of (train_loader, validation_loader, test_loader)
    """
    dataset_info = get_dataset_info(dataset_name)
    print(f"🔄 Creating {dataset_info['name']} data loaders...")
    print(f"  • Dataset: {dataset_name}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Target size: {target_size}")
    print(f"  • Max samples per split: {max_samples_per_split or 'All'}")
    print(f"  • Subset: {subset}")

    if binary_classification:
        positive_classes = binary_classification.get("positive_classes", [])
        print(f"  • Binary classification: {positive_classes} → positive")

    # Create datasets for each split
    train_dataset = GenericVisionDataset(
        dataset_name=dataset_name,
        split="train",
        target_size=target_size,
        max_samples=max_samples_per_split,
        subset=subset,
        data_root=data_root,
        binary_classification=binary_classification,
        use_augmentation=use_augmentation,
        augmentation_strength=augmentation_strength,
    )

    # For CIFAR datasets, validation and test use the same test split
    # We can split the test set if needed, but for now use test for both
    val_dataset = GenericVisionDataset(
        dataset_name=dataset_name,
        split="test",
        target_size=target_size,
        max_samples=max_samples_per_split,
        subset=subset,
        data_root=data_root,
        binary_classification=binary_classification,
        use_augmentation=False,  # No augmentation for validation
        augmentation_strength=augmentation_strength,
    )

    test_dataset = GenericVisionDataset(
        dataset_name=dataset_name,
        split="test",
        target_size=target_size,
        max_samples=max_samples_per_split,
        subset=subset,
        data_root=data_root,
        binary_classification=binary_classification,
        use_augmentation=False,  # No augmentation for test
        augmentation_strength=augmentation_strength,
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


def get_dataset_stats(dataset_name: str = "cifar10") -> Dict[str, Any]:
    """Get basic statistics about a dataset."""
    dataset_info = get_dataset_info(dataset_name)

    return {
        "name": dataset_info["name"],
        "description": f"{dataset_info['name']} dataset for computer vision",
        "num_classes": dataset_info["num_classes"],
        "class_names": dataset_info["classes"],
        "task": f"{dataset_info['num_classes']}-class classification",
        "input_type": "RGB images",
        "input_size": dataset_info["input_size"],
        "channels": dataset_info["channels"],
        "data_source": "torchvision.datasets",
    }


def print_dataset_stats(dataset_name: str = "cifar10") -> None:
    """Print comprehensive dataset statistics."""
    stats = get_dataset_stats(dataset_name)

    print("\n" + "=" * 60)
    print(f"📊 {stats['name'].upper()} DATASET STATISTICS")
    print("=" * 60)

    print(f"Dataset Name: {stats['name']}")
    print(f"Description: {stats['description']}")
    print(f"Task: {stats['task']}")
    print(f"Input Type: {stats['input_type']}")
    print(
        f"Input Size: {stats['input_size'][0]}x{stats['input_size'][1]}x{stats['channels']}"
    )
    print(f"Data Source: {stats['data_source']}")
    print(f"Number of Classes: {stats['num_classes']}")

    # Print class names (truncate if too many)
    if len(stats["class_names"]) <= 20:
        print(f"Class Names: {', '.join(stats['class_names'])}")
    else:
        print(
            f"Class Names: {', '.join(stats['class_names'][:10])} ... (and {len(stats['class_names'])-10} more)"
        )


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

    # Get class names from the data loader's dataset
    if hasattr(data_loader.dataset, "class_names"):
        class_names = data_loader.dataset.class_names
        label_distribution = {
            class_names[label.item()]: count.item()
            for label, count in zip(unique_labels, label_counts)
        }
    else:
        # Fallback to generic class names
        label_distribution = {
            f"class_{label.item()}": count.item()
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
