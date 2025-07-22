from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SyntheticVisualWakeWordsDataset(Dataset):
    """
    Synthetic Visual Wake Words dataset for demonstration.

    This creates random images with synthetic labels to demonstrate the training pipeline
    without requiring TensorFlow datasets. In a real implementation, you would replace
    this with actual Visual Wake Words data.
    """

    def __init__(
        self, split="train", transform=None, target_size=(96, 96), num_samples=1000
    ):
        """
        Args:
            split (str): Dataset split ('train', 'validation', 'test')
            transform: PyTorch transforms to apply to images
            target_size (tuple): Target image size (width, height)
            num_samples (int): Number of synthetic samples to generate
        """
        self.split = split
        self.target_size = target_size
        self.num_samples = num_samples

        # Default transforms if none provided
        if transform is None:
            if split == "train":
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(target_size),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                        ),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(target_size),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.transform = transform

        print(f"Created synthetic {split} dataset with {num_samples} samples")
        print(
            "NOTE: This is synthetic data for demonstration. Replace with real Visual Wake Words dataset for actual training."
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic image with some structure
        np.random.seed(idx)  # For reproducibility

        # Create a synthetic image with some patterns
        image_np = np.random.randint(
            0, 256, (self.target_size[1], self.target_size[0], 3), dtype=np.uint8
        )

        # Add some structure to make classification possible
        if idx % 2 == 0:  # "person" class
            # Add a vertical rectangle in the center (person-like shape)
            center_x, center_y = self.target_size[0] // 2, self.target_size[1] // 2
            image_np[center_y - 20 : center_y + 20, center_x - 8 : center_x + 8, :] = [
                255,
                128,
                64,
            ]
            label = 1
        else:  # "no person" class
            # Add random horizontal lines (background-like)
            for i in range(0, self.target_size[1], 10):
                image_np[i : i + 2, :, :] = [64, 128, 255]
            label = 0

        # Convert to PIL Image
        image = Image.fromarray(image_np)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def create_data_loaders(batch_size=32, num_workers=4, target_size=(96, 96), subset=1.0):
    """
    Create train, validation, and test data loaders for synthetic Visual Wake Words dataset.

    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        target_size (tuple): Target image size (width, height)
        subset (float): Fraction of dataset to use (0.0 < subset <= 1.0)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

    # Validate subset parameter
    if not (0.0 < subset <= 1.0):
        raise ValueError(f"subset must be between 0.0 and 1.0, got {subset}")

    # Base dataset sizes
    base_train_samples = 5000
    base_val_samples = 1000
    base_test_samples = 1000

    # Calculate actual sample sizes based on subset
    train_samples = int(base_train_samples * subset)
    val_samples = int(base_val_samples * subset)
    test_samples = int(base_test_samples * subset)

    # Ensure we have at least 1 sample per dataset
    train_samples = max(1, train_samples)
    val_samples = max(1, val_samples)
    test_samples = max(1, test_samples)

    # Create datasets with different sizes for train/val/test
    train_dataset = SyntheticVisualWakeWordsDataset(
        split="train", target_size=target_size, num_samples=train_samples
    )
    val_dataset = SyntheticVisualWakeWordsDataset(
        split="validation", target_size=target_size, num_samples=val_samples
    )
    test_dataset = SyntheticVisualWakeWordsDataset(
        split="test", target_size=target_size, num_samples=test_samples
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Created data loaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def get_dataset_stats():
    """
    Get basic statistics about the synthetic dataset.
    """
    print("Synthetic dataset statistics:")
    print("- This is a demonstration dataset with synthetic images")
    print("- Person class: vertical rectangle in center")
    print("- No-person class: horizontal lines")
    print("- Image size: 96x96 RGB")
    print("- Classes: 2 (person=1, no-person=0)")
    print()
    print("To use real Visual Wake Words data:")
    print("1. Install tensorflow-datasets: pip install tensorflow-datasets")
    print("2. Replace this synthetic dataset with tfds.load('visual_wake_words')")


# Real Visual Wake Words dataset loader (commented out due to TensorFlow dependency issues)
"""
import tensorflow_datasets as tfds

class VisualWakeWordsDataset(Dataset):
    def __init__(self, split='train', transform=None, target_size=(96, 96)):
        self.split = split
        self.target_size = target_size
        
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
            
        # Load dataset using TensorFlow Datasets
        print(f"Loading Visual Wake Words {split} dataset...")
        self.dataset = tfds.load(
            "visual_wake_words", 
            split=split, 
            as_supervised=True,
            download=True,
            shuffle_files=(split == 'train')
        )
        
        # Convert to list for indexing
        print("Converting dataset to list...")
        self.data_list = list(self.dataset.take(-1))
        print(f"Loaded {len(self.data_list)} samples from {split} split")
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        tf_image, tf_label = self.data_list[idx]
        
        # Convert TensorFlow tensors to NumPy
        image_np = tf_image.numpy()
        label = tf_label.numpy()
        
        # Convert to PIL Image for transforms
        if image_np.dtype == np.uint8:
            image = Image.fromarray(image_np)
        else:
            image = Image.fromarray((image_np * 255).astype(np.uint8))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)
"""


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing synthetic Visual Wake Words dataset loader...")

    # Get dataset statistics
    get_dataset_stats()

    # Create data loaders (full dataset)
    print("\nTesting with full dataset (subset=1.0)...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=8, num_workers=0
    )

    # Test a batch
    print("\nTesting a batch...")
    batch_images, batch_labels = next(iter(train_loader))
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Labels in batch: {batch_labels.tolist()}")
    print(f"Image value range: [{batch_images.min():.3f}, {batch_images.max():.3f}]")

    # Check class distribution
    class_0_count = (batch_labels == 0).sum().item()
    class_1_count = (batch_labels == 1).sum().item()
    print(
        f"Class distribution in batch: class 0 (no-person): {class_0_count}, class 1 (person): {class_1_count}"
    )

    # Test with subset functionality
    print("\nTesting subset functionality...")
    print("Creating smaller dataset with subset=0.1...")
    small_train_loader, small_val_loader, small_test_loader = create_data_loaders(
        batch_size=8, num_workers=0, subset=0.1
    )

    print("Dataset loader test completed successfully!")
