"""Generic dataset implementations for computer vision tasks."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Available datasets
AVAILABLE_DATASETS = {
    "cifar10": {
        "name": "CIFAR-10",
        "classes": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        "num_classes": 10,
        "input_size": (32, 32),
        "channels": 3,
    },
    "cifar100": {
        "name": "CIFAR-100",
        "classes": [f"class_{i}" for i in range(100)],  # Simplified for now
        "num_classes": 100,
        "input_size": (32, 32),
        "channels": 3,
    },
    "imagenet": {
        "name": "ImageNet",
        "classes": [f"n{i:08d}" for i in range(1000)],  # WordNet IDs simplified
        "num_classes": 1000,
        "input_size": (224, 224),
        "channels": 3,
    },
}

__all__ = ["GenericVisionDataset", "AVAILABLE_DATASETS", "get_dataset_info"]


def get_dataset_info(dataset_name: str) -> dict:
    """Get information about a dataset."""
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(AVAILABLE_DATASETS.keys())}"
        )
    return AVAILABLE_DATASETS[dataset_name]


class GenericVisionDataset(Dataset):
    """Generic vision dataset supporting multiple computer vision benchmarks."""

    def __init__(
        self,
        dataset_name: str = "cifar10",
        split: str = "train",
        target_size: Tuple[int, int] = (96, 96),
        max_samples: Optional[int] = None,
        subset: float = 1.0,
        data_root: str = "./data/raw",
        binary_classification: Optional[dict] = None,
    ):
        """
        Initialize generic vision dataset.

        Args:
            dataset_name: Name of dataset to load ('cifar10', 'cifar100')
            split: Dataset split ('train', 'test', 'val')
            target_size: Target image size (height, width)
            max_samples: Maximum number of samples to load (None for all)
            subset: Fraction of dataset to use (0.0 to 1.0)
            data_root: Root directory for dataset storage
            binary_classification: Dict with 'positive_classes' list for binary tasks
        """
        self.dataset_name = dataset_name
        self.split = split
        self.target_size = target_size
        self.max_samples = max_samples
        self.subset = subset
        self.data_root = Path(data_root)
        self.binary_classification = binary_classification

        # Get dataset info
        self.dataset_info = get_dataset_info(dataset_name)

        # Setup binary classification if specified
        if binary_classification:
            positive_classes = binary_classification.get("positive_classes", [])
            self.class_names = ["negative", "positive"]
            self.num_classes = 2
            self.positive_class_indices = set()

            # Convert class names to indices
            for class_name in positive_classes:
                if class_name in self.dataset_info["classes"]:
                    idx = self.dataset_info["classes"].index(class_name)
                    self.positive_class_indices.add(idx)
                else:
                    print(f"Warning: Class '{class_name}' not found in {dataset_name}")

            print(
                f"Binary classification: {len(self.positive_class_indices)} positive classes"
            )
        else:
            self.class_names = self.dataset_info["classes"]
            self.num_classes = self.dataset_info["num_classes"]
            self.positive_class_indices = None

        # Image preprocessing pipeline
        self.transform = self._build_transform()

        # Load dataset
        print(f"🔄 Loading {self.dataset_info['name']} dataset ({split} split)...")
        self.dataset = self._load_dataset()

        # Apply subset and max_samples
        self.samples = self._prepare_samples()

        print(f"✅ Loaded {len(self.samples)} samples from {self.dataset_info['name']}")

    def _build_transform(self) -> transforms.Compose:
        """Build image preprocessing pipeline."""
        transform_list = []

        # Resize if needed
        if self.target_size != self.dataset_info["input_size"]:
            transform_list.append(transforms.Resize(self.target_size))

        # Convert to tensor and normalize
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225],  # ImageNet stds
                ),
            ]
        )

        return transforms.Compose(transform_list)

    def _load_dataset(self):
        """Load the specified dataset using torchvision."""
        # Map split names
        is_train = self.split in ["train", "training"]

        if self.dataset_name == "cifar10":
            return datasets.CIFAR10(
                root=self.data_root,
                train=is_train,
                download=True,
                transform=None,  # We'll apply transform manually
            )
        elif self.dataset_name == "cifar100":
            return datasets.CIFAR100(
                root=self.data_root,
                train=is_train,
                download=True,
                transform=None,
            )
        elif self.dataset_name == "imagenet":
            # Map splits for ImageNet
            split_name = "train" if is_train else "val"
            return datasets.ImageNet(
                root=self.data_root,
                split=split_name,
                transform=None,  # We'll apply transform manually
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _prepare_samples(self) -> list:
        """Prepare sample list with subset and max_samples applied."""
        # Get all samples
        all_samples = []
        for i in range(len(self.dataset)):
            image, label = self.dataset[i]

            # Convert to binary classification if specified
            if self.binary_classification:
                binary_label = 1 if label in self.positive_class_indices else 0
                all_samples.append((image, binary_label))
            else:
                all_samples.append((image, label))

        # Apply max_samples first, then subset
        if self.max_samples is not None:
            all_samples = all_samples[: self.max_samples]

        # Apply subset after max_samples
        if self.subset < 1.0:
            subset_size = int(len(all_samples) * self.subset)
            all_samples = all_samples[:subset_size]

        return all_samples

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample from the dataset."""
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.samples)}"
            )

        image, label = self.samples[idx]

        # Apply transforms
        if isinstance(image, Image.Image):
            image_tensor = self.transform(image)
        else:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                image_tensor = self.transform(image)
            else:
                # Fallback
                image_tensor = torch.randn(3, *self.target_size)

        return image_tensor, label

    def get_class_distribution(self) -> dict:
        """Get the distribution of classes in the dataset."""
        if not hasattr(self, "_class_distribution"):
            distribution = {}
            for _, label in self.samples:
                class_name = self.class_names[label]
                distribution[class_name] = distribution.get(class_name, 0) + 1
            self._class_distribution = distribution
        return self._class_distribution
