"""Generic computer vision dataset support for multiple benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

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
        target_size: Optional[Tuple[int, int]] = None,  # Auto-determine from dataset
        max_samples: Optional[int] = None,
        subset: float = 1.0,
        data_root: str = None,  # Auto-detect persistent storage
        binary_classification: Optional[dict] = None,
        use_augmentation: bool = None,
        augmentation_strength: str = "medium",
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
            use_augmentation: Enable data augmentation (None=auto-detect from split)
            augmentation_strength: Augmentation intensity ('light', 'medium', 'heavy')
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_samples = max_samples
        self.subset = subset

        # Auto-detect persistent storage for datasets
        if data_root is None:
            # Check for RunPod volume first, then fall back to local
            if Path("/runpod-volume").exists():
                self.data_root = Path("/runpod-volume/datasets")
                print(f"📁 Using persistent dataset storage: {self.data_root}")
            else:
                self.data_root = Path("./data/raw")
                print(f"📁 Using local dataset storage: {self.data_root}")
        else:
            self.data_root = Path(data_root)

        self.binary_classification = binary_classification
        self.use_augmentation = use_augmentation
        self.augmentation_strength = augmentation_strength

        # Ensure dataset directory exists
        self.data_root.mkdir(parents=True, exist_ok=True)

        # Get dataset info
        self.dataset_info = get_dataset_info(dataset_name)

        # Auto-determine target_size from dataset native size if not provided
        if target_size is None:
            native_sizes = {
                "cifar10": (32, 32),
                "cifar100": (32, 32),
                "imagenet": (224, 224),  # Standard ImageNet size
                "mnist": (28, 28),
                "fashionmnist": (28, 28),
            }
            self.target_size = native_sizes.get(
                dataset_name.lower(), (96, 96)
            )  # Fallback to 96x96
            print(
                f"📏 Auto-determined target_size for {dataset_name}: {self.target_size}"
            )
        else:
            self.target_size = target_size

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
        self.transform = self._build_transform(self.use_augmentation)

        # Load dataset
        print(f"🔄 Loading {self.dataset_info['name']} dataset ({split} split)...")
        self.dataset = self._load_dataset()

        # Apply subset and max_samples
        self.samples = self._prepare_samples()

        print(f"✅ Loaded {len(self.samples)} samples from {self.dataset_info['name']}")

    def _build_transform(self, use_augmentation: bool = None) -> A.Compose:
        """Build image preprocessing pipeline with comprehensive augmentation."""
        if use_augmentation is None:
            use_augmentation = self.split in ["train", "training"]

        # Base transforms (no resize here - will be done at the end)
        base_transforms = []

        if use_augmentation:
            # Different augmentation strengths
            if self.augmentation_strength == "light":
                augmentation_transforms = [
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.2
                    ),
                    A.CoarseDropout(
                        num_holes_range=(1, 1),
                        hole_height_range=(0.05, 0.05),
                        hole_width_range=(0.05, 0.05),
                        p=0.2,
                    ),
                ]
            elif self.augmentation_strength == "heavy":
                # Start with all medium transformations
                medium_transforms = [
                    # Geometric augmentations
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Rotate(limit=15, p=0.3),
                    A.Affine(
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        scale=(0.9, 1.1),
                        rotate=(-15, 15),
                        p=0.3,
                    ),
                    # Brightness/contrast augmentations
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=0.3,
                    ),
                    # Blur and noise
                    A.OneOf(
                        [
                            A.Blur(blur_limit=3, p=1.0),
                            A.GaussianBlur(blur_limit=3, p=1.0),
                            A.MotionBlur(blur_limit=3, p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                    # Cutout augmentations
                    A.CoarseDropout(
                        num_holes_range=(1, 1),
                        hole_height_range=(0.05, 0.15),  # As proportions of image
                        hole_width_range=(0.05, 0.15),
                        p=0.3,
                    ),
                    # Advanced augmentations
                    A.RandomShadow(p=0.1),
                    A.RandomFog(p=0.1),
                    A.GridDistortion(p=0.1),
                    A.ElasticTransform(p=0.1),
                    # Color augmentations
                    A.CLAHE(clip_limit=2.0, p=0.1),
                    A.ColorJitter(p=0.2),
                    A.Posterize(p=0.1),
                    A.Equalize(p=0.1),
                ]

                # Add aggressive heavy augmentations on top
                heavy_extra_transforms = [
                    # Additional aggressive geometric augmentations
                    A.HorizontalFlip(
                        p=0.1
                    ),  # Extra flip probability (total becomes ~0.55)
                    A.RandomRotate90(p=0.1),  # Extra rotation probability
                    A.Rotate(limit=25, p=0.2),  # Stronger rotation
                    A.Affine(
                        translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                        scale=(0.85, 1.15),
                        rotate=(-25, 25),
                        p=0.2,
                    ),
                    # Additional strong brightness/contrast
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=0.2
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=15,
                        sat_shift_limit=20,
                        val_shift_limit=15,
                        p=0.2,
                    ),
                    # Additional stronger blur and noise
                    A.OneOf(
                        [
                            A.Blur(blur_limit=5, p=1.0),
                            A.GaussianBlur(blur_limit=5, p=1.0),
                            A.MotionBlur(blur_limit=5, p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.GaussNoise(std_range=(0.05, 0.1), p=0.2),
                    # Additional aggressive cutout
                    A.CoarseDropout(
                        num_holes_range=(1, 2),
                        hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25),
                        p=0.2,
                    ),
                    # Additional advanced augmentations
                    A.RandomShadow(p=0.1),
                    A.RandomFog(p=0.1),
                    A.GridDistortion(p=0.1),
                    A.ElasticTransform(p=0.1),
                    A.CLAHE(clip_limit=3.0, p=0.1),
                    A.ColorJitter(p=0.1),
                    A.Posterize(p=0.1),
                    A.Equalize(p=0.1),
                ]

                augmentation_transforms = medium_transforms + heavy_extra_transforms
            else:  # medium (default)
                augmentation_transforms = [
                    # Geometric augmentations
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Rotate(limit=15, p=0.3),
                    A.Affine(
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        scale=(0.9, 1.1),
                        rotate=(-15, 15),
                        p=0.3,
                    ),
                    # Brightness/contrast augmentations
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.3
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=0.3,
                    ),
                    # Blur and noise
                    A.OneOf(
                        [
                            A.Blur(blur_limit=3, p=1.0),
                            A.GaussianBlur(blur_limit=3, p=1.0),
                            A.MotionBlur(blur_limit=3, p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                    # Cutout augmentations
                    A.CoarseDropout(
                        num_holes_range=(1, 1),
                        hole_height_range=(0.05, 0.15),  # As proportions of image
                        hole_width_range=(0.05, 0.15),
                        p=0.3,
                    ),
                    # Advanced augmentations
                    A.RandomShadow(p=0.1),
                    A.RandomFog(p=0.1),
                    A.GridDistortion(p=0.1),
                    A.ElasticTransform(p=0.1),
                    # Color augmentations
                    A.CLAHE(clip_limit=2.0, p=0.1),
                    A.ColorJitter(p=0.2),
                    A.Posterize(p=0.1),
                    A.Equalize(p=0.1),
                ]

            transforms_list = base_transforms + augmentation_transforms
        else:
            # No augmentation for validation/test
            transforms_list = base_transforms

        # Add resize at the end to ensure final output size is always correct
        # (this handles dimension swaps from rotations, etc.)
        transforms_list.append(
            A.Resize(height=self.target_size[0], width=self.target_size[1], p=1.0)
        )

        # Add normalization and tensor conversion
        transforms_list.extend(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225],  # ImageNet stds
                ),
                ToTensorV2(),
            ]
        )

        return A.Compose(transforms_list)

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

        # Convert PIL Image to numpy array for Albumentations
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            # Fallback - create random image
            image_array = np.random.randint(
                0, 255, (*self.target_size, 3), dtype=np.uint8
            )

        # Apply transforms
        try:
            transformed = self.transform(image=image_array)
            image_tensor = transformed["image"]
        except Exception as e:
            print(f"Error: Transform failed for image {idx}: {e}")
            raise

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
