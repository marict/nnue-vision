#!/usr/bin/env python3
"""
Generic Dataset Inspector

A comprehensive tool for exploring and analyzing computer vision datasets.
Supports CIFAR-10, CIFAR-100, and other torchvision datasets.

Usage:
    python inspect_dataset.py                            # Interactive mode with CIFAR-10
    python inspect_dataset.py --dataset cifar100         # Use CIFAR-100
    python inspect_dataset.py --stats                    # Print statistics only
    python inspect_dataset.py --samples 10               # Show sample images
    python inspect_dataset.py --binary-vehicles          # Binary classification example
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Import from data module
from data import (
    AVAILABLE_DATASETS,
    create_data_loaders,
    get_dataset_info,
    get_dataset_stats,
    print_dataset_stats,
)
from data.loaders import calculate_dataset_statistics, print_loader_statistics


class DatasetInspector:
    """Interactive dataset inspector with visualization capabilities."""

    def __init__(
        self,
        dataset_name: str = "cifar10",
        batch_size: int = 16,
        binary_classification: Optional[dict] = None,
    ):
        """
        Initialize the dataset inspector.

        Args:
            dataset_name: Name of dataset to inspect ('cifar10', 'cifar100')
            batch_size: Batch size for data loading
            binary_classification: Optional binary classification config
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.binary_classification = binary_classification

        # Get dataset info
        self.dataset_info = get_dataset_info(dataset_name)

        print(f"🔍 Initializing Dataset Inspector for {self.dataset_info['name']}...")
        if binary_classification:
            positive_classes = binary_classification.get("positive_classes", [])
            print(f"  Binary classification: {positive_classes} → positive")

        # Create data loaders
        try:
            self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
                dataset_name=dataset_name,
                batch_size=batch_size,
                num_workers=0,  # Use 0 for easier debugging
                subset=0.05,  # Use small subset for faster loading during inspection
                binary_classification=binary_classification,
            )
            print("✅ Data loaders created successfully!")
        except Exception as e:
            print(f"❌ Error creating data loaders: {e}")
            print(
                f"💡 Please check that the {self.dataset_info['name']} dataset is available"
            )
            sys.exit(1)

        # Cache for loaded batches
        self._train_batch = None
        self._val_batch = None
        self._test_batch = None

    def print_overview(self):
        """Print a comprehensive overview of the dataset."""
        print("\n" + "=" * 80)
        print(f"🔍 {self.dataset_info['name'].upper()} DATASET INSPECTOR")
        print("=" * 80)

        # Dataset statistics
        print_dataset_stats(self.dataset_name)

        # Data loader statistics
        print("📊 Computing data loader statistics...")
        try:
            train_stats = calculate_dataset_statistics(self.train_loader)
            print_loader_statistics(train_stats)
        except Exception as e:
            print(f"⚠️  Warning: Could not compute detailed statistics: {repr(e)}")
            print(f"    Error type: {type(e).__name__}")
            print("📊 Basic statistics will be shown instead")

        # Quick sample check
        print("🖼️  Sample data check:")
        try:
            train_batch = self._get_batch("train")
            images, labels = train_batch
        except Exception as e:
            print(f"❌ Error getting batch: {repr(e)} (type: {type(e).__name__})")
            return

        print(f"  • Batch shape: {images.shape}")
        print(f"  • Label shape: {labels.shape}")
        print(f"  • Image dtype: {images.dtype}")
        print(f"  • Label dtype: {labels.dtype}")
        print(
            f"  • Image range: [{images.min().item():.3f}, {images.max().item():.3f}]"
        )
        print(f"  • Unique labels: {torch.unique(labels).tolist()}")

        # Class distribution in current batch
        unique_labels, counts = torch.unique(labels, return_counts=True)
        print(f"  • Batch class distribution:")

        # Get class names from the dataset
        if hasattr(self.train_loader.dataset, "class_names"):
            class_names = self.train_loader.dataset.class_names
        else:
            class_names = self.dataset_info["classes"]

        for label, count in zip(unique_labels, counts):
            class_name = class_names[label.item()]
            print(f"    - {class_name}: {count.item()} samples")

    def _get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a cached batch for the specified split."""
        if split == "train":
            if self._train_batch is None:
                self._train_batch = next(iter(self.train_loader))
            return self._train_batch
        elif split == "val":
            if self._val_batch is None:
                self._val_batch = next(iter(self.val_loader))
            return self._val_batch
        elif split == "test":
            if self._test_batch is None:
                self._test_batch = next(iter(self.test_loader))
            return self._test_batch
        else:
            raise ValueError(f"Unknown split: {split}")

    def show_sample_images(self, num_samples: int = 8, split: str = "train"):
        """
        Display a grid of sample images from the dataset.

        Args:
            num_samples: Number of images to display
            split: Dataset split to sample from
        """
        print(f"\n🖼️  Displaying {num_samples} sample images from {split} split...")

        images, labels = self._get_batch(split)

        # Limit to available samples
        num_samples = min(num_samples, images.shape[0])

        # Create subplot grid
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        fig.suptitle(
            f"{self.dataset_info['name']} - {split.title()} Split", fontsize=16
        )

        # Flatten axes for easier indexing
        if grid_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Get class names
        if hasattr(self.train_loader.dataset, "class_names"):
            class_names = self.train_loader.dataset.class_names
        else:
            class_names = self.dataset_info["classes"]

        for i in range(grid_size * grid_size):
            ax = axes[i]

            if i < num_samples:
                # Get image and label
                img_tensor = images[i]
                label = labels[i].item()
                class_name = class_names[label]

                # Denormalize image for display
                img_display = self._denormalize_image(img_tensor)

                # Display image
                ax.imshow(img_display)
                ax.set_title(f"#{i}: {class_name}", fontsize=10)
                ax.axis("off")
            else:
                # Hide unused subplots
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    def _denormalize_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor for display.

        Args:
            img_tensor: Normalized image tensor [C, H, W]

        Returns:
            Denormalized image array [H, W, C] in range [0, 1]
        """
        # ImageNet normalization parameters
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # Denormalize
        denorm = img_tensor * std + mean

        # Clamp to valid range and convert to numpy
        denorm = torch.clamp(denorm, 0, 1)
        img_array = denorm.permute(1, 2, 0).numpy()

        return img_array


def main():
    """Main entry point for the dataset inspector."""
    parser = argparse.ArgumentParser(
        description="Generic Computer Vision Dataset Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets: {', '.join(AVAILABLE_DATASETS.keys())}

Examples:
  python inspect_dataset.py                            # Interactive mode with CIFAR-10
  python inspect_dataset.py --dataset cifar100         # Use CIFAR-100
  python inspect_dataset.py --stats                    # Show statistics only  
  python inspect_dataset.py --samples 16               # Show 16 sample images
  python inspect_dataset.py --binary-vehicles          # Binary classification demo
        """,
    )

    parser.add_argument(
        "--dataset",
        choices=list(AVAILABLE_DATASETS.keys()),
        default="cifar10",
        help="Dataset to inspect (default: cifar10)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics only (no interactive mode)",
    )
    parser.add_argument(
        "--samples", type=int, metavar="N", help="Show N sample images and exit"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Batch size for data loading (default: 16)",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split to use for samples (default: train)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip matplotlib display (useful for testing or headless environments)",
    )
    parser.add_argument(
        "--binary-vehicles",
        action="store_true",
        help="Demonstrate binary classification (vehicles vs non-vehicles for CIFAR-10)",
    )

    args = parser.parse_args()

    # Setup binary classification if requested
    binary_classification = None
    if args.binary_vehicles and args.dataset == "cifar10":
        binary_classification = {
            "positive_classes": ["airplane", "automobile", "ship", "truck"]
        }
        print("🚗 Using binary classification: vehicles vs non-vehicles")

    try:
        # Create inspector
        inspector = DatasetInspector(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            binary_classification=binary_classification,
        )

        # Handle different modes
        if args.stats:
            # Stats only mode
            inspector.print_overview()
        elif args.samples:
            # Sample display mode
            inspector.print_overview()
            if not args.no_display:
                inspector.show_sample_images(num_samples=args.samples, split=args.split)
            else:
                print(
                    f"📊 Would display {args.samples} sample images from {args.split} split (skipped due to --no-display)"
                )
        else:
            # Interactive mode
            inspector.print_overview()

            if args.no_display:
                print(
                    "📊 Interactive mode would launch GUI (skipped due to --no-display)"
                )
                return

            print("\n" + "=" * 60)
            print("🎮 INTERACTIVE MODE")
            print("=" * 60)
            print("Choose an option:")
            print("1. 🖼️  Show sample images")
            print("2. 📊 Show detailed statistics")
            print("3. ❌ Exit")

            while True:
                try:
                    choice = input("\nEnter your choice (1-3): ").strip()

                    if choice == "1":
                        num_samples = input(
                            "Number of samples to show (default: 8): "
                        ).strip()
                        num_samples = int(num_samples) if num_samples else 8
                        inspector.show_sample_images(num_samples=num_samples)

                    elif choice == "2":
                        inspector.print_overview()

                    elif choice == "3":
                        print("👋 Goodbye!")
                        break

                    else:
                        print("❌ Invalid choice. Please enter 1, 2, or 3.")

                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except ValueError:
                    print("❌ Please enter a valid number.")
                except Exception as e:
                    print(f"❌ Error: {e}")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
