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
from matplotlib.widgets import Button, Slider

# Import from data module
from data import (AVAILABLE_DATASETS, create_data_loaders, get_dataset_info,
                  get_dataset_stats, print_dataset_stats)
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

        # UI state
        self.current_split = "train"
        self.current_sample_idx = 0

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

    def launch_interactive_explorer(self):
        """Launch an interactive GUI for exploring the dataset."""
        print("\n🎮 Launching Interactive Dataset Explorer...")
        print("Use the controls to navigate through samples and splits!")

        # Create interactive figure
        fig, ((ax_img, ax_hist), (ax_controls, ax_info)) = plt.subplots(
            2, 2, figsize=(15, 10)
        )
        fig.suptitle(f"Interactive {self.dataset_info['name']} Explorer", fontsize=16)

        # Initial display
        self._update_display(fig, ax_img, ax_hist, ax_info)

        # Add controls
        self._add_interactive_controls(fig, ax_controls, ax_img, ax_hist, ax_info)

        plt.tight_layout()
        plt.show()

    def _update_display(self, fig, ax_img, ax_hist, ax_info):
        """Update the interactive display with current sample."""
        # Get current batch and sample
        images, labels = self._get_batch(self.current_split)

        # Ensure sample index is valid
        self.current_sample_idx = self.current_sample_idx % images.shape[0]

        img_tensor = images[self.current_sample_idx]
        label = labels[self.current_sample_idx].item()

        # Get class names
        if hasattr(self.train_loader.dataset, "class_names"):
            class_names = self.train_loader.dataset.class_names
        else:
            class_names = self.dataset_info["classes"]

        class_name = class_names[label]

        # Clear axes
        ax_img.clear()
        ax_hist.clear()
        ax_info.clear()

        # Display image
        img_display = self._denormalize_image(img_tensor)
        ax_img.imshow(img_display)
        ax_img.set_title(
            f"Sample #{self.current_sample_idx} - {class_name}", fontsize=12
        )
        ax_img.axis("off")

        # Display histogram
        img_flat = img_display.flatten()
        ax_hist.hist(img_flat, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax_hist.set_title("Pixel Intensity Histogram", fontsize=12)
        ax_hist.set_xlabel("Pixel Intensity")
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, alpha=0.3)

        # Display info
        ax_info.axis("off")
        info_text = f"""
Dataset Info:
• Dataset: {self.dataset_info['name']}
• Split: {self.current_split.title()}
• Sample: {self.current_sample_idx + 1}/{len(images)}
• Class: {class_name} (label={label})
• Image shape: {img_tensor.shape}
• Image range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]

Current Batch:
• Batch size: {len(images)}
• Total labels: {torch.unique(labels).tolist()}
• Class distribution:
"""

        # Add class distribution
        unique_labels, counts = torch.unique(labels, return_counts=True)
        for lbl, count in zip(unique_labels, counts):
            class_nm = class_names[lbl.item()]
            percentage = count.item() / len(labels) * 100
            info_text += f"  - {class_nm}: {count.item()} ({percentage:.1f}%)\n"

        ax_info.text(
            0.05,
            0.95,
            info_text,
            transform=ax_info.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.draw()

    def _add_interactive_controls(self, fig, ax_controls, ax_img, ax_hist, ax_info):
        """Add interactive controls to the explorer."""
        ax_controls.axis("off")

        # Button positions
        button_width, button_height = 0.15, 0.08
        button_spacing = 0.18

        # Previous/Next buttons
        btn_prev = Button(plt.axes([0.1, 0.3, button_width, button_height]), "Previous")
        btn_next = Button(
            plt.axes([0.1 + button_spacing, 0.3, button_width, button_height]), "Next"
        )

        # Split buttons
        btn_train = Button(plt.axes([0.1, 0.2, button_width, button_height]), "Train")
        btn_val = Button(
            plt.axes([0.1 + button_spacing, 0.2, button_width, button_height]), "Val"
        )
        btn_test = Button(
            plt.axes([0.1 + 2 * button_spacing, 0.2, button_width, button_height]),
            "Test",
        )

        # Random button
        btn_random = Button(plt.axes([0.1, 0.1, button_width, button_height]), "Random")

        # Sample slider
        slider_ax = plt.axes([0.1, 0.05, 0.4, 0.03])
        max_samples = max(
            len(self._get_batch("train")[0]),
            len(self._get_batch("val")[0]),
            len(self._get_batch("test")[0]),
        )
        slider_sample = Slider(
            slider_ax,
            "Sample",
            0,
            max_samples - 1,
            valinit=self.current_sample_idx,
            valfmt="%d",
        )

        # Button callbacks
        def prev_sample(event):
            self.current_sample_idx = (self.current_sample_idx - 1) % len(
                self._get_batch(self.current_split)[0]
            )
            slider_sample.set_val(self.current_sample_idx)
            self._update_display(fig, ax_img, ax_hist, ax_info)

        def next_sample(event):
            self.current_sample_idx = (self.current_sample_idx + 1) % len(
                self._get_batch(self.current_split)[0]
            )
            slider_sample.set_val(self.current_sample_idx)
            self._update_display(fig, ax_img, ax_hist, ax_info)

        def set_split(split_name):
            def callback(event):
                self.current_split = split_name
                self.current_sample_idx = 0
                slider_sample.set_val(0)
                self._update_display(fig, ax_img, ax_hist, ax_info)

            return callback

        def random_sample(event):
            max_idx = len(self._get_batch(self.current_split)[0]) - 1
            self.current_sample_idx = np.random.randint(0, max_idx + 1)
            slider_sample.set_val(self.current_sample_idx)
            self._update_display(fig, ax_img, ax_hist, ax_info)

        def slider_update(val):
            self.current_sample_idx = int(val)
            self._update_display(fig, ax_img, ax_hist, ax_info)

        # Connect callbacks
        btn_prev.on_clicked(prev_sample)
        btn_next.on_clicked(next_sample)
        btn_train.on_clicked(set_split("train"))
        btn_val.on_clicked(set_split("val"))
        btn_test.on_clicked(set_split("test"))
        btn_random.on_clicked(random_sample)
        slider_sample.on_changed(slider_update)

        # Add instructions
        instructions = """
Interactive Controls:
• Previous/Next: Navigate samples
• Train/Val/Test: Switch splits
• Random: Jump to random sample
• Slider: Direct sample selection

Keyboard shortcuts:
• Left/Right: Previous/Next
• 1/2/3: Train/Val/Test splits
• R: Random sample
• Q: Quit
"""
        ax_controls.text(
            0.02,
            0.98,
            instructions,
            transform=ax_controls.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )


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
            print("2. 🎮 Launch interactive explorer")
            print("3. 📊 Show detailed statistics")
            print("4. ❌ Exit")

            while True:
                try:
                    choice = input("\nEnter your choice (1-4): ").strip()

                    if choice == "1":
                        num_samples = input(
                            "Number of samples to show (default: 8): "
                        ).strip()
                        num_samples = int(num_samples) if num_samples else 8
                        inspector.show_sample_images(num_samples=num_samples)

                    elif choice == "2":
                        inspector.launch_interactive_explorer()

                    elif choice == "3":
                        inspector.print_overview()

                    elif choice == "4":
                        print("👋 Goodbye!")
                        break

                    else:
                        print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

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
