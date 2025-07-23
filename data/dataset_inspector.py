from __future__ import annotations

"""Dataset Inspector module.

This module exposes the ``DatasetInspector`` class, providing interactive
visualization and analysis utilities for the Visual Wake Words dataset.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider
from PIL import Image

from . import create_data_loaders, get_dataset_stats, print_dataset_stats
from .datasets import VWW_CLASS_NAMES
from .loaders import calculate_dataset_statistics, print_loader_statistics

__all__ = ["DatasetInspector"]


class DatasetInspector:
    """Interactive dataset inspector with visualization capabilities for Visual Wake Words.

    Parameters
    ----------
    batch_size:
        Batch size used when iterating through the dataset for statistics and
        visualization.
    max_samples_per_split:
        Maximum number of samples to load per split for faster loading.
        Set to None to load all available samples.
    """

    def __init__(
        self, *, batch_size: int = 16, max_samples_per_split: Optional[int] = 100
    ):
        self.batch_size = batch_size
        self.max_samples_per_split = max_samples_per_split

        # Lazily initialise loaders to avoid long blocking operations in __init__
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

        # UI state
        self.current_split: str = "train"
        self.current_sample_idx: int = 0

    # ---------------------------------------------------------------------
    # Properties / lazy loaders
    # ---------------------------------------------------------------------
    @property
    def train_loader(self):
        if self._train_loader is None:
            self._initialise_loaders()
        return self._train_loader

    @property
    def val_loader(self):
        if self._val_loader is None:
            self._initialise_loaders()
        return self._val_loader

    @property
    def test_loader(self):
        if self._test_loader is None:
            self._initialise_loaders()
        return self._test_loader

    def _initialise_loaders(self):
        print("🔄 Creating Visual Wake Words data loaders (lazy init)…")
        (self._train_loader, self._val_loader, self._test_loader) = create_data_loaders(
            batch_size=self.batch_size,
            num_workers=0,
            max_samples_per_split=self.max_samples_per_split,
        )
        print("✅ Data loaders ready!")

    # ------------------------------------------------------------------
    # Public API methods (print_overview, show_sample_images, etc.)
    # ------------------------------------------------------------------
    def print_overview(self):
        """Print dataset overview and basic statistics."""
        print("\n" + "=" * 80)
        print("🔍 VISUAL WAKE WORDS DATASET INSPECTOR")
        print("=" * 80)

        # Dataset‐level stats
        print_dataset_stats()

        # Loader‐level stats
        stats = calculate_dataset_statistics(self.train_loader)
        print_loader_statistics(stats)

    # ------------------------------------------------------------------
    # Sample handling utilities
    # ------------------------------------------------------------------
    def _get_batch(self, split: str):
        if split == "train":
            return next(iter(self.train_loader))
        if split == "val":
            return next(iter(self.val_loader))
        if split == "test":
            return next(iter(self.test_loader))
        raise ValueError(f"Unknown split: {split}")

    @staticmethod
    def _denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
        """Convert normalised tensor back to uint8 image for display."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denorm = torch.clamp(img_tensor * std + mean, 0, 1)
        return denorm.permute(1, 2, 0).cpu().numpy()

    # ------------------------------------------------------------------
    # Basic visualisation helpers
    # ------------------------------------------------------------------
    def show_sample_images(self, *, num_samples: int = 8, split: str = "train") -> None:
        """Display a grid of sample images from the Visual Wake Words dataset."""
        images, labels = self._get_batch(split)
        num_samples = min(num_samples, images.shape[0])

        grid = int(np.ceil(np.sqrt(num_samples)))
        fig, axes = plt.subplots(grid, grid, figsize=(12, 12))
        plt.suptitle(f"Visual Wake Words - {split.title()} split", fontsize=16)
        axes = axes.flatten() if grid > 1 else [axes]

        for i in range(grid * grid):
            ax = axes[i]
            if i < num_samples:
                img = self._denormalize_image(images[i])
                cls = VWW_CLASS_NAMES[labels[i].item()]
                ax.imshow(img)
                ax.set_title(f"#{i}: {cls}", fontsize=9)
                ax.axis("off")
            else:
                ax.axis("off")

        plt.tight_layout()
        plt.show()
