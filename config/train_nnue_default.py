"""Default configuration for NNUE training.

Designed to complete quickly for development and testing.
Uses minimal data and tiny model parameters for fast iteration.
"""

# Project identification
name = "nnue_default"

# Dataset and model settings
batch_size = 2  # Minimal batch size
num_workers = 0  # No multiprocessing for speed
input_size = (32, 32)  # Smaller images for speed
num_classes = 10  # CIFAR-10 classes for testing
learning_rate = 1e-3
weight_decay = 5e-4  # L2 regularization strength
subset = 0.001  # Use only 0.1% of data (just a few samples)

# NNUE-specific minimal settings
num_ls_buckets = 2  # Minimal buckets
visual_threshold = 0.0

# Training settings
max_epochs = 1  # Single epoch only
patience = 1  # Minimal patience
save_top_k = 1  # Save only best model

# Data augmentation settings
use_augmentation = True  # Enable strong data augmentation to prevent overfitting
augmentation_strength = "heavy"  # Options: "light", "medium", "heavy"


# System settings
accelerator = "cpu"  # Force CPU for consistent timing
devices = 1
deterministic = True
seed = 42

# Logging and monitoring
log_interval = 1  # Log every step for debugging
always_save_checkpoint = True  # Keep checkpointing enabled to avoid conflicts
enable_progress_bar = True  # Keep progress bar enabled to avoid conflicts
check_val_every_n_epoch = 1

# RunPod settings
keep_alive = False  # Don't keep RunPod instances alive after training

# Default settings
log_dir = "logs"
project_name = "nnue_default"
