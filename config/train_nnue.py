"""NNUE GPU training configuration.

Optimised defaults for NNUE-Vision when running on a single GPU (local or cloud).
Values are larger than the tiny local-dev config but still modest so tests can
import this file quickly.
"""

# Project identification
name = "nnue_vision-gpu"

# NNUE-specific settings
num_ls_buckets = 8  # Standard bucket count
visual_threshold = 0.0

# Dataset and model settings
batch_size = 512  # GPU-friendly
num_workers = 8  # Reasonable parallelism
input_size = (32, 32)  # Native CIFAR-10 resolution (no upsampling needed)
num_classes = 10  # CIFAR-10 classes
learning_rate = 3e-4 * (512 / 64)
subset = 1.0  # Use full dataset

# Data augmentation settings
use_augmentation = True  # Enable strong data augmentation to prevent overfitting
augmentation_strength = "heavy"  # Options: "light", "medium", "heavy"

# Training settings
max_epochs = 500
patience = 10  # Early-stopping patience
save_top_k = 3  # Keep best checkpoints

# System settings
accelerator = "auto"
devices = "auto"
deterministic = False
seed = 42

# Logging and monitoring
log_interval = 50
always_save_checkpoint = True
enable_progress_bar = True
check_val_every_n_epoch = 1

# Logging backends
log_dir = "logs"
project_name = "nnue-vision-train"
