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
batch_size = 1024  # GPU-friendly
num_workers = 8  # Reasonable parallelism
input_size = (32, 32)  # Native CIFAR-10 resolution (no upsampling needed)
num_classes = 10  # CIFAR-10 classes
learning_rate = 0.5  # Initial LR for cosine annealing (was 3e-4 * (1024 / 64))
weight_decay = 2e-4  # L2 regularization strength
subset = 1.0  # Use full dataset

# Data augmentation settings
use_augmentation = True
augmentation_strength = "heavy"  # Options: "light", "medium", "heavy"

# Training settings
max_epochs = 800
patience = 99999
save_top_k = 1  # Save only best model

# System settings
accelerator = "auto"
devices = "auto"
deterministic = False
seed = 42

# Logging and monitoring
log_interval = 1
always_save_checkpoint = False
enable_progress_bar = True
check_val_every_n_epoch = 1

# RunPod settings
keep_alive = False  # Don't keep RunPod instances alive after training

# Logging backends
log_dir = "logs"
project_name = "nnue-vision-train"
