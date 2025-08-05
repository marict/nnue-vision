"""EtinyNet training configuration.

Optimized settings for training EtinyNet models on CIFAR-10/CIFAR-100.
Uses SGD with momentum and cosine annealing as per the EtinyNet paper.
"""

# Project identification
name = "etinynet-cifar"

# EtinyNet-specific settings
etinynet_variant = "0.98M"
use_asq = False  # Disable ASQ for initial training
asq_bits = 4  # ASQ bits if enabled

# Dataset and model settings
batch_size = 1024
num_workers = 8  # Parallel data loading
input_size = (32, 32)  # CIFAR image size
num_classes = 10  # Will be overridden based on dataset choice
learning_rate = 0.5
weight_decay = 2e-4  # L2 regularization strength
subset = 1.0  # Use full dataset

# Data augmentation settings
use_augmentation = True
augmentation_strength = "heavy"  # Options: "light", "medium", "heavy"

# Training settings
max_epochs = 800
patience = 99999  # Never stop
save_top_k = 1  # Save only best model

# System settings
accelerator = "auto"  # Auto-detect GPU/CPU
devices = "auto"
deterministic = True
seed = 42

# Logging and monitoring
log_interval = 1
always_save_checkpoint = False
enable_progress_bar = True
check_val_every_n_epoch = 1
keep_alive = True  # Keep RunPod instance alive after training

# Simplified: only save best models to wandb (no volume storage)

# Logging backends
log_dir = "logs"
project_name = "nnue_training"
