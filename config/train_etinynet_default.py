"""Default configuration for EtinyNet training.

Designed to complete quickly for development and testing.
Uses minimal data and tiny model parameters for fast iteration.
"""

# Project identification
name = "etinynet_default"

# EtinyNet-specific settings
etinynet_variant = "micro"  # Ultra-tiny variant for local testing
use_asq = False  # Disable ASQ for speed
asq_bits = 4

# Dataset and model settings
batch_size = 2  # Minimal batch size
num_workers = 0  # No multiprocessing for speed
input_size = (32, 32)  # CIFAR image size
num_classes = 10
learning_rate = 1e-3
weight_decay = 1e-4  # L2 regularization strength
subset = 0.001  # Use only 0.1% of data (just a few samples)

# Training settings
max_epochs = 1  # Single epoch only
patience = 1  # Minimal patience
save_top_k = 1  # Save only best model

# System settings
accelerator = "cpu"  # Force CPU for consistent timing
devices = 1
deterministic = True
seed = 42

# Logging and monitoring
always_save_checkpoint = True  # Keep checkpointing enabled to avoid conflicts
enable_progress_bar = True  # Keep progress bar enabled to avoid conflicts
check_val_every_n_epoch = 1

# RunPod settings
keep_alive = False  # Terminate RunPod instance after training completes


# Default settings
log_dir = "logs"
project_name = "nnue_default"
