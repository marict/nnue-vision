"""Local development configuration for EtinyNet training.

Minimal settings for quick local development and testing on CIFAR-10.
"""

# Project identification
name = "etinynet_local"

# EtinyNet-specific settings
etinynet_variant = "0.75"  # Smallest variant for fast iteration
use_asq = False  # Disable ASQ for local runs
asq_bits = 4

# Dataset and model settings
batch_size = 16  # Small batch size for quick tests
num_workers = 2
input_size = 32  # CIFAR image size
num_classes = 10
learning_rate = 1e-3
subset = 0.1  # Use 10% of data for speed

# Training settings
max_epochs = 10  # Short run
patience = 5
save_top_k = 1

# System settings
accelerator = "auto"
devices = "auto"
deterministic = True
seed = 42

# Logging and monitoring
log_interval = 20
always_save_checkpoint = True
enable_progress_bar = True
check_val_every_n_epoch = 1

# Logging backends
log_dir = "logs"
project_name = "etinynet_local"
