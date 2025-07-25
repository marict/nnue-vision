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
batch_size = 64  # Good balance for EtinyNet
num_workers = 4  # Parallel data loading
input_size = 32  # CIFAR image size (integer, not tuple)
num_classes = 10  # Will be overridden based on dataset choice
learning_rate = 0.1  # SGD with momentum works best for EtinyNet
subset = 1.0  # Use full dataset

# Training settings
max_epochs = 200  # EtinyNet needs more epochs than NNUE
patience = 20  # Higher patience for SGD convergence
save_top_k = 3  # Save more models for analysis

# System settings
accelerator = "auto"  # Auto-detect GPU/CPU
devices = "auto"
deterministic = True
seed = 42

# Logging and monitoring
log_interval = 50  # Reasonable logging frequency
always_save_checkpoint = True
enable_progress_bar = True
check_val_every_n_epoch = 1
