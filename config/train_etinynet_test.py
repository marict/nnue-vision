"""Minimal EtinyNet configuration for fast testing."""

# Basic training configuration
name = "etinynet_test"
project_name = "etinynet_testing"

# Training parameters
max_epochs = 1
batch_size = 16
learning_rate = 0.1
weight_decay = 1e-4

# Model parameters
etinynet_variant = "0.75"
use_asq = False
asq_bits = 4

# Dataset parameters
dataset_name = "cifar10"
max_samples_per_split = 32  # Use only 32 samples for fast testing
use_augmentation = False  # Disable augmentation for speed

# Logging
log_interval = 10
keep_alive = False
