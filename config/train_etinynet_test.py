"""Minimal EtinyNet configuration for fast testing."""

# Basic training configuration
name = "etinynet_test"
project_name = "etinynet_testing"

# Training parameters
max_epochs = 1
batch_size = 16
num_workers = 0
subset = 0.001
use_augmentation = False
augmentation_strength = "light"
optimizer_type = "sgd"
momentum = 0.9
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
num_classes = 10
input_size = 32

# Learning rate schedule parameters
decay_lr = True
use_cyclical_lr = False
cyclical_lr_period = 1000
cyclical_lr_amplitude = 0.1

# Logging
keep_alive = False
