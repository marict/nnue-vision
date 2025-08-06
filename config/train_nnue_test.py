"""Minimal NNUE configuration for fast testing."""

# Basic training configuration
name = "nnue_test"
project_name = "nnue_testing"

# Training parameters
max_epochs = 1
batch_size = 16
num_workers = 0
subset = 1.0
optimizer_type = "adam"
learning_rate = 1e-3
weight_decay = 5e-4

# Model parameters
l1_size = 64  # Much smaller than default
l2_size = 4  # Much smaller than default
l3_size = 8  # Much smaller than default
num_classes = 10
input_size = 32  # Native CIFAR-10 size
grid_size = 8  # Smaller grid for testing
num_features_per_square = 4  # Fewer features for testing

# Dataset parameters
dataset_name = "cifar10"
max_samples_per_split = 32  # Use only 32 samples for fast testing
use_augmentation = False  # Disable augmentation for speed
augmentation_strength = "light"

# Logging
keep_alive = False


# Loss parameters
start_lambda = 1.0
end_lambda = 1.0
