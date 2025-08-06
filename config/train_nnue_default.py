"""NNUE Configuration with Exact EtinyNet Training Setup

Matches EtinyNet's optimizer and schedule exactly:
- SGD with momentum (instead of Adam)
- High initial learning rate (0.5)
- Cosine annealing schedule
- Longer training duration

This tests if the optimizer choice is key to EtinyNet's superior performance.
"""

# Project identification
name = "nnue_vision"

# Dataset and model settings
batch_size = 2
num_workers = 8
num_classes = 10  # CIFAR-10 classes

# NNUE model architecture (proven working)
l1_size = 64  # Feature transformer output size
l2_size = 32  # Expanded bottleneck
l3_size = 8  # Second classifier hidden layer
input_size = 32  # Model architecture: 32x32 image size
grid_size = 10  # Grid size for feature extraction
num_features_per_square = 8  # Features per grid square

learning_rate = 0.01
weight_decay = 2e-4  # EtinyNet's weight decay
momentum = 0.9  # SGD momentum (EtinyNet uses this)
optimizer_type = "sgd"  # Use SGD instead of Adam
subset = 0.001
max_epochs = 2
patience = 999999  # Let cosine schedule finish
max_grad_norm = 1.0

# Learning rate schedule - EtinyNet's secret sauce
use_cosine_scheduler = True  # Enable cosine annealing

use_augmentation = False
augmentation_strength = "light"

# System settings
accelerator = "auto"
devices = "auto"
deterministic = False
seed = 42

# Logging and monitocan ring
always_save_checkpoint = False
enable_progress_bar = True
check_val_every_n_epoch = 1
save_top_k = 1

# RunPod settings
keep_alive = False

# Logging backends
log_dir = "logs"
project_name = "nnue_default"

# Debug info
print("🛡️ NNUE Default Training Config Loaded:")
print(f"  • Optimizer: {optimizer_type.upper()} with momentum={momentum}")
print(f"  • Learning rate: {learning_rate} (conservative)")
print(f"  • Gradient clipping: {max_grad_norm}")
print(f"  • Weight decay: {weight_decay}")
print(f"  • Max epochs: {max_epochs}")
print(f"  • Batch size: {batch_size}")
