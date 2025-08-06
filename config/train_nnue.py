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
batch_size = 512
num_workers = 8
num_classes = 10  # CIFAR-10 classes

# NNUE model architecture (proven working)
l1_size = 1024  # Feature transformer output size
l2_size = 128  # Expanded bottleneck
l3_size = 32  # Second classifier hidden layer
input_size = 32  # Model architecture: 32x32 image size
grid_size = 10  # Grid size for feature extraction
num_features_per_square = 8  # Features per grid square

# Training settings - NNUE-optimized (post-quantization constraint removal)
learning_rate = 0.01
weight_decay = 2e-4  # EtinyNet's weight decay
momentum = 0.9  # SGD momentum (EtinyNet uses this)
optimizer_type = "sgd"  # Use SGD instead of Adam
subset = 1.0  # Use full dataset
max_epochs = 300  # EtinyNet duration
patience = 999999  # Let cosine schedule finish
max_grad_norm = 1.0
# max_grad_norm = 0.0  # Try without gradient clipping

# Learning rate schedule - EtinyNet's secret sauce
use_cosine_scheduler = True  # Enable cosine annealing

# Learning rate schedule parameters
decay_lr = True
use_cyclical_lr = False
cyclical_lr_period = 1000
cyclical_lr_amplitude = 0.1

use_augmentation = True
augmentation_strength = "light"  # Conservative augmentation for stability

# System settings
accelerator = "auto"
devices = "auto"
deterministic = False
seed = 42

# Logging and monitoring
always_save_checkpoint = False
enable_progress_bar = True
check_val_every_n_epoch = 1
save_top_k = 1


# Simplified: only save best models to wandb (no volume storage)

# RunPod settings
keep_alive = False

# Logging backends
log_dir = "logs"
project_name = "nnue_training"

# Debug info
print("🛡️ NNUE Stable Training Config Loaded:")
print(f"  • Optimizer: SGD with momentum={momentum} (like EtinyNet)")
print(f"  • Learning rate: {learning_rate} → 0 (cosine annealing, conservative)")
print(f"  • Gradient clipping: {max_grad_norm} (prevent NaN)")
print(f"  • Weight decay: {weight_decay} (EtinyNet value)")
print(f"  • Max epochs: {max_epochs} (EtinyNet duration)")
print(f"  • Batch size: {batch_size} (efficiency)")
print(f"  • Augmentation: {augmentation_strength} (stable)")
print(f"  • L2 bottleneck: {l2_size} (8.5x expansion)")
