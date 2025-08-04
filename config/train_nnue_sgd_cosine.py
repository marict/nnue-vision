"""NNUE Configuration with Exact EtinyNet Training Setup

Matches EtinyNet's optimizer and schedule exactly:
- SGD with momentum (instead of Adam)
- High initial learning rate (0.5)
- Cosine annealing schedule
- Longer training duration

This tests if the optimizer choice is key to EtinyNet's superior performance.
"""

# Project identification
name = "nnue_vision-sgd-cosine"

# Dataset and model settings
batch_size = 512
num_workers = 8
num_classes = 10  # CIFAR-10 classes

# NNUE model architecture (proven working)
l1_size = 1024  # Feature transformer output size
l2_size = 128  # Expanded bottleneck
l3_size = 32  # Second classifier hidden layer
input_size = 32  # Model architecture: 32x32 image size

# Training settings - EXACT EtinyNet setup
learning_rate = 0.5  # EtinyNet's aggressive initial LR
weight_decay = 2e-4  # EtinyNet's weight decay
momentum = 0.9  # SGD momentum (EtinyNet uses this)
optimizer_type = "sgd"  # Use SGD instead of Adam
subset = 1.0  # Use full dataset
max_epochs = 300  # EtinyNet duration
patience = 999999  # Let cosine schedule finish

# Learning rate schedule - EtinyNet's secret sauce
use_cosine_scheduler = True  # Enable cosine annealing

use_augmentation = True
augmentation_strength = "heavy"

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
save_top_k = 1

# RunPod settings
keep_alive = False

# Logging backends
log_dir = "logs"
project_name = "nnue_training"

# Debug info
print("⚡ NNUE + EtinyNet Exact Setup Config Loaded:")
print(f"  • Optimizer: SGD with momentum={momentum} (like EtinyNet)")
print(f"  • Learning rate: {learning_rate} → 0 (cosine annealing)")
print(f"  • Weight decay: {weight_decay} (EtinyNet value)")
print(f"  • Max epochs: {max_epochs} (EtinyNet duration)")
print(f"  • Batch size: {batch_size} (EtinyNet efficiency)")
print(f"  • L2 bottleneck: {l2_size} (NNUE architecture advantage)")
print("🚀 Testing: Can NNUE match EtinyNet with exact same training?")
