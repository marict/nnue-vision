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
num_workers = 1
num_classes = 10  # CIFAR-10 classes

# NNUE model architecture (proven working)
input_size = 32  # Model architecture: 32x32 image size

learning_rate = 0.01
weight_decay = 2e-4  # EtinyNet's weight decay
momentum = 0.9  # SGD momentum (EtinyNet uses this)
optimizer_type = "sgd"  # Use SGD instead of Adam
subset = 0.1
max_epochs = 2  # EtinyNet duration
patience = 999999  # Let cosine schedule finish
max_grad_norm = 1.0

# Learning rate schedule - EtinyNet's secret sauce
use_cosine_scheduler = True  # Enable cosine annealing

use_augmentation = True
augmentation_strength = "heavy"

# System settings
accelerator = "auto"
devices = "auto"
deterministic = False
seed = 42

# Logging and monitocan ring
log_interval = 1
always_save_checkpoint = False
enable_progress_bar = True
check_val_every_n_epoch = 1
save_top_k = 1

# Wandb checkpoint settings
save_checkpoint_every_n_epochs = 10  # Save checkpoint to wandb every N epochs
always_save_best_to_wandb = True  # Always upload best checkpoints to wandb

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
print(f"  • Wandb checkpoint frequency: every {save_checkpoint_every_n_epochs} epochs")
print(f"  • Save best models to wandb: {always_save_best_to_wandb}")
print("🎯 Goal: Stable NNUE training with automatic wandb checkpoint saving!")
