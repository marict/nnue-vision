"""NNUE Configuration with EtinyNet-Style Cosine Annealing

Adopts the successful EtinyNet learning schedule to break through NNUE's 45% plateau.
Uses cosine annealing for dynamic learning rate that starts aggressive and fine-tunes.
"""

# Project identification
name = "nnue_vision-cosine"

# Dataset and model settings
batch_size = 128  # Increase to match EtinyNet efficiency
num_workers = 8
num_classes = 10  # CIFAR-10 classes

# NNUE model architecture (proven working architecture)
l1_size = 1024  # Feature transformer output size
l2_size = 128  # Expanded bottleneck - key to success!
l3_size = 32  # Second classifier hidden layer
input_size = 32  # Model architecture: 32x32 image size

# Training settings - EtinyNet-inspired schedule
learning_rate = 0.1  # Higher initial LR for cosine annealing (vs 0.001 constant)
weight_decay = 5e-4  # Keep NNUE weight decay
subset = 1.0  # Use full dataset
max_epochs = 300  # Longer training like EtinyNet (vs 100)
patience = 999999  # Let cosine schedule finish

# Learning rate schedule - THE KEY IMPROVEMENT
use_cosine_scheduler = True  # Enable cosine annealing like EtinyNet!

# Data augmentation - light for convergence
use_augmentation = True
augmentation_strength = "light"  # Proven to work

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
print("📈 NNUE Cosine Annealing Config Loaded:")
print(f"  • Learning rate: {learning_rate} → 0 (cosine annealing)")
print(f"  • Max epochs: {max_epochs} (longer training)")
print(f"  • Batch size: {batch_size} (efficiency)")
print(f"  • L2 bottleneck: {l2_size} (8.5x expansion)")
print(f"  • Scheduler: Cosine annealing (like EtinyNet)")
print("🎯 Expected: Break through 45% plateau with dynamic learning!")
