"""Minimal configuration for quick local testing of NNUE-Vision training.

This is scaled down for fast execution on a laptop/CI runner.
All settings are optimized for quick iteration and debugging.
"""

# Project settings
# ----------------
name = "nnue_vision-minimal"

# Training intervals and logging
# ------------------------------
eval_interval = 1
log_interval = 10  # Log more frequently for debugging
eval_only = False
always_save_checkpoint = True
clear_previous_checkpoints = True

# Model initialization
init_from = "scratch"

# Dataset configuration
# ---------------------
dataset = "visual_wake_words"
batch_size = 16  # Smaller batch size for CPU/limited memory
num_workers = 2  # Reduced for stability
image_size = 96

# Model architecture
# ------------------
input_size = (96, 96)
num_classes = 2
learning_rate = 1e-3

# Training hyperparameters
# ------------------------
max_epochs = 5  # Very few epochs for quick testing
patience = 3  # Early stopping patience
save_top_k = 1  # Save only the best model

# System settings
# ---------------
accelerator = "auto"
devices = "auto"
deterministic = True
precision = "32"

# Logging and checkpointing
# -------------------------
log_dir = "logs"
project_name = "visual_wake_words_minimal"
use_wandb = False  # Disable wandb for minimal testing
use_tensorboard = False
wandb_api_key = None

# Data augmentation
# -----------------
use_augmentation = False  # Disable for faster training

# Random seeds
# ------------
seed = 42

# Advanced settings
# -----------------
compile_model = False
check_val_every_n_epoch = 1
enable_progress_bar = True
enable_model_summary = True

# Model-specific parameters
# -------------------------
conv_channels = [16, 32, 64]  # Smaller model for quick training
conv_kernel_sizes = [3, 3, 3]
conv_strides = [2, 2, 2]
dropout_rate = 0.0
use_batch_norm = True

# Loss function settings
# ----------------------
loss_function = "cross_entropy"
class_weights = None

# Metrics to track
# ----------------
track_metrics = ["accuracy", "precision", "recall", "f1"]
