"""Default configuration for NNUE-Vision training.

This configuration provides reasonable defaults for training a CNN on the Visual Wake Words dataset.
Modify these values as needed for your specific training requirements.
"""

# Project settings
# ----------------
name = "nnue_vision-default"

# Training intervals and logging
# ------------------------------
eval_interval = 1  # Epochs between evaluations
log_interval = 50  # Steps between logging
eval_only = False
always_save_checkpoint = True
clear_previous_checkpoints = False

# Model initialization
init_from = "scratch"  # or "resume" to continue from checkpoint

# Dataset configuration
# ---------------------
dataset = "visual_wake_words"  # Dataset to use
batch_size = 32
num_workers = 4
image_size = 96  # Input image size (square)

# Model architecture
# ------------------
input_size = (96, 96)
num_classes = 2  # person/no person for Visual Wake Words
learning_rate = 1e-3

# Training hyperparameters
# ------------------------
max_epochs = 50
patience = 10  # Early stopping patience
save_top_k = 3  # Number of best models to save

# Optimization settings
# ---------------------
# Note: These would be used if implementing custom optimizers
weight_decay = 1e-4
beta1 = 0.9
beta2 = 0.999
grad_clip = 1.0

# Learning rate schedule (for future implementation)
# --------------------------------------------------
warmup_epochs = 5
lr_decay_epochs = 40
min_lr = 1e-6

use_cyclical_lr = False
cyclical_lr_period = 10
cyclical_lr_amplitude = 0.1

# System settings
# ---------------
accelerator = "auto"  # "cpu", "gpu", or "auto"
devices = "auto"  # Device specification
deterministic = True
precision = "32"  # "16", "32", or "bf16"

# Logging and checkpointing
# -------------------------
log_dir = "logs"
project_name = "visual_wake_words"
use_wandb = True
use_tensorboard = False
wandb_api_key = None  # Set to your API key or use environment variable

# Data augmentation (for future implementation)
# ----------------------------------------------
use_augmentation = True
augmentation_params = {
    "rotation_degrees": 10,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
}

# Random seeds
# ------------
seed = 42

# Advanced settings
# -----------------
compile_model = False  # PyTorch 2.0 compilation
check_val_every_n_epoch = 1
enable_progress_bar = True
enable_model_summary = True

# Model-specific parameters
# -------------------------
# CNN architecture settings (for future customization)
conv_channels = [32, 64, 128]
conv_kernel_sizes = [3, 3, 3]
conv_strides = [2, 2, 2]
dropout_rate = 0.0
use_batch_norm = True

# Loss function settings
# ----------------------
loss_function = "cross_entropy"
class_weights = None  # For imbalanced datasets

# Metrics to track
# ----------------
track_metrics = ["accuracy", "precision", "recall", "f1", "confusion_matrix"]
