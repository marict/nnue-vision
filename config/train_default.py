"""Local development configuration for NNUE-Vision training.

Minimal settings for quick local development and testing.
Perfect for experimenting with the Visual Wake Words dataset on your local machine.
"""

# Project identification
name = "nnue_vision-local"

# Dataset and model settings
batch_size = 16  # Small batch size for local development
num_workers = 2  # Conservative for local machines
input_size = (96, 96)  # Square input images
num_classes = 2  # person/no person classification
learning_rate = 1e-3

# Training settings
max_epochs = 10  # Quick training for local testing
patience = 5  # Early stopping patience
save_top_k = 1  # Save only the best model

# System settings
accelerator = "auto"  # Auto-detect GPU/CPU
devices = "auto"
deterministic = True
seed = 42

# Logging and monitoring
log_interval = 20  # Log every 20 steps
always_save_checkpoint = True
enable_progress_bar = True
check_val_every_n_epoch = 1

# Logging backends
use_wandb = False  # Disabled for local development
use_tensorboard = False  # Disabled for local development
log_dir = "logs"
project_name = "visual_wake_words_local"
