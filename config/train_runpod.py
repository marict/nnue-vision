"""RunPod cloud training configuration for NNUE-Vision.

Optimized for large-scale training on RunPod's cloud GPUs.
Higher batch sizes, more epochs, and comprehensive logging.
"""

# Project identification
name = "nnue_vision-runpod"

# Dataset and model settings (optimized for cloud GPUs)
batch_size = 64  # Large batch size for GPU efficiency
num_workers = 8  # Utilize cloud CPU cores
input_size = (96, 96)
num_classes = 2
learning_rate = 1e-3

# Training settings (long-running cloud job)
max_epochs = 100  # Extended training for better results
patience = 15  # Higher patience for cloud training
save_top_k = 5  # Save more models for analysis

# System settings (cloud GPU optimized)
accelerator = "auto"
devices = "auto"
deterministic = True
seed = 42

# Logging and monitoring (cloud-friendly)
log_interval = 50  # Reasonable logging frequency
always_save_checkpoint = True
enable_progress_bar = True
check_val_every_n_epoch = 1

# Logging backends (enabled for cloud tracking)
use_wandb = True  # Essential for cloud training tracking
use_tensorboard = False  # Prefer wandb for cloud
log_dir = "logs"
project_name = "visual_wake_words_runpod"
wandb_api_key = None  # Set via environment variable
