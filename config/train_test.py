"""Test configuration for NNUE-Vision training - minimal settings for fast e2e testing."""

# Project identification
name = "nnue_vision-test"

# Dataset and model settings - minimal for fast testing
batch_size = 4  # Very small batch size
num_workers = 0  # No parallel workers to avoid complications
input_size = (96, 96)  # Square input images
num_classes = 2  # person/no person classification
learning_rate = 1e-3
subset = 0.01  # Use only 1% of dataset for speed

# Training settings - minimal for fast testing
max_epochs = 2  # Very quick training
patience = 1  # Early stopping after 1 epoch without improvement
save_top_k = 1  # Save only the best model

# System settings
accelerator = "auto"  # Auto-detect GPU/CPU
devices = "auto"
deterministic = True
seed = 42

# Logging and monitoring - minimal
log_interval = 1  # Log every step for visibility
always_save_checkpoint = True
enable_progress_bar = True
check_val_every_n_epoch = 1

# Logging backends - disabled for testing

log_dir = "logs"
project_name = "visual_wake_words_test"

# NNUE-specific settings
num_ls_buckets = 2  # Minimal number of layer stacks for testing
visual_threshold = 0.0
start_lambda = 1.0
end_lambda = 1.0
