"""ImageNet training configuration for RunPod cloud GPUs.

Optimized settings for training NNUE and EtinyNet models on ImageNet.
Uses smaller batch sizes due to 224x224 input images and appropriate
learning rate schedules for large-scale training.
"""

# Project identification
name = "imagenet-nnue-etinynet"

# Dataset and model settings (ImageNet optimized)
batch_size = 32  # Smaller due to 224x224 images and memory constraints
num_workers = 8  # Utilize cloud CPU cores
input_size = (224, 224)  # ImageNet standard resolution
num_classes = 1000  # ImageNet class count
learning_rate = 1e-4  # Lower LR for large dataset convergence
subset = 1.0  # Use full ImageNet dataset

# Training settings (long-running ImageNet job)
max_epochs = 50  # Reasonable for ImageNet (each epoch is massive)
patience = 10  # Patience for large dataset training
save_top_k = 3  # Save top models for analysis

# System settings (cloud GPU optimized)
accelerator = "auto"
devices = "auto"
deterministic = True
seed = 42

# Logging and monitoring (essential for long ImageNet training)
log_interval = 100  # More frequent logging for progress tracking
always_save_checkpoint = True
enable_progress_bar = True
check_val_every_n_epoch = 1

# Logging backends (critical for cloud tracking)
use_wandb = True  # Essential for ImageNet training tracking
use_tensorboard = False  # Prefer wandb for cloud
log_dir = "logs"
project_name = "imagenet_nnue_etinynet"
wandb_api_key = None  # Set via environment variable

# ImageNet-specific settings
dataset_name = "imagenet"  # Use ImageNet dataset
data_root = "/workspace/data"  # Common RunPod data mount point

# Memory and performance optimization
grad_accumulation_steps = 4  # Simulate larger batch size with gradient accumulation
mixed_precision = True  # Enable automatic mixed precision for memory efficiency
pin_memory = True  # Pin memory for faster GPU transfer

# Model-specific overrides for ImageNet
# These will be applied when creating models
imagenet_model_settings = {
    "nnue": {
        # Adjust feature set for ImageNet scale
        "grid_size": 14,  # 224/16 = 14 for reasonable feature density
        "num_features_per_square": 16,  # More features for complex images
        "l1_size": 1024,  # Keep 0.98M parameter target
        "visual_threshold": 0.3,  # Slightly lower threshold for natural images
    },
    "etinynet": {
        "variant": "0.98M",  # Use our balanced parameter variant
        "input_size": 224,  # ImageNet resolution
        "use_asq": False,  # Disable quantization for initial training
    },
}
