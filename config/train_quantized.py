"""Configuration for Quantized NNUE-Vision training.

This configuration is optimized for quantization-aware training of the NNUE model
with minimal model size and maximum efficiency.
"""

# Project settings
# ----------------
name = "nnue_vision-quantized"

# Training intervals and logging
# ------------------------------
eval_interval = 1  # Epochs between evaluations
log_interval = 25  # Steps between logging (more frequent for QAT monitoring)
eval_only = False
always_save_checkpoint = True
clear_previous_checkpoints = False

# Model initialization
init_from = "scratch"  # or "resume" to continue from checkpoint

# Dataset configuration
# ---------------------
dataset = "visual_wake_words"  # Dataset to use
batch_size = 64  # Larger batch size for stable QAT
num_workers = 4
image_size = 96  # Input image size (square)

# Quantized Model architecture
# ----------------------------
input_size = (96, 96)
num_classes = 2  # person/no person for Visual Wake Words
learning_rate = 8e-4  # Slightly lower LR for quantized training

# Quantization-specific parameters
# --------------------------------
feature_dim = 128  # Compressed feature dimension
hidden_dim = 64  # Hidden layer dimension
ft_scale = 127  # Feature transformer scale (traditional NNUE)
hidden_scale = 127  # Hidden layer scale
output_scale = 361  # Output scale (traditional NNUE)
weight_bits = 8  # Weight quantization bits
activation_bits = 8  # Activation quantization bits
hidden_bits = 16  # Hidden layer bits

# Training hyperparameters
# ------------------------
max_epochs = 60  # More epochs for QAT convergence
patience = 15  # Higher patience for QAT
save_top_k = 3  # Number of best models to save

# Optimization settings for QAT
# ------------------------------
weight_decay = 1e-5  # Lower weight decay for quantized training
beta1 = 0.9
beta2 = 0.999
grad_clip = 0.5  # Lower gradient clipping for stability

# Learning rate schedule optimized for QAT
# -----------------------------------------
warmup_epochs = 3  # Shorter warmup for QAT
lr_decay_epochs = 50
min_lr = 1e-6
use_onecycle_lr = True  # Use OneCycleLR for QAT

# QAT-specific settings
# ---------------------
qat_start_epoch = 5  # Start QAT after initial training
fake_quantize = True  # Use fake quantization during training
calibration_steps = 100  # Steps for calibration

# System settings
# ---------------
accelerator = "auto"  # "cpu", "gpu", or "auto"
devices = "auto"  # Device specification
deterministic = True
precision = "32"  # Keep FP32 for QAT stability

# Logging and checkpointing
# -------------------------
log_dir = "logs"
project_name = "visual_wake_words_quantized"
use_wandb = True
use_tensorboard = False
wandb_api_key = None  # Set to your API key or use environment variable

# Data augmentation - minimal for quantized training
# --------------------------------------------------
use_augmentation = True
augmentation_params = {
    "rotation_degrees": 5,  # Reduced augmentation
    "brightness": 0.1,
    "contrast": 0.1,
    "saturation": 0.1,
    "hue": 0.05,
}

# Random seeds
# ------------
seed = 42

# Advanced settings
# -----------------
compile_model = False  # Disable compilation for QAT
check_val_every_n_epoch = 1
enable_progress_bar = True
enable_model_summary = True

# Quantization settings
# --------------------
quantization_backend = "fbgemm"  # Quantization backend
observer_type = "minmax"  # Observer for calibration
qscheme = "per_tensor_symmetric"  # Quantization scheme

# Model export settings
# --------------------
export_quantized = True  # Export quantized model
export_jit = True  # Export JIT traced model
export_onnx = True  # Export ONNX model
target_device = "cpu"  # Target deployment device

# Loss function settings
# ----------------------
loss_function = "cross_entropy"
class_weights = None  # For imbalanced datasets
label_smoothing = 0.1  # Light label smoothing for QAT

# Metrics to track
# ----------------
track_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "model_size",
    "compression_ratio",
    "inference_speed",
]

# Early stopping criteria
# -----------------------
early_stopping_metric = "val_acc"
early_stopping_mode = "max"
early_stopping_min_delta = 0.001

# Model size optimization
# ----------------------
target_model_size_kb = 50  # Target model size in KB
prune_model = False  # Model pruning (future feature)
knowledge_distillation = False  # Knowledge distillation (future feature)

# Deployment settings
# -------------------
optimize_for_mobile = True  # Optimize for mobile deployment
optimize_for_inference = True  # Optimize for inference speed
batch_inference = False  # Single image inference optimization
