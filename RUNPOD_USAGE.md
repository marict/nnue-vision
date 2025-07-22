# RunPod Cloud Training for NNUE-Vision

This guide explains how to use the RunPod service to train NNUE-Vision models on cloud GPUs.

## Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **Weights & Biases Account**: Sign up at [wandb.ai](https://wandb.ai)
3. **API Keys**: Get API keys from both services

### Environment Setup

```bash
# Set your API keys
export RUNPOD_API_KEY="your_runpod_api_key_here"
export WANDB_API_KEY="your_wandb_api_key_here"

# Install additional dependencies
pip install -r requirements-runpod.txt
```

### Update Repository URL

Before using, update the `REPO_URL` in `runpod_service.py` to point to your NNUE-Vision repository:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/nnue-vision.git"
```

## Quick Start

### Command Line Usage

```bash
# Basic training
python runpod_service.py train --max_epochs 50 --batch_size 32

# Advanced training with custom GPU
python runpod_service.py train --max_epochs 100 --batch_size 64 --gpu-type "NVIDIA A40" --note "experiment-1"

# Keep pod alive after training (for debugging/analysis)
python runpod_service.py train --max_epochs 50 --keep-alive
```

### Python API Usage

```python
from runpod_service import start_cloud_training

# Launch training job
pod_id = start_cloud_training(
    train_args="--max_epochs 100 --batch_size 64 --learning_rate 0.001",
    gpu_type="NVIDIA RTX 6000 Ada Generation",
    note="my-experiment",
    keep_alive=False
)

print(f"Training started on pod: {pod_id}")
```

## Training Arguments

All training arguments from `train.py` are supported:

### Dataset Arguments
- `--batch_size`: Batch size for training (default: 32)
- `--num_workers`: Number of data loader workers (default: 4)
- `--image_size`: Input image size in pixels (default: 96)

### Model Arguments
- `--learning_rate`: Learning rate (default: 1e-3)

### Training Arguments
- `--max_epochs`: Maximum number of epochs (default: 50)
- `--gpus`: GPUs to use (handled automatically on RunPod)
- `--accelerator`: Accelerator type (handled automatically)

### Logging Arguments
- `--log_dir`: Directory for logs (automatically set to persistent storage)
- `--project_name`: Project name for W&B logging (default: "visual_wake_words")
- `--save_top_k`: Number of best models to save (default: 3)
- `--patience`: Early stopping patience (default: 10)

## GPU Types

Available GPU types (may vary by region):
- `NVIDIA RTX 2000 Ada Generation` (default, cost-effective)
- `NVIDIA RTX 6000 Ada Generation`
- `NVIDIA A40`
- `NVIDIA A100 80GB PCIe`
- `NVIDIA H100 PCIe`
- `NVIDIA L40S`

## Monitoring Training

1. **Weights & Biases**: The script automatically opens your W&B run in the browser
2. **Logs**: All training logs are saved to `/runpod-volume/train_YYYYMMDD_HHMMSS.log`
3. **Checkpoints**: Model checkpoints are automatically saved to persistent storage

## File Persistence

The following files/directories are automatically saved to persistent storage:
- Training logs: `/runpod-volume/train_*.log`
- Lightning logs: `/runpod-volume/nnue-vision-logs/`
- Model checkpoints: Saved within Lightning logs directory
- Final model: `/runpod-volume/visual_wake_words_model.pt`

## Examples

See `examples/runpod_training_example.py` for comprehensive usage examples.

### Basic Training
```bash
python runpod_service.py train --max_epochs 50 --batch_size 32 --note "baseline-run"
```

### Large Image Experiment
```bash
python runpod_service.py train \
    --max_epochs 100 \
    --batch_size 32 \
    --image_size 224 \
    --learning_rate 0.0005 \
    --gpu-type "NVIDIA A40" \
    --note "large-images"
```

### Quick Test with Minimal Settings
```bash
python runpod_service.py train --max_epochs 5 --batch_size 32 --note "quick-test"
```

## Stopping Pods

```python
from runpod_service import stop_runpod

# Stop specific pod
stop_runpod(pod_id="your_pod_id")

# Stop using environment variables
# (set RUNPOD_POD_ID and RUNPOD_API_KEY)
stop_runpod()
```

## Troubleshooting

### Common Issues

1. **"Repository not found"**: Make sure `REPO_URL` points to your repository
2. **W&B not logging**: Ensure `WANDB_API_KEY` is set correctly
3. **Pod creation fails**: Check your RunPod credits and GPU availability
4. **Training fails**: Check the training logs in `/runpod-volume/train_*.log`

### Getting Help

1. Check the training logs for detailed error messages
2. Use `--keep-alive` flag to keep the pod running for debugging
3. Access the pod via SSH if needed (modify the script to enable SSH)

### Network Volume

Update the `network_volume_id` in `runpod_service.py` to match your RunPod network volume for persistent storage across pod restarts.

## Cost Optimization

1. Use smaller GPU types for initial experiments
2. Don't use `--keep-alive` unless needed for debugging
3. Set appropriate `--max_epochs` and `--patience` for early stopping
4. Monitor your RunPod balance regularly 