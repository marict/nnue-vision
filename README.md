# NNUE-Vision

> Modern NNUE techniques applied to computer vision tasks

## Overview

This project started as the original [NNUE PyTorch](https://github.com/official-stockfish/nnue-pytorch) codebase designed for chess evaluation, but has been stripped down and adapted for computer vision tasks. The current implementation focuses on binary image classification using the **Visual Wake Words** dataset.

### What's Changed from Original NNUE

- ✅ **Removed**: All chess-specific components (feature extraction, piece evaluation, game data loading)
- ✅ **Removed**: Complex C++ data loaders and chess position encoding  
- ✅ **Removed**: Chess-specific loss functions and evaluation metrics
- ✅ **Added**: Simple CNN architecture for image classification
- ✅ **Added**: Visual Wake Words dataset integration via TensorFlow Datasets
- ✅ **Added**: Standard computer vision preprocessing and data augmentation
- ✅ **Kept**: PyTorch Lightning training framework for easy experimentation

## Quick Start

### Configuration-Based Training (New!)

The project now supports flexible configuration files for easy experiment management:

```bash
# Quick testing (5 epochs, minimal setup)
python train_minimal.py --config config/train_minimal.py

# Full training with wandb logging
python train.py --config config/train_default.py

# Custom experiment
python train.py --config examples/custom_config_example.py --batch_size 64
```

Create your own config files to define training parameters:
```python
# my_config.py
name = "my_experiment"
batch_size = 64
max_epochs = 30
learning_rate = 2e-3
use_wandb = True
# ... and more
```

📖 **See [Configuration System Documentation](docs/configuration_system.md) for complete details**

### Legacy Training (Command Line)

You can still use the original command-line interface:

```bash
# Minimal training for quick testing
python train_minimal.py

# Full training with wandb logging
python train.py --batch_size 32 --max_epochs 50 --learning_rate 1e-3
```

### 4. Comprehensive Experiment Tracking with Weights & Biases

The training script now includes comprehensive wandb logging for detailed experiment tracking:

```bash
# Set your wandb API key
export WANDB_API_KEY=your_api_key_here

# Run training with wandb logging
python train.py --project_name my_experiment --note "baseline_run"

# Run multiple experiments for comparison
python scripts/train_with_wandb.py
```

**Wandb Features:**
- 🔍 **Detailed Metrics**: Training/validation loss, accuracy, precision, recall, F1
- 📊 **System Monitoring**: GPU usage, memory consumption, training speed
- 🎯 **Gradient Analysis**: Gradient norms, parameter tracking, optimization insights
- 📈 **Visualizations**: Confusion matrices, sample predictions, learning curves
- 🏷️ **Model Artifacts**: Automatic model saving and versioning
- ⚡ **Performance Tracking**: Step timing, throughput, resource utilization

See [`docs/wandb_logging.md`](docs/wandb_logging.md) for complete documentation.

## Dataset: Visual Wake Words

The Visual Wake Words dataset is designed for binary classification: detecting whether a person is present in an image or not. It's commonly used for benchmarking efficient neural networks on mobile devices.

- **Train**: ~82,000 images
- **Validation**: ~8,000 images  
- **Test**: ~8,000 images
- **Classes**: 2 (person/no person)
- **Input size**: 96x96 pixels (RGB)

The dataset will be automatically downloaded via TensorFlow Datasets on first run.

## Model Architecture

The current model is a simple CNN optimized for the Visual Wake Words task:

```
Input (96x96x3)
    ↓
Conv2D(32) + BatchNorm + ReLU + Stride(2)
    ↓  
Conv2D(64) + BatchNorm + ReLU + Stride(2)
    ↓
Conv2D(128) + BatchNorm + ReLU + Stride(2)
    ↓
Global Average Pooling
    ↓
Dense(2) → Softmax
```

**Parameters**: ~45,000 (much smaller than typical vision models)

## File Structure

### Core Files
- `model.py` - CNN model definition using PyTorch Lightning
- `dataset.py` - Visual Wake Words dataset loader and preprocessing
- `train.py` - Full training script with all options
- `train_minimal.py` - Simple training script for quick demos

### Scripts and Examples
- `scripts/train_with_wandb.py` - Example script for running multiple wandb experiments

### Dependencies  
- `requirements.txt` - Python package dependencies

### Documentation
- `docs/wandb_logging.md` - Complete guide to wandb experiment tracking
- `docs/configuration_system.md` - Configuration system documentation

### Logs and Outputs
- `logs/` - TensorBoard logs and model checkpoints
- `visual_wake_words_model.pt` - Saved model weights

## Results

With default settings (10 epochs), you can expect:
- **Training time**: ~5-10 minutes on GPU, ~30 minutes on CPU
- **Model size**: ~180KB
- **Test accuracy**: ~85-90% (varies with training time)

## Extending to Other Datasets

The codebase is designed to be easily adaptable to other computer vision tasks:

1. **Replace dataset loader** in `dataset.py` with your data
2. **Adjust model architecture** in `model.py` for your task
3. **Modify loss function** if needed (currently uses CrossEntropy for classification)

Example datasets you could try:
- CIFAR-10/100 for object classification
- Fashion-MNIST for clothing classification  
- Custom image datasets

## Development Notes

This is a simplified version focused on demonstrating how to adapt NNUE concepts to computer vision. For production use, consider:

- More sophisticated data augmentation
- Learning rate scheduling
- Model ensembling
- Advanced architectures (ResNet, EfficientNet, etc.)

## License

Same as original NNUE PyTorch project.
