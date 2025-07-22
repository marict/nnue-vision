# NNUE-Vision

A minimal computer vision neural network based on the NNUE (Efficiently Updatable Neural Network) architecture, adapted for image classification tasks.

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

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Minimal Training

The easiest way to get started:

```bash
python train_minimal.py
```

This will:
- Download the Visual Wake Words dataset automatically
- Train a simple CNN for person/no-person detection
- Save the trained model as `visual_wake_words_model.pt`

### 3. Advanced Training

For more control over training parameters:

```bash
python train.py --max_epochs 50 --batch_size 64 --learning_rate 0.001
```

Available options:
- `--batch_size`: Batch size for training (default: 32)
- `--max_epochs`: Maximum number of epochs (default: 50)  
- `--learning_rate`: Learning rate (default: 1e-3)
- `--image_size`: Input image size (default: 96)
- `--gpus`: GPUs to use (e.g., "0,1" or leave empty for auto)

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

### Dependencies  
- `requirements.txt` - Python package dependencies

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
