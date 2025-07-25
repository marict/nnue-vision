# NNUE-Vision 

Neural Network Efficiently Updatable (NNUE) adapted for computer vision tasks, specifically Visual Wake Words detection.

## Quick Start

### Basic Training

```bash
# Train with default configuration
python train_nnue.py

# Train with custom configuration  
python train_nnue.py --config config/train_default.py

# Override specific parameters
python train_nnue.py --config config/train_default.py --max_epochs 50 --batch_size 64
```

### Cloud Training (RunPod)

```bash
# Install cloud dependencies
pip install -r requirements-runpod.txt

# Start cloud training
python runpod_service.py train --max_epochs 100 --batch_size 64

# See RUNPOD_USAGE.md for detailed instructions
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/nnue-vision.git
cd nnue-vision

# Install dependencies
pip install -r requirements.txt

# For cloud training
pip install -r requirements-runpod.txt
```

### 4. Comprehensive Experiment Tracking with Weights & Biases

The training script now includes comprehensive wandb logging for detailed experiment tracking:

```bash
# Set your wandb API key
export WANDB_API_KEY=your_api_key_here

# Run training with wandb logging
python train_nnue.py --project_name my_experiment --note "baseline_run"

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
- `train_nnue.py` - Training script with wandb/tensorboard logging support

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
