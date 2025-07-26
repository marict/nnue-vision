# NNUE-Vision

Neural Network Efficiently Updatable (NNUE) adapted for computer vision tasks, featuring NNUE and EtinyNet architectures for efficient image classification.

## Overview

This project implements NNUE (Neural Network Efficiently Updatable) architectures for computer vision, originally designed for chess engines but adapted for image classification tasks. It includes:

- **NNUE models**: Efficient neural networks with bucketed layer stacks
- **EtinyNet models**: Extremely lightweight CNNs for edge deployment
- **Multiple datasets**: CIFAR-10, CIFAR-100, and ImageNet support
- **C++ engines**: High-performance inference implementations
- **Cloud training**: RunPod integration for GPU training

## Quick Start

### Local Training

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Train NNUE model on CIFAR-10
python train.py nnue --config config/train_nnue_default.py

# Train EtinyNet model on CIFAR-10  
python train.py etinynet --config config/train_etinynet_default.py

# Override specific parameters
python train.py nnue --config config/train_nnue_default.py --max_epochs 50 --batch_size 64
```

### Cloud Training (RunPod)

```bash
# Install dependencies (includes RunPod support)
pip install -r requirements-dev.txt

# Set environment variables
export WANDB_API_KEY=your_wandb_key
export RUNPOD_API_KEY=your_runpod_key

# Start NNUE cloud training
python runpod_service.py train --script train.py --config config/train_nnue_default.py --model nnue

# Start EtinyNet cloud training
python runpod_service.py train --script train.py --config config/train_etinynet_default.py --model etinynet
```

## Model Architectures

### NNUE (Neural Network Efficiently Updatable)
- Grid-based feature extraction from images
- Bucketed layer stacks for efficiency
- Quantization-ready for C++ deployment
- ~1M parameters (configurable)

### EtinyNet (Extremely Tiny Network)
- Linear Depthwise Blocks (LB) and Dense Linear Depthwise Blocks (DLB)
- Multiple variants: 0.75 (680K params), 1.0 (976K params)
- Optimized for TinyML and edge devices

## Supported Datasets

- **CIFAR-10**: 10-class object classification (32×32)
- **CIFAR-100**: 100-class object classification (32×32)  
- **ImageNet**: 1000-class classification (224×224)

## Project Structure

### Core Training
- `train_nnue.py` - NNUE model training script
- `train_etinynet.py` - EtinyNet model training script
- `model.py` - Model implementations (NNUE, EtinyNet)
- `runpod_service.py` - Cloud training orchestration

### Data & Configuration
- `data/` - Dataset loaders and utilities
- `config/` - Training configuration files
- `serialize.py` - Model serialization for C++ deployment

### C++ Engine
- `engine/` - High-performance C++ inference implementations
- `lib/` - C++ headers and utilities

### Testing & Benchmarks
- `tests/` - Comprehensive test suite
- `benchmarks/` - Performance benchmarking and MCU simulation

### Scripts
- `scripts/` - Training utilities and benchmark scripts
- `container_setup.sh` - Docker environment setup

## Performance

### NNUE Models
- **Parameters**: ~1M (configurable)
- **Accuracy**: 85-90% on CIFAR-10
- **C++ inference**: <1ms on modern CPUs

### EtinyNet Models
- **EtinyNet-0.75**: 680K parameters, 75M MAdds
- **EtinyNet-1.0**: 976K parameters, 117M MAdds
- **Accuracy**: 90-95% on CIFAR-10

## Development

### Testing
```bash
# Run all tests
python -m pytest tests/ --timeout=300

# Run specific test categories
python -m pytest tests/test_model.py -v
python -m pytest tests/test_training.py -v
```

### Benchmarking
```bash
# Run MCU benchmarks
python scripts/run_mcu_benchmarks.py --model checkpoints/model.pt --dataset cifar10

# Run example benchmark
python run_example_benchmark.py
```

## C++ Engine Integration

Models can be serialized for high-performance C++ inference:

```bash
# Serialize NNUE model
python serialize.py --model checkpoints/nnue_model.ckpt --output model.nnue

# Serialize EtinyNet model  
python serialize.py --model checkpoints/etinynet_model.ckpt --output model.etiny
```

## License

This project follows the same license as the original NNUE PyTorch project.
