# Data Module for NNUE-Vision

This directory contains all dataset-related functionality for the NNUE-Vision project, organized into clear, modular components.

## Structure

```
data/
├── __init__.py          # Module exports and imports
├── datasets.py          # Dataset classes (synthetic + real Visual Wake Words)
├── loaders.py           # Data loading utilities and statistics
├── inspect_dataset.py   # Interactive dataset inspection tool
└── README.md            # This documentation
```

## Components

### 📦 `datasets.py`

Contains dataset classes for Visual Wake Words:

- **`SyntheticVisualWakeWordsDataset`**: Generates synthetic data for testing/demo
  - Creates distinguishable patterns: vertical rectangles for "person", horizontal lines for "no person"
  - Configurable image size, number of samples, and random seed
  - Perfect for development and testing without requiring large downloads

- **`VisualWakeWordsDataset`**: Real Visual Wake Words dataset loader  
  - Uses TensorFlow Datasets to load the actual VWW dataset from COCO
  - Requires `tensorflow-datasets` package
  - Provides access to ~82K training, ~8K validation, and ~8K test images

- **Constants and utilities**:
  - `VWW_IMAGE_SIZE = (96, 96)`: Standard image dimensions
  - `VWW_CLASS_NAMES = ["no_person", "person"]`: Class labels
  - `get_transform()`: Default image preprocessing transforms

### 🔄 `loaders.py`

Provides data loading utilities and statistics:

- **`create_data_loaders()`**: Main function to create train/val/test data loaders
  - Supports both synthetic and real datasets
  - Configurable batch size, workers, subset, etc.
  - Returns tuple of (train_loader, val_loader, test_loader)

- **`get_dataset_stats()`**: Get comprehensive dataset statistics
  - Returns structured dictionary with dataset information
  - Supports both synthetic and real dataset types

- **`print_dataset_stats()`**: Pretty-print dataset statistics
  - Formatted output with emojis and clear sections
  - Error handling for missing dependencies

- **`calculate_dataset_statistics()`**: Analyze data loader contents
  - Computes pixel statistics, class distributions, etc.
  - Progress tracking for large datasets

## Usage Examples

### Basic Data Loading

```python
from data import create_data_loaders

# Create synthetic data loaders for development
train_loader, val_loader, test_loader = create_data_loaders(
    dataset_type="synthetic",
    batch_size=32,
    subset=0.1  # Use 10% for faster iteration
)

# Create real data loaders for production
train_loader, val_loader, test_loader = create_data_loaders(
    dataset_type="real",
    batch_size=64,
    cache_dir="./vww_cache"  # Cache downloaded data
)
```

### Dataset Statistics

```python
from data import print_dataset_stats, get_dataset_stats

# Print formatted statistics
print_dataset_stats("synthetic")
print_dataset_stats("real")

# Get structured statistics
stats = get_dataset_stats("synthetic")
print(f"Dataset has {stats['num_classes']} classes: {stats['classes']}")
```

### Direct Dataset Access

```python
from data.datasets import SyntheticVisualWakeWordsDataset, VisualWakeWordsDataset

# Create synthetic dataset
synthetic_ds = SyntheticVisualWakeWordsDataset(
    split="train",
    num_samples=1000,
    target_size=(96, 96)
)

# Create real dataset (requires tensorflow-datasets)
real_ds = VisualWakeWordsDataset(
    split="train",
    cache_dir="./vww_cache"
)

# Access samples
image, label = synthetic_ds[0]
print(f"Image shape: {image.shape}, Label: {label}")
```

## Dataset Inspector Tool

The `inspect_dataset.py` tool (in `data/` directory) provides interactive dataset exploration:

```bash
# Interactive mode with synthetic data
python -m data.inspect_dataset

# Show statistics only
python -m data.inspect_dataset --stats

# Display sample images
python -m data.inspect_dataset --samples 16

# Use real dataset (requires tensorflow-datasets)
python -m data.inspect_dataset --real --samples 8

# Interactive GUI explorer
python -m data.inspect_dataset
# Then choose option 2 for interactive explorer
```

### Inspector Features

- 📊 **Comprehensive Statistics**: Dataset info, pixel stats, class distributions
- 🖼️ **Sample Visualization**: Grid display of dataset images with labels
- 🎮 **Interactive Explorer**: GUI with navigation controls, histograms, and metadata
- 🔄 **Multi-split Support**: Easy switching between train/val/test splits
- 📈 **Real-time Analysis**: Pixel histograms and class distribution charts

## Dependencies

### Core Dependencies (included in requirements.txt)
- `torch` - PyTorch tensor operations
- `torchvision` - Image transforms and utilities  
- `Pillow` - Image processing
- `numpy` - Numerical operations
- `matplotlib` - Visualization (for inspector tool)

### Optional Dependencies
- `tensorflow-datasets` - For real Visual Wake Words dataset
  ```bash
  pip install tensorflow-datasets
  ```

## Migration from Old Structure

The data loading functionality was moved from the root-level `dataset.py` file to this organized structure:

### Import Changes

**Old imports:**
```python
from dataset import create_data_loaders, SyntheticVisualWakeWordsDataset
```

**New imports:**
```python
from data import create_data_loaders, SyntheticVisualWakeWordsDataset
```

### Enhanced Features

- ✅ Better organization with clear separation of concerns
- ✅ Enhanced error handling and user feedback
- ✅ Support for both synthetic and real datasets
- ✅ Comprehensive statistics and analysis tools
- ✅ Interactive dataset exploration capabilities
- ✅ Improved documentation and examples

### Backward Compatibility

All existing functionality is preserved. The API remains the same, only the import paths have changed. All tests continue to pass after the migration.

## Development

### Adding New Datasets

1. Create a new dataset class in `datasets.py` inheriting from `torch.utils.data.Dataset`
2. Add it to the `__init__.py` exports
3. Update `loaders.py` to support the new dataset type
4. Add appropriate constants and transforms

### Testing

```bash
# Run dataset-specific tests
python -m pytest tests/test_dataset.py -v

# Test the inspector tool
python -m data.inspect_dataset --stats
python -m data.inspect_dataset --samples 4
```

### Contributing

When modifying the data module:
1. Update the appropriate docstrings
2. Add tests for new functionality
3. Update this README if the API changes
4. Test with both synthetic and real datasets 