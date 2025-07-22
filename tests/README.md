# NNUE-Vision Tests

This directory contains comprehensive tests for the NNUE-Vision project, verifying dataset import functionality and NNUE forward/backward passes.

## Test Structure

- `test_dataset.py` - Tests for dataset import and data loading functionality
- `test_model.py` - Tests for NNUE model forward and backward passes
- `test_integration.py` - End-to-end integration tests
- `conftest.py` - Shared pytest fixtures and utilities

## Running Tests

### Prerequisites

Install core dependencies and testing dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# From project root
pytest tests/

# With coverage report
pytest tests/ --cov=. --cov-report=html

# Verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_dataset.py
pytest tests/test_model.py
pytest tests/test_integration.py
```

### Run Tests in Parallel

```bash
# Use multiple CPU cores
pytest tests/ -n auto
```

### Common Test Commands

```bash
# Quick test run (just check everything works)
pytest tests/ -x --tb=short

# Full test with coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term-missing

# Debug mode (stop on first failure, verbose)
pytest tests/ -vvs -x

# Run only dataset tests
pytest tests/test_dataset.py -v

# Run only model tests  
pytest tests/test_model.py -v

# Run only integration tests
pytest tests/test_integration.py -v
```

## Test Categories

#### Dataset Tests (`test_dataset.py`)
- Dataset creation and configuration
- Data loading and iteration
- Data transformations and preprocessing
- Shape and type consistency
- Integration with PyTorch data loaders

#### Model Tests (`test_model.py`)
- Model architecture and initialization
- Forward pass functionality
- Backward pass and gradient computation
- PyTorch Lightning integration
- Model persistence (save/load)
- Robustness and edge cases

#### Integration Tests (`test_integration.py`)
- Complete training workflows
- Data pipeline to model integration
- PyTorch Lightning trainer functionality
- Memory usage and performance
- Error handling

## Test Configuration

Tests are configured to:
- Work on both CPU and GPU (if available)
- Use small datasets for fast execution
- Include comprehensive edge case testing
- Verify numerical stability and correctness

## Fixtures Available

The `conftest.py` file provides shared fixtures:
- `device` - Best available device (CPU/GPU)
- `model_params` - Standard model parameters
- `simple_model` - Basic SimpleCNN instance
- `trained_model` - Pre-trained model for testing
- `sample_batch` - Sample batch of data
- `data_loaders` - Small data loaders for testing

## Expected Test Coverage

The test suite verifies:
- ✅ Dataset import works correctly
- ✅ Forward passes produce valid outputs
- ✅ Backward passes compute gradients
- ✅ Training loops update parameters
- ✅ Model can be saved and loaded
- ✅ Integration with PyTorch Lightning
- ✅ Error handling for edge cases
- ✅ Memory efficiency
- ✅ Numerical stability 