import sys

sys.path.insert(0, ".")

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import torch

from data.loaders import create_data_loaders
from evaluate import (
    compute_metrics,
    evaluate_compiled_model,
    evaluate_model,
    extract_nnue_features,
)
from nnue import NNUE, GridFeatureSet


def test_nnue_feature_extraction_fix():
    """Test that NNUE feature extraction now uses the proper pipeline."""

    print("=== Testing NNUE Feature Extraction Fix ===")

    # Create a simple NNUE model
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=2)
    model = NNUE(
        feature_set=feature_set,
        l1_size=32,
        l2_size=16,
        l3_size=4,
        num_classes=10,  # CIFAR-10 has 10 classes
        input_size=32,
    )

    # Create sample images
    images = torch.randn(4, 3, 32, 32)  # 4 samples, 3 channels, 32x32

    # Test the feature extraction function
    outputs = extract_nnue_features(model, images)

    print(f"Model outputs shape: {outputs.shape}")
    print(f"Expected shape: [4, 10] (batch_size, num_classes)")

    assert outputs.shape == (4, 10), f"Expected shape (4, 10), got {outputs.shape}"
    print("✅ NNUE feature extraction produces correct output shape")


def test_compiled_vs_validation_consistency():
    """Test that compiled and validation metrics are now consistent."""

    print("\n=== Testing Compiled vs Validation Consistency ===")

    # Create a simple NNUE model
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=2)
    model = NNUE(
        feature_set=feature_set,
        l1_size=32,
        l2_size=16,
        l3_size=4,
        num_classes=10,  # CIFAR-10 has 10 classes
        input_size=32,
    )

    # Create small data loaders for testing
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="cifar10",
        batch_size=4,
        max_samples_per_split=20,
        use_augmentation=False,  # Disable augmentation for consistent testing
    )

    # Define a simple loss function
    def simple_loss_fn(model, batch):
        images, labels = batch
        outputs = model(images)
        return torch.nn.functional.cross_entropy(outputs, labels)

    # Test validation evaluation
    val_loss, val_metrics = evaluate_model(model, val_loader, simple_loss_fn)
    print(f"Validation metrics: {val_metrics}")

    # Test compiled evaluation
    compiled_metrics = evaluate_compiled_model(model, val_loader, "nnue")
    print(f"Compiled metrics: {compiled_metrics}")

    # Check that metrics are similar (within reasonable tolerance)
    f1_diff = abs(val_metrics["f1"] - compiled_metrics["f1"])
    acc_diff = abs(val_metrics["acc"] - compiled_metrics["acc"])

    print(f"F1 difference: {f1_diff:.4f}")
    print(f"Accuracy difference: {acc_diff:.4f}")

    # The metrics should be very close since we're now using the same pipeline
    assert f1_diff < 0.1, f"F1 difference too large: {f1_diff}"
    assert acc_diff < 0.1, f"Accuracy difference too large: {acc_diff}"

    print("✅ Compiled and validation metrics are now consistent")


def test_compiled_evaluation_no_wrong_features():
    """Test that compiled evaluation no longer uses the wrong feature extraction."""

    print("\n=== Testing No Wrong Feature Extraction ===")

    # Create a simple NNUE model
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=2)
    model = NNUE(
        feature_set=feature_set,
        l1_size=32,
        l2_size=16,
        l3_size=4,
        num_classes=10,  # CIFAR-10 has 10 classes
        input_size=32,
    )

    # Create a mock loader with known data
    mock_loader = Mock()
    sample_images = torch.randn(4, 3, 32, 32)
    sample_labels = torch.tensor([0, 1, 2, 3])  # Use valid CIFAR-10 class indices
    mock_loader.__iter__ = Mock(return_value=iter([(sample_images, sample_labels)]))
    mock_loader.__len__ = Mock(return_value=1)

    # Test that compiled evaluation works without the wrong feature extraction
    compiled_metrics = evaluate_compiled_model(model, mock_loader, "nnue")

    print(f"Compiled metrics: {compiled_metrics}")

    # The metrics should be reasonable (not artificially high like before)
    assert (
        compiled_metrics["f1"] < 0.9
    ), f"Compiled F1 too high: {compiled_metrics['f1']}"
    assert (
        compiled_metrics["acc"] < 0.9
    ), f"Compiled accuracy too high: {compiled_metrics['acc']}"

    print("✅ Compiled evaluation no longer produces artificially high metrics")


def test_metric_calculation_consistency():
    """Test that metric calculation is consistent across different scenarios."""

    print("\n=== Testing Metric Calculation Consistency ===")

    # Test case 1: Perfect predictions (binary classification)
    outputs = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([0, 1, 0, 1])

    metrics = compute_metrics(outputs, targets)
    print(f"Perfect predictions metrics: {metrics}")

    assert metrics["acc"] == 1.0, f"Expected accuracy 1.0, got {metrics['acc']}"
    assert metrics["f1"] == 1.0, f"Expected F1 1.0, got {metrics['f1']}"

    # Test case 2: Random predictions
    outputs = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    targets = torch.tensor([0, 1, 0, 1])

    metrics = compute_metrics(outputs, targets)
    print(f"Random predictions metrics: {metrics}")

    # Random predictions should give around 0.5 accuracy
    assert (
        0.4 <= metrics["acc"] <= 0.6
    ), f"Random accuracy should be ~0.5, got {metrics['acc']}"

    print("✅ Metric calculation is consistent across scenarios")


def test_compiled_evaluation_error_handling():
    """Test that compiled evaluation handles errors gracefully."""

    print("\n=== Testing Compiled Evaluation Error Handling ===")

    # Test with non-existent C++ executable
    model = Mock()
    mock_loader = Mock()

    with patch("pathlib.Path.exists", return_value=False):
        try:
            evaluate_compiled_model(model, mock_loader, "nnue")
            assert False, "Should have raised RuntimeError for missing executable"
        except RuntimeError as e:
            assert "C++ NNUE engine not found" in str(e)
            print("✅ Correctly handles missing C++ executable")

    # Test with invalid model type
    try:
        evaluate_compiled_model(model, mock_loader, "invalid_model")
        assert False, "Should have raised ValueError for invalid model type"
    except ValueError as e:
        assert "Unknown model type" in str(e)
        print("✅ Correctly handles invalid model type")


def test_end_to_end_consistency():
    """Test end-to-end consistency between PyTorch and compiled evaluation."""

    print("\n=== Testing End-to-End Consistency ===")

    # Create a simple NNUE model
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=2)
    model = NNUE(
        feature_set=feature_set,
        l1_size=32,
        l2_size=16,
        l3_size=4,
        num_classes=10,  # CIFAR-10 has 10 classes
        input_size=32,
    )

    # Create small data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="cifar10",
        batch_size=4,
        max_samples_per_split=20,
        use_augmentation=False,
    )

    # Define loss function
    def simple_loss_fn(model, batch):
        images, labels = batch
        outputs = model(images)
        return torch.nn.functional.cross_entropy(outputs, labels)

    # Run both evaluations
    val_loss, val_metrics = evaluate_model(model, val_loader, simple_loss_fn)
    compiled_metrics = evaluate_compiled_model(model, val_loader, "nnue")

    print(f"Validation metrics: {val_metrics}")
    print(f"Compiled metrics: {compiled_metrics}")

    # Check consistency
    f1_diff = abs(val_metrics["f1"] - compiled_metrics["f1"])
    acc_diff = abs(val_metrics["acc"] - compiled_metrics["acc"])

    print(f"F1 difference: {f1_diff:.4f}")
    print(f"Accuracy difference: {acc_diff:.4f}")

    # Metrics should be very close
    assert f1_diff < 0.05, f"F1 difference too large: {f1_diff}"
    assert acc_diff < 0.05, f"Accuracy difference too large: {acc_diff}"

    print("✅ End-to-end consistency verified")


if __name__ == "__main__":
    test_nnue_feature_extraction_fix()
    test_compiled_vs_validation_consistency()
    test_compiled_evaluation_no_wrong_features()
    test_metric_calculation_consistency()
    test_compiled_evaluation_error_handling()
    test_end_to_end_consistency()
    print("\n✅ All compiled evaluation fix tests passed!")
