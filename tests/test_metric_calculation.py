import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, ".")
from evaluate import compute_metrics


def test_metric_calculation_debug():
    """Debug the metric calculation issue where F1 and accuracy are mismatched."""

    # Test case 1: Simple binary classification
    print("=== Test Case 1: Simple Binary Classification ===")
    outputs = torch.tensor([[0.8], [0.2], [0.9], [0.1]])  # 4 samples, 1 class
    targets = torch.tensor([1, 0, 1, 0])  # Binary labels

    metrics = compute_metrics(outputs, targets)
    print(f"Outputs: {outputs.numpy().flatten()}")
    print(f"Targets: {targets.numpy()}")
    print(f"Metrics: {metrics}")

    # Manual calculation for verification
    predictions = (outputs.numpy().flatten() > 0.5).astype(int)
    targets_binary = targets.numpy().astype(int)

    manual_acc = accuracy_score(targets_binary, predictions)
    manual_f1 = f1_score(
        targets_binary, predictions, average="weighted", zero_division=0
    )

    print(f"Manual accuracy: {manual_acc}")
    print(f"Manual F1: {manual_f1}")
    print(f"Predictions: {predictions}")
    print(f"Targets binary: {targets_binary}")
    print()

    # Test case 2: Multi-class classification
    print("=== Test Case 2: Multi-class Classification ===")
    outputs = torch.tensor(
        [
            [0.8, 0.1, 0.1],  # Class 0
            [0.1, 0.8, 0.1],  # Class 1
            [0.1, 0.1, 0.8],  # Class 2
            [0.8, 0.1, 0.1],  # Class 0
        ]
    )
    targets = torch.tensor([0, 1, 2, 0])

    metrics = compute_metrics(outputs, targets)
    print(f"Outputs shape: {outputs.shape}")
    print(f"Targets: {targets.numpy()}")
    print(f"Metrics: {metrics}")

    # Manual calculation for verification
    predictions = outputs.numpy().argmax(axis=1)
    targets_binary = targets.numpy().astype(int)

    manual_acc = accuracy_score(targets_binary, predictions)
    manual_f1 = f1_score(
        targets_binary, predictions, average="weighted", zero_division=0
    )

    print(f"Manual accuracy: {manual_acc}")
    print(f"Manual F1: {manual_f1}")
    print(f"Predictions: {predictions}")
    print(f"Targets binary: {targets_binary}")
    print()

    # Test case 3: Edge case - all same class
    print("=== Test Case 3: All Same Class ===")
    outputs = torch.tensor([[0.9], [0.8], [0.7], [0.6]])  # All predicting class 1
    targets = torch.tensor([1, 1, 1, 1])  # All actually class 1

    metrics = compute_metrics(outputs, targets)
    print(f"Outputs: {outputs.numpy().flatten()}")
    print(f"Targets: {targets.numpy()}")
    print(f"Metrics: {metrics}")

    # Manual calculation for verification
    predictions = (outputs.numpy().flatten() > 0.5).astype(int)
    targets_binary = targets.numpy().astype(int)

    manual_acc = accuracy_score(targets_binary, predictions)
    manual_f1 = f1_score(
        targets_binary, predictions, average="weighted", zero_division=0
    )

    print(f"Manual accuracy: {manual_acc}")
    print(f"Manual F1: {manual_f1}")
    print(f"Predictions: {predictions}")
    print(f"Targets binary: {targets_binary}")
    print()

    # Test case 4: Edge case - all wrong
    print("=== Test Case 4: All Wrong ===")
    outputs = torch.tensor([[0.9], [0.8], [0.7], [0.6]])  # All predicting class 1
    targets = torch.tensor([0, 0, 0, 0])  # All actually class 0

    metrics = compute_metrics(outputs, targets)
    print(f"Outputs: {outputs.numpy().flatten()}")
    print(f"Targets: {targets.numpy()}")
    print(f"Metrics: {metrics}")

    # Manual calculation for verification
    predictions = (outputs.numpy().flatten() > 0.5).astype(int)
    targets_binary = targets.numpy().astype(int)

    manual_acc = accuracy_score(targets_binary, predictions)
    manual_f1 = f1_score(
        targets_binary, predictions, average="weighted", zero_division=0
    )

    print(f"Manual accuracy: {manual_acc}")
    print(f"Manual F1: {manual_f1}")
    print(f"Predictions: {predictions}")
    print(f"Targets binary: {targets_binary}")
    print()


def test_metric_calculation_consistency():
    """Test that F1 and accuracy are consistent."""

    # Create a realistic scenario that might cause the issue
    print("=== Testing Metric Consistency ===")

    # Simulate a scenario where we have imbalanced classes
    # 90% class 0, 10% class 1
    n_samples = 1000
    n_class_0 = 900
    n_class_1 = 100

    # Create targets
    targets = torch.cat([torch.zeros(n_class_0), torch.ones(n_class_1)])

    # Create outputs where model is good at class 0 but poor at class 1
    outputs_class_0 = torch.rand(n_class_0) * 0.3  # Mostly < 0.5 (correct)
    outputs_class_1 = torch.rand(n_class_1) * 0.3 + 0.7  # Mostly > 0.5 (correct)

    outputs = torch.cat([outputs_class_0, outputs_class_1]).unsqueeze(1)

    metrics = compute_metrics(outputs, targets)
    print(f"Class distribution: {n_class_0} class 0, {n_class_1} class 1")
    print(f"Metrics: {metrics}")

    # Manual calculation
    predictions = (outputs.numpy().flatten() > 0.5).astype(int)
    targets_binary = targets.numpy().astype(int)

    manual_acc = accuracy_score(targets_binary, predictions)
    manual_f1 = f1_score(
        targets_binary, predictions, average="weighted", zero_division=0
    )

    print(f"Manual accuracy: {manual_acc}")
    print(f"Manual F1: {manual_f1}")
    print(f"Predictions distribution: {np.bincount(predictions)}")
    print(f"Targets distribution: {np.bincount(targets_binary)}")

    # Check if there's a significant difference
    acc_diff = abs(metrics["acc"] - manual_acc)
    f1_diff = abs(metrics["f1"] - manual_f1)

    print(f"Accuracy difference: {acc_diff}")
    print(f"F1 difference: {f1_diff}")

    assert acc_diff < 1e-6, f"Accuracy calculation mismatch: {acc_diff}"
    assert f1_diff < 1e-6, f"F1 calculation mismatch: {f1_diff}"

    print("✅ Metric calculations are consistent!")


if __name__ == "__main__":
    test_metric_calculation_debug()
    test_metric_calculation_consistency()
    print("✅ All metric calculation tests passed!")
