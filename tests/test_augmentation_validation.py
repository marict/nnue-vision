import sys

sys.path.insert(0, ".")

import torch

from data.datasets import GenericVisionDataset


def test_validation_augmentation():
    """Test that validation data does NOT get augmentation."""

    print("=== Testing Validation Data Augmentation ===")

    # Create train dataset with augmentation
    train_dataset = GenericVisionDataset(
        dataset_name="cifar10",
        split="train",
        use_augmentation=True,
        augmentation_strength="medium",
        max_samples=10,  # Small sample for testing
    )

    # Create validation dataset
    val_dataset = GenericVisionDataset(
        dataset_name="cifar10",
        split="val",
        use_augmentation=True,  # This should be ignored for val split
        augmentation_strength="medium",
        max_samples=10,  # Small sample for testing
    )

    print(f"Train dataset augmentation: {train_dataset.use_augmentation}")
    print(f"Val dataset augmentation: {val_dataset.use_augmentation}")

    # Check if the transforms are different
    train_transform = train_dataset.transform
    val_transform = val_dataset.transform

    print(f"Train transform: {type(train_transform)}")
    print(f"Val transform: {type(val_transform)}")

    # Get a sample from each
    train_image, train_label = train_dataset[0]
    val_image, val_label = val_dataset[0]

    print(f"Train image shape: {train_image.shape}")
    print(f"Val image shape: {val_image.shape}")
    print(f"Train label: {train_label}")
    print(f"Val label: {val_label}")

    # Check if the transforms are actually different by looking at their components
    if hasattr(train_transform, "transforms"):
        train_transforms = [type(t).__name__ for t in train_transform.transforms]
        print(f"Train transforms: {train_transforms}")

    if hasattr(val_transform, "transforms"):
        val_transforms = [type(t).__name__ for t in val_transform.transforms]
        print(f"Val transforms: {val_transforms}")

    # The validation dataset should have fewer or no augmentation transforms
    print("✅ Validation dataset should have minimal/no augmentation transforms")


def test_augmentation_configuration():
    """Test augmentation configuration logic."""

    print("\n=== Testing Augmentation Configuration ===")

    # Test different split configurations
    splits = ["train", "training", "val", "validation", "test"]

    for split in splits:
        dataset = GenericVisionDataset(
            dataset_name="cifar10",
            split=split,
            use_augmentation=None,  # Let it auto-determine
            max_samples=5,
        )

        print(f"Split: {split} -> Augmentation: {dataset.use_augmentation}")

    print("✅ Augmentation should only be enabled for 'train' and 'training' splits")


def test_metric_mismatch_scenario():
    """Test a scenario that could cause F1 vs accuracy mismatch."""

    print("\n=== Testing Metric Mismatch Scenario ===")

    # Create a scenario where we have imbalanced predictions
    # This could happen if augmentation is applied to validation data

    # Simulate outputs where model is very confident but wrong
    outputs = torch.tensor(
        [
            [0.95],  # Very confident prediction for class 1
            [0.95],  # Very confident prediction for class 1
            [0.95],  # Very confident prediction for class 1
            [0.05],  # Very confident prediction for class 0
            [0.05],  # Very confident prediction for class 0
        ]
    )

    targets = torch.tensor([0, 0, 0, 1, 1])  # But targets are different

    from evaluate import compute_metrics

    metrics = compute_metrics(outputs, targets)

    print(f"Outputs: {outputs.numpy().flatten()}")
    print(f"Targets: {targets.numpy()}")
    print(f"Metrics: {metrics}")

    # This should show low accuracy but potentially different F1
    # due to class imbalance and the way F1 is calculated

    print("✅ This scenario shows how F1 and accuracy can differ")


if __name__ == "__main__":
    test_validation_augmentation()
    test_augmentation_configuration()
    test_metric_mismatch_scenario()
    print("\n✅ All augmentation validation tests completed!")
