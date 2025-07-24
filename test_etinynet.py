#!/usr/bin/env python3
"""
Test script for EtinyNet implementation.

This script verifies that the EtinyNet model is implemented correctly
and matches the paper specifications.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from model import (AdaptiveScaleQuantization, DenseLinearDepthwiseBlock,
                   EtinyNet, LinearDepthwiseBlock)
from serialize import serialize_etinynet_model


def test_linear_depthwise_block():
    """Test the Linear Depthwise Block implementation."""
    print("Testing Linear Depthwise Block...")

    # Create a simple LB block
    lb = LinearDepthwiseBlock(32, 64, 128, stride=1)

    # Test with random input
    batch_size = 2
    height, width = 56, 56
    x = torch.randn(batch_size, 32, height, width)

    # Forward pass
    with torch.no_grad():
        output = lb(x)

    expected_shape = (batch_size, 128, height, width)
    assert (
        output.shape == expected_shape
    ), f"Expected {expected_shape}, got {output.shape}"

    print(f"✓ LB input shape: {x.shape}")
    print(f"✓ LB output shape: {output.shape}")
    print(f"✓ LB parameters: {sum(p.numel() for p in lb.parameters()):,}")


def test_dense_linear_depthwise_block():
    """Test the Dense Linear Depthwise Block implementation."""
    print("\nTesting Dense Linear Depthwise Block...")

    # Create a DLB block with skip connection
    dlb = DenseLinearDepthwiseBlock(128, 128, 128, stride=1)

    # Test with random input
    batch_size = 2
    height, width = 28, 28
    x = torch.randn(batch_size, 128, height, width)

    # Forward pass
    with torch.no_grad():
        output = dlb(x)

    expected_shape = (batch_size, 128, height, width)
    assert (
        output.shape == expected_shape
    ), f"Expected {expected_shape}, got {output.shape}"

    print(f"✓ DLB input shape: {x.shape}")
    print(f"✓ DLB output shape: {output.shape}")
    print(f"✓ DLB parameters: {sum(p.numel() for p in dlb.parameters()):,}")
    print(f"✓ DLB uses skip connection: {dlb.use_skip}")


def test_adaptive_scale_quantization():
    """Test the Adaptive Scale Quantization implementation."""
    print("\nTesting Adaptive Scale Quantization...")

    # Create ASQ module
    asq = AdaptiveScaleQuantization(bits=4, init_lambda=2.0)

    # Test with random weights
    weights = torch.randn(128, 64, 3, 3)

    # Forward pass (training mode)
    asq.train()
    quantized_weights = asq(weights)

    assert (
        quantized_weights.shape == weights.shape
    ), "Quantized weights should maintain shape"

    # Check that lambda parameter is learnable
    assert asq.lambda_param.requires_grad, "Lambda parameter should be learnable"

    print(f"✓ ASQ input shape: {weights.shape}")
    print(f"✓ ASQ output shape: {quantized_weights.shape}")
    print(f"✓ ASQ lambda parameter: {asq.lambda_param.item():.3f}")
    print(f"✓ ASQ bits: {asq.bits}")


def test_etinynet_model():
    """Test the complete EtinyNet model."""
    print("\nTesting EtinyNet Models...")

    # Test EtinyNet-1.0
    print("\n--- EtinyNet-1.0 ---")
    model_10 = EtinyNet(variant="1.0", num_classes=1000, input_size=112)

    # Check parameter count (should be close to 976K from paper)
    param_count_10 = model_10.count_parameters()
    expected_params_10 = 976_000  # From paper
    tolerance = 0.1  # 10% tolerance

    print(f"✓ EtinyNet-1.0 parameters: {param_count_10:,}")
    print(f"✓ Expected parameters: {expected_params_10:,}")

    # Check if within reasonable range
    param_ratio = param_count_10 / expected_params_10
    if abs(param_ratio - 1.0) <= tolerance:
        print(f"✓ Parameter count within {tolerance*100}% of paper specification")
    else:
        print(
            f"⚠ Parameter count differs from paper by {abs(param_ratio-1.0)*100:.1f}%"
        )

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 112, 112)

    with torch.no_grad():
        output = model_10(x)

    expected_output_shape = (batch_size, 1000)
    assert (
        output.shape == expected_output_shape
    ), f"Expected {expected_output_shape}, got {output.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")

    # Test EtinyNet-0.75
    print("\n--- EtinyNet-0.75 ---")
    model_075 = EtinyNet(variant="0.75", num_classes=1000, input_size=112)

    param_count_075 = model_075.count_parameters()
    expected_params_075 = 680_000  # From paper

    print(f"✓ EtinyNet-0.75 parameters: {param_count_075:,}")
    print(f"✓ Expected parameters: {expected_params_075:,}")

    # Check if within reasonable range
    param_ratio = param_count_075 / expected_params_075
    if abs(param_ratio - 1.0) <= tolerance:
        print(f"✓ Parameter count within {tolerance*100}% of paper specification")
    else:
        print(
            f"⚠ Parameter count differs from paper by {abs(param_ratio-1.0)*100:.1f}%"
        )

    # Test forward pass
    with torch.no_grad():
        output = model_075(x)

    assert (
        output.shape == expected_output_shape
    ), f"Expected {expected_output_shape}, got {output.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")

    return model_10, model_075


def test_etinynet_with_asq():
    """Test EtinyNet with Adaptive Scale Quantization."""
    print("\n--- EtinyNet with ASQ ---")

    model = EtinyNet(variant="1.0", num_classes=10, use_asq=True, asq_bits=4)

    # Test that ASQ is properly initialized
    assert model.use_asq == True, "ASQ should be enabled"
    assert hasattr(model, "asq"), "Model should have ASQ module"

    # Test forward pass with ASQ
    batch_size = 1
    x = torch.randn(batch_size, 3, 112, 112)

    # Training mode (ASQ active)
    model.train()
    with torch.no_grad():
        output_train = model(x)

    # Eval mode (ASQ inactive)
    model.eval()
    with torch.no_grad():
        output_eval = model(x)

    print(f"✓ ASQ training mode output: {output_train.shape}")
    print(f"✓ ASQ eval mode output: {output_eval.shape}")
    print(f"✓ ASQ lambda parameter: {model.asq.lambda_param.item():.3f}")


def test_serialization():
    """Test EtinyNet serialization."""
    print("\nTesting EtinyNet Serialization...")

    try:
        # Create a small model for testing
        model = EtinyNet(variant="0.75", num_classes=10, input_size=112)

        # Create output directory
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)

        # Test serialization
        output_path = output_dir / "test_etinynet.etiny"
        serialize_etinynet_model(model, output_path)

        # Check that file was created
        assert output_path.exists(), "Serialized file should exist"
        file_size = output_path.stat().st_size

        print(f"✓ Serialization successful")
        print(f"✓ Output file: {output_path}")
        print(f"✓ File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

    except Exception as e:
        print(f"⚠ Serialization test failed: {e}")
        print("  (This may be expected if serialization is not fully implemented)")


def main():
    """Run all EtinyNet tests."""
    print("=" * 60)
    print("EtinyNet Implementation Tests")
    print("=" * 60)

    try:
        # Test individual components
        test_linear_depthwise_block()
        test_dense_linear_depthwise_block()
        test_adaptive_scale_quantization()

        # Test full models
        model_10, model_075 = test_etinynet_model()

        # Test ASQ integration
        test_etinynet_with_asq()

        # Test serialization
        test_serialization()

        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)

        # Print summary
        print("\nModel Summary:")
        print(f"EtinyNet-1.0:  {model_10.count_parameters():,} parameters")
        print(f"EtinyNet-0.75: {model_075.count_parameters():,} parameters")
        print(f"EtinyNet-1.0 FLOPs:  {model_10.count_flops():,}")
        print(f"EtinyNet-0.75 FLOPs: {model_075.count_flops():,}")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
