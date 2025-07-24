#!/usr/bin/env python3
"""
Test script to validate EtinyNet serialization pipeline.
"""

import tempfile
from pathlib import Path

import torch

from model import EtinyNet
from serialize import serialize_etinynet_model


def test_etinynet_serialization():
    """Test complete EtinyNet serialization pipeline."""
    print("🧪 Testing EtinyNet Serialization Pipeline")
    print("=" * 50)

    # Create a small EtinyNet model for testing
    torch.manual_seed(42)
    model = EtinyNet(
        variant="0.75",  # Smaller variant for faster testing
        num_classes=10,  # CIFAR-10 style
        input_size=32,  # 32x32 images
        use_asq=False,  # Disable ASQ for deterministic testing
    )
    model.eval()

    print(f"✓ Created EtinyNet-{model.variant} model")
    print(f"  - Parameters: {model.count_parameters():,}")
    print(f"  - Input size: {model.input_size}x{model.input_size}")
    print(f"  - Classes: {model.num_classes}")

    # Test PyTorch inference
    test_input = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        pytorch_output = model(test_input)

    print(f"✓ PyTorch inference successful")
    print(f"  - Output shape: {pytorch_output.shape}")
    print(
        f"  - Output range: [{pytorch_output.min().item():.3f}, {pytorch_output.max().item():.3f}]"
    )

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix=".etiny", delete=False) as f:
        model_path = Path(f.name)

    try:
        print(f"🔄 Serializing model to {model_path}")
        serialize_etinynet_model(model, model_path)

        # Verify file was created and has reasonable size
        assert model_path.exists(), "Serialization should create file"
        file_size = model_path.stat().st_size
        assert file_size > 1000, f"File size {file_size} seems too small"

        print(f"✓ Serialization successful")
        print(f"  - File size: {file_size:,} bytes")

        # Verify file header
        with open(model_path, "rb") as f:
            magic = f.read(4)
            assert magic == b"ETNY", f"Expected ETNY magic, got {magic}"

        print(f"✓ File header validation successful")

        # Read some basic metadata from file
        with open(model_path, "rb") as f:
            f.read(4)  # Skip magic
            version = int.from_bytes(f.read(4), byteorder="little")
            variant_len = int.from_bytes(f.read(4), byteorder="little")
            variant_bytes = f.read(variant_len)
            variant = variant_bytes.decode("utf-8")
            num_classes = int.from_bytes(f.read(4), byteorder="little")
            input_size = int.from_bytes(f.read(4), byteorder="little")

        print(f"✓ Metadata validation successful")
        print(f"  - Version: {version}")
        print(f"  - Variant: {variant}")
        print(f"  - Classes: {num_classes}")
        print(f"  - Input size: {input_size}")

        assert variant == model.variant
        assert num_classes == model.num_classes
        assert input_size == model.input_size

    finally:
        if model_path.exists():
            model_path.unlink()

    print("\n🎉 All EtinyNet serialization tests PASSED!")
    print("✓ Model creation and inference work correctly")
    print("✓ Serialization produces valid .etiny files")
    print("✓ File format validation passes")
    print("✓ Metadata is preserved correctly")


if __name__ == "__main__":
    test_etinynet_serialization()
