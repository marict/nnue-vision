#!/usr/bin/env python3
"""
Test to verify the NNUE inference fix works correctly.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, ".")


def test_nnue_inference_fix():
    """Test that NNUE inference now outputs only the float result."""

    print("=== Testing NNUE Inference Fix ===")

    # Create a simple test model file
    with tempfile.NamedTemporaryFile(suffix=".nnue", delete=False) as f:
        model_path = f.name
        # Write some dummy model data
        f.write(b"dummy_model_data")

    # Create a simple test image file
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        image_path = f.name
        # Write some dummy image data (32x32x3 float32)
        import numpy as np

        dummy_image = np.random.randn(32, 32, 3).astype(np.float32)
        f.write(dummy_image.tobytes())

    try:
        # Test the nnue_inference executable
        cpp_executable = Path("engine/build/nnue_inference")

        if not cpp_executable.exists():
            print("❌ nnue_inference executable not found")
            return False

        # Run the executable
        result = subprocess.run(
            [
                str(cpp_executable),
                model_path,
                image_path,
                "32",  # H
                "32",  # W
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        print(f"Return code: {result.returncode}")
        print(f"Stdout: {repr(result.stdout)}")
        print(f"Stderr: {repr(result.stderr)}")

        if result.returncode == 0:
            # Check that output is a single float
            lines = result.stdout.strip().split("\n")
            if lines:
                try:
                    float_value = float(lines[0])
                    print(f"✅ Successfully parsed float: {float_value}")

                    # Check that there's no debug output
                    if len(lines) == 1:
                        print("✅ No debug output found - clean interface")
                        return True
                    else:
                        print(f"❌ Found {len(lines)} lines of output, expected 1")
                        print(f"   Lines: {lines}")
                        return False
                except ValueError as e:
                    print(f"❌ Failed to parse float: {e}")
                    print(f"   Output: {lines[0]}")
                    return False
            else:
                print("❌ No output lines")
                return False
        else:
            print(f"❌ Executable failed with return code {result.returncode}")
            return False

    finally:
        # Clean up
        Path(model_path).unlink(missing_ok=True)
        Path(image_path).unlink(missing_ok=True)


if __name__ == "__main__":
    success = test_nnue_inference_fix()
    sys.exit(0 if success else 1)
