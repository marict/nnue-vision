import os
import sys
import tempfile

import pytest

sys.path.insert(0, ".")
from nnue_runpod_service import _extract_project_name_from_config


def test_runpod_service_function_signature_fix():
    """Test that the RunPod service function signature fix works correctly."""

    # Create a temporary config file with project_name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("project_name = 'test-project-name'\n")
        config_path = f.name

    try:
        # Test that the function works with the correct signature (only config_path)
        project_name = _extract_project_name_from_config(config_path)
        assert project_name == "test-project-name"
        print("✅ Function signature fix works correctly!")

    finally:
        # Clean up
        os.unlink(config_path)


def test_runpod_service_imports_without_error():
    """Test that the RunPod service can be imported without TypeError."""
    try:
        print("✅ RunPod service imports successfully without TypeError")
    except TypeError as e:
        pytest.fail(f"RunPod service import failed with TypeError: {e}")


if __name__ == "__main__":
    test_runpod_service_function_signature_fix()
    test_runpod_service_imports_without_error()
    print("✅ All RunPod service integration tests passed!")
