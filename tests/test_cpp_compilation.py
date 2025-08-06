"""Test C++ compilation with strict flags to catch issues before RunPod deployment."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_strict_cpp_compilation():
    """Test that C++ engine compiles with strict flags (mimicking RunPod environment)."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Run the strict compilation script
    script_path = project_root / "test_strict_compilation.sh"

    if not script_path.exists():
        pytest.skip("Strict compilation script not found")

    try:
        # Run the compilation script
        result = subprocess.run(
            [str(script_path)],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=60,  # 60 second timeout
        )

        # Check if compilation succeeded
        if result.returncode != 0:
            print("=== C++ Compilation Output ===")
            print(result.stdout)
            print("=== C++ Compilation Errors ===")
            print(result.stderr)
            raise AssertionError(
                f"C++ compilation failed with return code {result.returncode}"
            )

        # Print success output for debugging
        print("=== C++ Compilation Success ===")
        print(result.stdout)

    except subprocess.TimeoutExpired:
        raise AssertionError("C++ compilation timed out after 60 seconds")
    except FileNotFoundError:
        pytest.skip("Strict compilation script not found or not executable")


if __name__ == "__main__":
    test_strict_cpp_compilation()
