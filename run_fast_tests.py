#!/usr/bin/env python3
"""
Fast test runner that completes in under 10 seconds.
This runs the most critical tests without the slow dataset loading tests.
"""

import subprocess
import sys
from pathlib import Path


def run_fast_tests():
    """Run fast tests that complete in under 10 seconds."""

    # Truly minimal set of fastest tests
    fast_test_patterns = [
        "tests/test_model.py::TestGridFeatureSet",
        "tests/test_metric_calculation.py",
        "tests/test_weight_decay.py",
    ]

    # Build the pytest command
    cmd = [
        "python3",
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "--maxfail=3",
        "--timeout=5",
        "--disable-warnings",
        "-q",  # Quiet mode for faster output
    ] + fast_test_patterns

    print("🚀 Running fast tests (target: <10 seconds)...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, capture_output=False, timeout=10)
        if result.returncode == 0:
            print("\n✅ Fast tests completed successfully!")
            return True
        else:
            print(f"\n❌ Fast tests failed with exit code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("\n⏰ Fast tests timed out (>10 seconds)")
        return False
    except Exception as e:
        print(f"\n💥 Error running fast tests: {e}")
        return False


if __name__ == "__main__":
    success = run_fast_tests()
    sys.exit(0 if success else 1)
