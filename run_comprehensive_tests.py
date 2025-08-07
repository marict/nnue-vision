#!/usr/bin/env python3
"""
Comprehensive test runner that includes actual training tests and other essential tests.
This runs a broader set of tests including the two actual training tests.
"""

import subprocess
import sys
from pathlib import Path


def run_comprehensive_tests():
    """Run comprehensive tests including actual training tests."""

    # Comprehensive set of tests including actual training
    comprehensive_test_patterns = [
        "tests/test_model.py",
        "tests/test_metric_calculation.py",
        "tests/test_checkpoint_saving_fix.py",
        "tests/test_augmentation_validation.py",
        "tests/test_weight_decay.py",
        "tests/test_benchmark_integration.py",
        "tests/test_mcu_benchmarks.py",
        "tests/test_docker_args_graphql_safe.py",
        "tests/test_runpod_service.py",
        "tests/test_runpod_integration.py",
        "tests/test_wandb_checkpoints.py",
        "tests/test_actual_training.py",  # The two actual training tests
    ]

    # Build the pytest command
    cmd = [
        "python3",
        "-m",
        "pytest",
        "-v",
        "--tb=short",
        "--maxfail=5",
        "--timeout=60",  # Longer timeout for training tests
        "--disable-warnings",
    ] + comprehensive_test_patterns

    print("🚀 Running comprehensive tests (includes actual training)...")
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd, capture_output=False, timeout=120
        )  # 2 minutes for training tests
        if result.returncode == 0:
            print("\n✅ Comprehensive tests completed successfully!")
            return True
        else:
            print(f"\n❌ Comprehensive tests failed with exit code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print("\n⏰ Comprehensive tests timed out (>120 seconds)")
        return False
    except Exception as e:
        print(f"\n💥 Error running comprehensive tests: {e}")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
