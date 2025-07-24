#!/usr/bin/env python3
"""
Test Runner for MCU Benchmarking System

This script runs comprehensive tests for the MCU benchmarking infrastructure
with appropriate timeouts and helpful output.

Usage:
    python scripts/run_tests.py              # Run all benchmark tests
    python scripts/run_tests.py --quick      # Run quick tests only
    python scripts/run_tests.py --unit       # Run unit tests only
    python scripts/run_tests.py --integration # Run integration tests only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", verbose=True, timeout=300):
    """Run benchmark tests with specified configuration."""

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add timeout
    cmd.extend([f"--timeout={timeout}"])

    # Add verbosity
    if verbose:
        cmd.append("-v")

    # Select test files based on type
    if test_type == "unit":
        cmd.append("tests/test_mcu_benchmarks.py")
    elif test_type == "integration":
        cmd.append("tests/test_benchmark_integration.py")
    elif test_type == "quick":
        # Run subset of fast tests
        cmd.extend(
            [
                "tests/test_mcu_benchmarks.py::TestParameterCounting",
                "tests/test_mcu_benchmarks.py::TestMACOperations",
                "tests/test_mcu_benchmarks.py::TestMemoryAnalysis",
            ]
        )
    else:  # all
        cmd.extend(
            ["tests/test_mcu_benchmarks.py", "tests/test_benchmark_integration.py"]
        )

    # Add markers for benchmark tests
    cmd.extend(["-m", "not slow"])  # Skip slow tests by default

    print(f"🧪 Running {test_type} benchmark tests...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    try:
        result = subprocess.run(cmd, timeout=timeout)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"❌ Tests timed out after {timeout} seconds")
        return 1
    except KeyboardInterrupt:
        print("⚠️  Tests interrupted by user")
        return 1


def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = ["torch", "numpy", "pytest", "pytest_timeout"]

    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print("❌ Missing required dependencies:")
        for module in missing:
            print(f"   - {module}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing)}")
        return False

    return True


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Run MCU benchmark tests")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick subset of tests"
    )
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Test timeout in seconds (default: 300)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Check dependencies first
    if not check_dependencies():
        return 1

    # Determine test type
    if args.quick:
        test_type = "quick"
    elif args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    else:
        test_type = "all"

    # Run tests
    return_code = run_tests(
        test_type=test_type, verbose=not args.quiet, timeout=args.timeout
    )

    if return_code == 0:
        print("\n✅ All benchmark tests passed!")
    else:
        print(f"\n❌ Tests failed with return code {return_code}")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
