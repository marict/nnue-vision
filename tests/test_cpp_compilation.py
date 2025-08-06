"""Test C++ compilation with strict flags to catch issues before RunPod deployment."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_strict_cpp_compilation():
    """Test that C++ engine compiles with strict flags (mimicking RunPod environment)."""
    project_root = Path(__file__).parent.parent
    engine_dir = project_root / "engine"

    if not engine_dir.exists():
        raise AssertionError("Engine directory not found")

    # Check if g++ is available in PATH or common locations
    gpp_paths = ["g++", "/opt/homebrew/bin/g++", "/usr/local/bin/g++", "/usr/bin/g++"]
    gpp_found = None

    for gpp_path in gpp_paths:
        try:
            result = subprocess.run(
                [gpp_path, "--version"], capture_output=True, check=True
            )
            gpp_found = gpp_path
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if not gpp_found:
        raise AssertionError(
            "g++ is not available! This is required for C++ compilation.\n"
            "Install g++:\n"
            "  macOS: brew install gcc\n"
            "  Ubuntu/Debian: sudo apt-get install build-essential\n"
            "  Or install Xcode Command Line Tools on macOS"
        )

    # Detect compiler and use appropriate flags
    try:
        result = subprocess.run(
            [gpp_found, "--version"], capture_output=True, text=True, check=True
        )
        if "GNU" in result.stdout:
            # GCC-specific flags (without sign conversion for now)
            cxxflags = [
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wpedantic",
                "-Wshadow",
                "-Wunused-parameter",
                "-Wtype-limits",
                "-Wuninitialized",
                "-Wmaybe-uninitialized",
                "-Wstrict-overflow=5",
                "-Warray-bounds",
                "-Wformat=2",
                "-Wformat-security",
                "-Wnull-dereference",
                "-Wduplicated-cond",
                "-Wduplicated-branches",
                "-Wlogical-op",
                "-Wrestrict",
                "-Waggressive-loop-optimizations",
                "-Wno-unknown-pragmas",
                "-fstack-protector-strong",
                "-D_FORTIFY_SOURCE=2",
            ]
        else:
            # Clang-specific flags (without sign conversion for now)
            cxxflags = [
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wpedantic",
                "-Wshadow",
                "-Wunused-parameter",
                "-Wtype-limits",
                "-Wuninitialized",
                "-Warray-bounds",
                "-Wformat=2",
                "-Wformat-security",
                "-Wnull-dereference",
                "-Wno-unknown-pragmas",
                "-fstack-protector-strong",
            ]
    except subprocess.CalledProcessError:
        raise AssertionError("Could not detect compiler type")

    includes = "-Iinclude"

    # Compile all source files with strict flags
    source_files = [
        "src/simd_scalar.cpp",
        "src/simd_neon.cpp",
        "src/simd_avx2.cpp",
        "src/nnue_engine.cpp",
        "tests/test_nnue_engine.cpp",
        "tests/test_etinynet_engine.cpp",
    ]

    cmd = [
        "bash",
        "-c",
        f"""
        cd {engine_dir} && 
        mkdir -p build && 
        cd build && 
        {gpp_found} {' '.join(cxxflags)} {includes} -c {' '.join([f'../{f}' for f in source_files])}
        """,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=90,  # 90 second timeout
        )

        if result.returncode != 0:
            print("=== Strict C++ Compilation Output ===")
            print(result.stdout)
            print("=== Strict C++ Compilation Errors ===")
            print(result.stderr)
            raise AssertionError(
                f"Strict C++ compilation failed with return code {result.returncode}"
            )

        print("=== Strict C++ Compilation Success ===")
        print(result.stdout)

    except subprocess.TimeoutExpired:
        raise AssertionError("Strict C++ compilation timed out after 90 seconds")
    except FileNotFoundError:
        raise AssertionError("g++ not available or not executable")


def test_runpod_like_compilation():
    """Test compilation with flags that match RunPod environment exactly."""
    project_root = Path(__file__).parent.parent
    engine_dir = project_root / "engine"

    if not engine_dir.exists():
        pytest.skip("Engine directory not found")

    # Check if cmake is available in PATH or common locations
    cmake_paths = ["cmake", "/opt/homebrew/bin/cmake", "/usr/local/bin/cmake"]
    cmake_found = None

    for cmake_path in cmake_paths:
        try:
            result = subprocess.run(
                [cmake_path, "--version"], capture_output=True, check=True
            )
            cmake_found = cmake_path
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if not cmake_found:
        raise AssertionError(
            "cmake is not available! This is required for C++ compilation.\n"
            "Install cmake:\n"
            "  macOS: brew install cmake\n"
            "  Ubuntu/Debian: sudo apt-get install cmake\n"
            "  Or download from: https://cmake.org/download/"
        )

    # RunPod-like compilation flags (matching container_setup.sh)
    cmd = [
        "bash",
        "-c",
        f"""
        cd {engine_dir} && 
        mkdir -p build && 
        cd build && 
        {cmake_found} .. -DCMAKE_BUILD_TYPE=Release && 
        make -j$(nproc)
        """,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=120,  # 2 minute timeout for full build
        )

        if result.returncode != 0:
            print("=== RunPod-like Compilation Output ===")
            print(result.stdout)
            print("=== RunPod-like Compilation Errors ===")
            print(result.stderr)
            raise AssertionError(
                f"RunPod-like compilation failed with return code {result.returncode}"
            )

        print("=== RunPod-like Compilation Success ===")
        print(result.stdout)

    except subprocess.TimeoutExpired:
        raise AssertionError("RunPod-like compilation timed out after 120 seconds")
    except FileNotFoundError:
        pytest.skip("cmake or make not available")


def test_manual_strict_compilation():
    """Test manual compilation with maximum strictness."""
    project_root = Path(__file__).parent.parent
    engine_dir = project_root / "engine"

    if not engine_dir.exists():
        pytest.skip("Engine directory not found")

    # Check if g++ is available in PATH or common locations
    gpp_paths = ["g++", "/opt/homebrew/bin/g++", "/usr/local/bin/g++", "/usr/bin/g++"]
    gpp_found = None

    for gpp_path in gpp_paths:
        try:
            result = subprocess.run(
                [gpp_path, "--version"], capture_output=True, check=True
            )
            gpp_found = gpp_path
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if not gpp_found:
        raise AssertionError(
            "g++ is not available! This is required for C++ compilation.\n"
            "Install g++:\n"
            "  macOS: brew install gcc\n"
            "  Ubuntu/Debian: sudo apt-get install build-essential\n"
            "  Or install Xcode Command Line Tools on macOS"
        )

    # Detect compiler and use appropriate flags
    try:
        result = subprocess.run(
            [gpp_found, "--version"], capture_output=True, text=True, check=True
        )
        if "GNU" in result.stdout:
            # GCC-specific flags
            cxxflags = [
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wpedantic",
                "-Wshadow",
                "-Wunused-parameter",
                "-Wtype-limits",
                "-Wuninitialized",
                "-Wmaybe-uninitialized",
                "-Wstrict-overflow=5",
                "-Warray-bounds",
                "-Wformat=2",
                "-Wformat-security",
                "-Wnull-dereference",
                "-Wduplicated-cond",
                "-Wduplicated-branches",
                "-Wlogical-op",
                "-Wrestrict",
                "-Waggressive-loop-optimizations",
                "-Wno-unknown-pragmas",
                "-fstack-protector-strong",
                "-D_FORTIFY_SOURCE=2",
            ]
        else:
            # Clang-specific flags (more conservative)
            cxxflags = [
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wpedantic",
                "-Wshadow",
                "-Wunused-parameter",
                "-Wtype-limits",
                "-Wuninitialized",
                "-Warray-bounds",
                "-Wformat=2",
                "-Wformat-security",
                "-Wnull-dereference",
                "-Wno-unknown-pragmas",
                "-fstack-protector-strong",
            ]
    except subprocess.CalledProcessError:
        pytest.skip("Could not detect compiler type")

    includes = "-Iinclude"

    # Compile all source files individually
    source_files = [
        "src/simd_scalar.cpp",
        "src/simd_neon.cpp",
        "src/simd_avx2.cpp",
        "src/nnue_engine.cpp",
        "tests/test_nnue_engine.cpp",
        "tests/test_etinynet_engine.cpp",
        "regression_test.cpp",
        "etinynet_inference.cpp",
        "benchmark_engine.cpp",
        "benchmark_etinynet_engine.cpp",
    ]

    cmd = [
        "bash",
        "-c",
        f"""
        cd {engine_dir} && 
        mkdir -p build && 
        cd build && 
        {gpp_found} {' '.join(cxxflags)} {includes} -c {' '.join([f'../{f}' for f in source_files])}
        """,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=90,  # 90 second timeout
        )

        if result.returncode != 0:
            print("=== Manual Strict Compilation Output ===")
            print(result.stdout)
            print("=== Manual Strict Compilation Errors ===")
            print(result.stderr)
            raise AssertionError(
                f"Manual strict compilation failed with return code {result.returncode}"
            )

        print("=== Manual Strict Compilation Success ===")
        print(result.stdout)

    except subprocess.TimeoutExpired:
        raise AssertionError("Manual strict compilation timed out after 90 seconds")
    except FileNotFoundError:
        pytest.skip("g++ not available")


if __name__ == "__main__":
    test_strict_cpp_compilation()
