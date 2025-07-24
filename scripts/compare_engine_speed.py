#!/usr/bin/env python3
"""Compare C++ engine runtime speed between NNUE and EtinyNet models.

This script automates the following steps:

1. Creates small NNUE & EtinyNet PyTorch models (sized for quick benchmarking).
2. Serialises them to binary formats (.nnue / .etiny) via serialize.py.
3. Builds the C++ engine (Release mode) as well as benchmark executables
   (benchmark_engine for NNUE and benchmark_etinynet_engine for EtinyNet).
4. Executes both benchmarks, parses their outputs and prints a speed comparison.

Run:
    python scripts/compare_engine_speed.py

Notes:
- The script assumes a Unix-like environment with CMake & a C++17 compiler.
- For repeatability it uses fixed random seeds.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import torch

from model import NNUE, EtinyNet, GridFeatureSet
from serialize import serialize_etinynet_model, serialize_model

# -------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------


def run(
    cmd: list[str] | str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Thin wrapper around subprocess.run printing the command."""
    if isinstance(cmd, list):
        printable = " ".join(cmd)
    else:
        printable = cmd
        cmd = [cmd]
    print(f"$ {printable}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def ensure_engine_built(build_dir: Path) -> None:
    """Configure & build the C++ engine in *Release* mode (only once)."""
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure step (run only if cache is absent)
    if not (build_dir / "CMakeCache.txt").exists():
        # Point to the engine directory which contains CMakeLists.txt
        engine_src_dir = build_dir.parent.parent  # engine/build/build_bench -> engine
        run(["cmake", "-DCMAKE_BUILD_TYPE=Release", str(engine_src_dir)], cwd=build_dir)

    # Build static library and NNUE benchmark executable
    run(
        ["cmake", "--build", ".", "--target", "nnue_engine", "benchmark_engine"],
        cwd=build_dir,
    )

    # Build EtinyNet benchmark executable (may have been added after initial CMake)
    etiny_exe = build_dir / "benchmark_etinynet_engine"
    if not etiny_exe.exists():
        # Compile the EtinyNet benchmark directly with the produced static library
        cpp_path = Path("engine/benchmark_etinynet_engine.cpp").resolve()
        lib_path = build_dir / "libnnue_engine.a"
        include_dir = Path("engine/include").resolve()

        if not lib_path.exists():
            raise FileNotFoundError("libnnue_engine.a not found – build failed?")

        compile_cmd = [
            "g++",
            "-std=c++17",
            "-O3",
            "-march=native",
            str(cpp_path),
            str(lib_path),
            "-I",
            str(include_dir),
            "-pthread",
            "-o",
            str(etiny_exe),
        ]
        run(compile_cmd)


# -------------------------------------------------------------
# Model creation & serialisation
# -------------------------------------------------------------


def create_models(tmp_dir: Path) -> Tuple[Path, Path]:
    """Instantiate tiny versions of NNUE & EtinyNet and serialise them."""
    torch.manual_seed(42)

    # NNUE – extremely small for speed demo
    feature_set = GridFeatureSet(grid_size=8, num_features_per_square=8)
    nnue = NNUE(
        feature_set=feature_set,
        l1_size=128,
        l2_size=8,
        l3_size=16,
        num_ls_buckets=2,
        visual_threshold=0.5,
    )
    nnue.eval()

    nnue_path = tmp_dir / "demo.nnue"
    serialize_model(nnue, nnue_path)

    # EtinyNet – CIFAR-10 sized input
    etiny = EtinyNet(variant="0.75", num_classes=10, input_size=32, use_asq=False)
    etiny.eval()

    etiny_path = tmp_dir / "demo.etiny"
    serialize_etinynet_model(etiny, etiny_path)

    return nnue_path, etiny_path


# -------------------------------------------------------------
# Benchmark execution helpers
# -------------------------------------------------------------


def parse_nnue_benchmark(output: str) -> float:
    """Extract average latency (ms) for *Dense (90%)* scenario from benchmark_engine output."""
    pattern = re.compile(r"Dense \(90%\).*?\s([0-9]+\.[0-9]+)")
    match = pattern.search(output)
    if match:
        return float(match.group(1))
    raise RuntimeError("Failed to parse NNUE benchmark output – pattern not found.")


def parse_etiny_benchmark(output: str) -> float:
    """Extract RESULT_AVG_MS value emitted by EtinyNet benchmark binary."""
    for line in output.splitlines():
        if line.startswith("RESULT_AVG_MS"):
            return float(line.split(":")[1].strip())
    raise RuntimeError("Failed to parse EtinyNet benchmark output – tag not found.")


# -------------------------------------------------------------
# Main routine
# -------------------------------------------------------------


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    engine_dir = repo_root / "engine"
    build_dir = engine_dir / "build" / "build_bench"

    # Step 1: Build engine & benchmarks
    print("🔧 Building C++ engine & benchmarks (Release mode)…")
    ensure_engine_built(build_dir)

    nnue_benchmark = build_dir / "benchmark_engine"
    etiny_benchmark = build_dir / "benchmark_etinynet_engine"

    if not nnue_benchmark.exists():
        sys.exit("benchmark_engine not found – build failed")
    if not etiny_benchmark.exists():
        sys.exit("benchmark_etinynet_engine not found – build failed")

    # Step 2: Create models & serialise
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        nnue_path, etiny_path = create_models(tmp_dir)
        try:
            print(f"📄 Serialised NNUE to {nnue_path.relative_to(repo_root)}")
            print(f"📄 Serialised EtinyNet to {etiny_path.relative_to(repo_root)}")
        except ValueError:
            # Temp files are outside repo; just print absolute paths
            print(f"📄 Serialised NNUE to {nnue_path}")
            print(f"📄 Serialised EtinyNet to {etiny_path}")

        # Step 3: Run benchmarks
        print("🚀 Running NNUE benchmark…")
        nnue_res = run([str(nnue_benchmark), str(nnue_path)])
        nnue_time = parse_nnue_benchmark(nnue_res.stdout)
        print(f"   NNUE Dense avg latency: {nnue_time:.4f} ms")

        print("🚀 Running EtinyNet benchmark…")
        etiny_res = run([str(etiny_benchmark), str(etiny_path)])
        etiny_time = parse_etiny_benchmark(etiny_res.stdout)
        print(f"   EtinyNet avg latency:    {etiny_time:.4f} ms")

    # Step 4: Print comparison
    speed_ratio = nnue_time / etiny_time if etiny_time else float("inf")
    faster = "NNUE" if nnue_time < etiny_time else "EtinyNet"

    print("\n📊 Speed comparison (lower is better):")
    print(f"   NNUE:      {nnue_time:.4f} ms")
    print(f"   EtinyNet:  {etiny_time:.4f} ms")
    print(f"🏆 {faster} is {speed_ratio:.2f}× faster on this benchmark.")


if __name__ == "__main__":
    main()
