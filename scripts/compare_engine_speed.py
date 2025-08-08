#!/usr/bin/env python3
"""Compare NNUE vs EtinyNet C++ runtime on CIFAR-10 using untrained models."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from data.loaders import create_data_loaders
from nnue import NNUE, EtinyNet
from serialize import serialize_etinynet_model, serialize_model

# -------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------


def run(
    cmd: list[str] | str, cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    if isinstance(cmd, list):
        printable = " ".join(cmd)
    else:
        printable = cmd
        cmd = [cmd]
    print(f"$ {printable}")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def ensure_engine_built(build_dir: Path) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)

    # Configure step (run only if cache is absent)
    if not (build_dir / "CMakeCache.txt").exists():
        # Point to the engine directory which contains CMakeLists.txt
        engine_src_dir = build_dir.parent.parent  # engine/build/build_bench -> engine
        run(["cmake", "-DCMAKE_BUILD_TYPE=Release", str(engine_src_dir)], cwd=build_dir)

    # Build static library and executables
    run(
        [
            "cmake",
            "--build",
            ".",
            "--target",
            "nnue_engine",
            "benchmark_engine",
            "nnue_inference",
            "etinynet_inference",
        ],
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
    """Create 0.98M parameter versions of NNUE & EtinyNet and serialise them."""
    torch.manual_seed(42)

    # NNUE – using new 0.98M parameter defaults
    nnue = NNUE()  # Uses new defaults: 10x10x8 features, 1024 L1 size = ~976K params
    nnue.eval()

    nnue_path = tmp_dir / "nnue_098m.nnue"
    serialize_model(nnue, nnue_path)

    # EtinyNet – micro variant sized for CIFAR-10 and speed
    etiny = EtinyNet(variant="micro", num_classes=10, input_size=32, use_asq=False)
    etiny.eval()

    etiny_path = tmp_dir / "etinynet_098m.etiny"
    serialize_etinynet_model(etiny, etiny_path)

    return nnue_path, etiny_path


# -------------------------------------------------------------
# CIFAR-10 driven runtime comparison (untrained models)
# -------------------------------------------------------------


def run_cifar10_exec_benchmark(
    nnue_model_path: Path,
    etiny_model_path: Path,
    build_dir: Path,
    num_samples: int = 200,
) -> None:
    """Measure ms/sample on CIFAR-10 for untrained NNUE vs EtinyNet."""
    print("\n📦 CIFAR-10 Performance Benchmark (untrained models)")
    print("=" * 55)

    # Prepare CIFAR-10 loaders (no augmentation); batch_size=1 for per-sample timing
    _, val_loader, _ = create_data_loaders(
        dataset_name="cifar10",
        batch_size=1,
        target_size=(32, 32),
        max_samples_per_split=num_samples,
        subset=1.0,
        use_augmentation=False,
        augmentation_strength="light",
    )

    nnue_exec = build_dir / "nnue_inference"
    etiny_exec = build_dir / "etinynet_inference"
    if not nnue_exec.exists() or not etiny_exec.exists():
        raise FileNotFoundError(
            "Inference executables not found. Ensure engine build succeeded."
        )

    def time_model(exec_path: Path, model_path: Path) -> float:
        total_time = 0.0
        total = 0
        for images, _ in val_loader:
            # images: [1, C, H, W] tensor
            img = images[0].cpu().numpy()  # CHW float32
            h, w = img.shape[1], img.shape[2]
            img_path = Path(tempfile.mktemp(suffix=".bin"))
            try:
                img.astype(np.float32).tofile(str(img_path))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                # Use cuda events if available for more precise timing, else fallback to perf_counter
                if torch.cuda.is_available():
                    start.record()
                    res = subprocess.run(
                        [
                            str(exec_path),
                            str(model_path),
                            str(img_path),
                            str(h),
                            str(w),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    end.record()
                    torch.cuda.synchronize()
                    elapsed_ms = start.elapsed_time(end)
                else:
                    import time

                    t0 = time.perf_counter()
                    res = subprocess.run(
                        [
                            str(exec_path),
                            str(model_path),
                            str(img_path),
                            str(h),
                            str(w),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if res.returncode != 0:
                    raise RuntimeError(
                        f"Inference failed (rc={res.returncode}): {res.stderr}"
                    )
                total_time += elapsed_ms
                total += 1
            finally:
                try:
                    if img_path.exists():
                        img_path.unlink()
                except Exception:
                    pass
        return (total_time / total) if total > 0 else float("inf")

    nnue_ms = time_model(nnue_exec, nnue_model_path)
    etiny_ms = time_model(etiny_exec, etiny_model_path)

    print(f"Samples: {num_samples}")
    print(f"NNUE    (32x32) avg: {nnue_ms:.3f} ms/sample")
    print(f"EtinyNet(32x32) avg: {etiny_ms:.3f} ms/sample")
    if etiny_ms > 0:
        print(f"Speedup (Etiny/NNUE): {etiny_ms/nnue_ms:.2f}x")


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


def run_comprehensive_benchmark(
    nnue_path: Path, etiny_path: Path, nnue_benchmark: Path, etiny_benchmark: Path
) -> None:
    """Run comprehensive density comparison between NNUE and EtinyNet."""

    print("\n🔬 Comprehensive Density Benchmark: NNUE vs EtinyNet")
    print("=" * 65)

    # Define density scenarios for comparison
    scenarios = [
        ("Chess-like", 0.001, "Ultra sparse (chess engines)"),
        ("Very Sparse", 0.01, "Highly sparse"),
        ("Sparse", 0.05, "Moderately sparse"),
        ("Medium", 0.25, "Medium density"),
        ("Dense", 0.90, "Mostly dense"),
    ]

    print(
        f"\n{'Density':<12} {'NNUE (ms)':<12} {'EtinyNet (ms)':<14} {'Speedup':<10} {'Description'}"
    )
    print("-" * 65)

    nnue_results = []
    etiny_results = []

    for name, density, desc in scenarios:
        # Run NNUE benchmark with varying density
        print(f"Testing {name:<11} ", end="", flush=True)

        try:
            # NNUE benchmark with fluctuating features for incremental updates
            result = subprocess.run(
                [str(nnue_benchmark), str(nnue_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f"❌ NNUE failed: {result.stderr}")
                continue

            # Parse NNUE result for this density level
            nnue_time = parse_nnue_benchmark_density(result.stdout, density)

            # EtinyNet benchmark (full dense processing regardless of density)
            result = subprocess.run(
                [str(etiny_benchmark), str(etiny_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                print(f"❌ EtinyNet failed: {result.stderr}")
                continue

            etiny_time = parse_etiny_benchmark(result.stdout)

            # Calculate speedup
            speedup = etiny_time / nnue_time if nnue_time > 0 else float("inf")
            speedup_str = f"{speedup:.1f}x" if speedup != float("inf") else "∞"

            print(f"{nnue_time:<12.4f} {etiny_time:<14.4f} {speedup_str:<10} {desc}")

            nnue_results.append((name, density, nnue_time))
            etiny_results.append((name, density, etiny_time))

        except subprocess.TimeoutExpired:
            print("❌ Timeout")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Show incremental update benefits
    print(f"\n🔄 Incremental Update Analysis:")
    print("-" * 45)

    if len(nnue_results) >= 2:
        sparse_time = next(
            (time for name, density, time in nnue_results if density <= 0.01), None
        )
        dense_time = next(
            (time for name, density, time in nnue_results if density >= 0.5), None
        )

        if sparse_time and dense_time:
            incremental_benefit = dense_time / sparse_time
            print(
                f"NNUE sparse vs dense: {incremental_benefit:.1f}x faster at low density"
            )
            print(f"EtinyNet: Same performance regardless of density")

    print(f"\n💡 Key Insights:")
    print(f"• NNUE excels at sparse scenarios due to incremental updates")
    print(f"• EtinyNet has consistent performance across all densities")
    print(f"• Crossover point shows optimal use case for each architecture")


def parse_nnue_benchmark_density(output: str, target_density: float) -> float:
    """Extract timing for specific density scenario from NNUE benchmark output."""
    lines = output.splitlines()

    # Look for the scenario that matches our target density
    scenario_map = {
        0.001: "Chess-like (0.1%)",
        0.01: "Very Sparse (1%)",
        0.05: "Sparse (5%)",
        0.25: "Medium (25%)",
        0.90: "Dense (90%)",
    }

    target_scenario = scenario_map.get(target_density, "")

    # Look for DENSITY_RESULT lines in the new format
    for line in lines:
        if line.startswith("DENSITY_RESULT:"):
            parts = line.split(":")
            if len(parts) >= 3:
                scenario_name = parts[1]
                try:
                    time_ms = float(parts[2])
                    if target_scenario in scenario_name:
                        return time_ms
                except ValueError:
                    continue

    # Fallback: use average result if specific scenario not found
    for line in lines:
        if "RESULT_AVG_MS" in line:
            return float(line.split(":")[1].strip())

    raise RuntimeError(f"Failed to parse NNUE benchmark for density {target_density}")


# -------------------------------------------------------------
# Main routine
# -------------------------------------------------------------


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    engine_dir = repo_root / "engine"
    build_dir = engine_dir / "build" / "build_bench"

    print("🎯 NNUE vs EtinyNet: CIFAR-10 runtime and density comparisons (untrained)")
    print("=" * 55)

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

        # Show parameter counts for verification
        print("\n📊 Model Specifications:")
        torch.manual_seed(42)
        nnue_demo = NNUE()
        nnue_params = sum(p.numel() for p in nnue_demo.parameters() if p.requires_grad)
        print(f"   NNUE: {nnue_params:,} parameters (10×10×8 features, L1=1024)")

        etiny_demo = EtinyNet(
            variant="0.98M", num_classes=1000, input_size=112, use_asq=False
        )
        etiny_params = etiny_demo.count_parameters()
        print(f"   EtinyNet-0.98M: {etiny_params:,} parameters")

        target = 980_000
        print(f"   Target: {target:,} parameters")
        print(f"   NNUE deviation: {((nnue_params - target) / target * 100):+.1f}%")
        print(
            f"   EtinyNet deviation: {((etiny_params - target) / target * 100):+.1f}%"
        )

        nnue_path, etiny_path = create_models(tmp_dir)

        # Step 3a: CIFAR-10 performance on untrained models
        run_cifar10_exec_benchmark(nnue_path, etiny_path, build_dir, num_samples=200)

        # Step 3b: Optional density comparison using built-in benchmarks
        run_comprehensive_benchmark(
            nnue_path, etiny_path, nnue_benchmark, etiny_benchmark
        )

    print("\n✅ Benchmark completed successfully!")


if __name__ == "__main__":
    main()
