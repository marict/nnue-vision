#!/usr/bin/env python3
"""
Run MCU Benchmarks for NNUE-Vision

This script demonstrates how to run comprehensive MCU benchmarks including:
- MCU simulation with various platforms
- TinyML benchmark generation
- Comparison with published baselines
- NNUE-specific incremental update analysis
- Report generation with plots

Usage:
    python scripts/run_mcu_benchmarks.py --model checkpoints/model.pt --dataset cifar10
    python scripts/run_mcu_benchmarks.py --help
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.mcu_simulation import MCU_SPECS, run_mcu_benchmark
from benchmarks.tinyml_benchmarks import generate_tinyml_report
from data.datasets import GenericVisionDataset
from data.loaders import create_data_loaders
from nnue import NNUE
from serialize import load_model_from_checkpoint


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MCU benchmarks for NNUE-Vision models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt or .ckpt)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to benchmark on",
    )

    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for accuracy evaluation",
    )

    parser.add_argument(
        "--target-sparsity",
        type=float,
        default=None,
        help="Target sparsity ratio (if not measuring from data)",
    )

    parser.add_argument(
        "--mcu-platforms",
        nargs="+",
        default=["cortex_m4", "cortex_m33", "cortex_m7"],
        choices=list(MCU_SPECS.keys()),
        help="MCU platforms to benchmark",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for benchmark results",
    )

    parser.add_argument(
        "--device", type=str, default="auto", help="Device to run on (cpu, cuda, auto)"
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with fewer samples"
    )

    parser.add_argument(
        "--visual-wake-words",
        action="store_true",
        help="Configure dataset for Visual Wake Words style evaluation",
    )

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def setup_dataset(
    dataset_name: str, batch_size: int, visual_wake_words: bool = False
) -> Tuple[DataLoader, str]:
    """Setup dataset and data loader."""

    if visual_wake_words:
        # Configure for Visual Wake Words style binary classification
        # Map to "person" vs "no person" for CIFAR-10
        if dataset_name == "cifar10":
            dataset = GenericVisionDataset(
                dataset_name="cifar10",
                split="test",
                target_size=(96, 96),
                max_samples=5000,
                binary_classification={
                    "positive_classes": ["horse", "truck"]  # Arbitrary mapping for demo
                },
            )
            actual_dataset_name = "visual_wake_words"
        else:
            raise ValueError("Visual Wake Words mapping only implemented for CIFAR-10")
    else:
        # Standard dataset
        dataset = GenericVisionDataset(
            dataset_name=dataset_name,
            split="test",
            target_size=(96, 96),
            max_samples=5000,
        )
        actual_dataset_name = dataset_name

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
    )

    return data_loader, actual_dataset_name


def load_model(model_path: Path, device: torch.device) -> NNUE:
    """Load NNUE model from checkpoint."""
    print(f"📥 Loading model from {model_path}")

    model = load_model_from_checkpoint(model_path)
    model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully")
    print(
        f"   Architecture: {model.feature_set.num_features} → {model.l1_size} → {model.l2_size} → {model.l3_size}"
    )
    print(f"   Feature set: {model.feature_set}")
    print(f"   Number of classes: {model.num_classes}")

    return model


def generate_benchmark_plots(benchmark_results: Dict, output_dir: Path) -> None:
    """Generate visualization plots for benchmark results."""
    try:
        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # 1. Accuracy vs Model Size Pareto Plot
        from benchmarks.tinyml_benchmarks import create_pareto_plot_data

        pareto_data = create_pareto_plot_data(
            benchmark_results, x_metric="model_size_kb", y_metric="accuracy"
        )

        plt.figure(figsize=(10, 6))

        # Plot NNUE models
        if pareto_data["nnue_models"]:
            nnue_x = [m["x"] for m in pareto_data["nnue_models"]]
            nnue_y = [m["y"] for m in pareto_data["nnue_models"]]
            plt.scatter(nnue_x, nnue_y, c="red", s=100, label="NNUE-Vision", marker="o")

            # Add labels for NNUE points
            for model in pareto_data["nnue_models"]:
                plt.annotate(
                    model["platform"],
                    (model["x"], model["y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        # Plot baseline models
        if pareto_data["baseline_models"]:
            baseline_x = [m["x"] for m in pareto_data["baseline_models"]]
            baseline_y = [m["y"] for m in pareto_data["baseline_models"]]
            plt.scatter(
                baseline_x,
                baseline_y,
                c="blue",
                s=80,
                label="Published Baselines",
                marker="s",
            )

            # Add labels for baseline points
            for model in pareto_data["baseline_models"]:
                plt.annotate(
                    model["name"],
                    (model["x"], model["y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        plt.xlabel("Model Size (KB)")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Model Size - TinyML Comparison")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            plots_dir / "accuracy_vs_model_size.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # 2. Performance across MCU platforms
        plt.figure(figsize=(12, 8))

        platforms = list(benchmark_results["mlperf_results"].keys())
        metrics = ["latency_ms", "energy_uj", "throughput_fps"]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Latency plot
        latencies = [
            benchmark_results["mlperf_results"][p]["latency_ms"] for p in platforms
        ]
        axes[0].bar(platforms, latencies, color="skyblue")
        axes[0].set_title("Latency (ms)")
        axes[0].set_ylabel("Milliseconds")

        # Energy plot
        energies = [
            benchmark_results["mlperf_results"][p]["energy_uj"] for p in platforms
        ]
        axes[1].bar(platforms, energies, color="lightgreen")
        axes[1].set_title("Energy per Inference (μJ)")
        axes[1].set_ylabel("Microjoules")

        # Throughput plot
        throughputs = [
            benchmark_results["mlperf_results"][p]["throughput_fps"] for p in platforms
        ]
        axes[2].bar(platforms, throughputs, color="orange")
        axes[2].set_title("Throughput (FPS)")
        axes[2].set_ylabel("Frames per Second")

        # Memory usage plot
        memory_usage = [
            benchmark_results["mlperf_results"][p]["memory_peak_kb"] for p in platforms
        ]
        axes[3].bar(platforms, memory_usage, color="pink")
        axes[3].set_title("Peak Memory Usage (KB)")
        axes[3].set_ylabel("Kilobytes")

        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "mcu_performance_comparison.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        print(f"📊 Plots saved to {plots_dir}")

    except ImportError:
        print("⚠️  Matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"⚠️  Error generating plots: {e}")


def main():
    """Main function."""
    args = parse_arguments()

    # Setup
    device = get_device(args.device)
    print(f"🖥️  Using device: {device}")

    # Adjust parameters for quick run
    if args.quick:
        args.num_samples = 100
        print("⚡ Quick mode: using fewer samples")

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model, device)

    # Setup dataset
    data_loader, actual_dataset_name = setup_dataset(
        args.dataset, args.batch_size, args.visual_wake_words
    )

    print(f"📊 Dataset: {actual_dataset_name}")
    print(f"   Samples for evaluation: {args.num_samples}")
    print(f"   Batch size: {args.batch_size}")

    # Get MCU specs to test
    mcu_specs = [MCU_SPECS[platform] for platform in args.mcu_platforms]
    print(f"🔧 Testing MCU platforms: {args.mcu_platforms}")

    # Run MCU simulation benchmark
    print(f"\n" + "=" * 60)
    print("🚀 RUNNING MCU SIMULATION BENCHMARK")
    print("=" * 60)

    mcu_results = run_mcu_benchmark(
        model=model,
        data_loader=data_loader,
        mcu_specs=mcu_specs,
        target_sparsity=args.target_sparsity,
        verbose=True,
    )

    # Run TinyML benchmark report
    print(f"\n" + "=" * 60)
    print("🎯 GENERATING TINYML BENCHMARK REPORT")
    print("=" * 60)

    report_path = args.output_dir / f"tinyml_report_{actual_dataset_name}.json"

    tinyml_results = generate_tinyml_report(
        model=model,
        dataset_name=actual_dataset_name,
        data_loader=data_loader,
        device=device,
        mcu_specs=mcu_specs,
        output_path=report_path,
    )

    # Generate summary
    print(f"\n" + "=" * 60)
    print("📋 BENCHMARK SUMMARY")
    print("=" * 60)

    summary = tinyml_results["summary"]

    print(f"🏆 Best Performance Platform: {summary['best_platform']}")
    print(f"   Latency: {summary['best_latency_ms']:.2f} ms")
    print(f"   Throughput: {summary['best_throughput_fps']:.1f} FPS")
    print(f"   Energy: {summary['best_energy_uj']:.1f} μJ per inference")

    print(f"\n🎯 Key NNUE Advantages:")
    for advantage in summary["key_advantages"]:
        print(f"   • {advantage}")

    # Generate plots
    generate_benchmark_plots(tinyml_results, args.output_dir)

    # Save MCU simulation results separately
    mcu_results_path = args.output_dir / f"mcu_simulation_{actual_dataset_name}.json"

    with open(mcu_results_path, "w") as f:
        json.dump(mcu_results, f, indent=2, default=str)

    print(f"\n💾 Results saved:")
    print(f"   TinyML Report: {report_path}")
    print(f"   MCU Simulation: {mcu_results_path}")
    print(f"   Plots: {args.output_dir / 'plots'}")

    print(f"\n✅ MCU benchmark analysis complete!")


if __name__ == "__main__":
    main()
