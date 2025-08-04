#!/usr/bin/env python3
"""
Example: Quick MCU Benchmark for NNUE-Vision

This script demonstrates how to run MCU benchmarks and generate
TinyML-style comparison reports. Run this after training a model.

Usage:
    # Train a model first
    python train.py nnue --config config/train_nnue_default.py --max_epochs 20

    # Then run this benchmark
    python run_example_benchmark.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Import the new benchmarking modules
from benchmarks.mcu_simulation import MCU_SPECS, run_mcu_benchmark
from benchmarks.tinyml_benchmarks import generate_tinyml_report
from data.datasets import GenericVisionDataset
from model import NNUE, GridFeatureSet


def create_example_model():
    """Create a small NNUE model for demonstration."""
    print("🔧 Creating example NNUE model...")

    # Small feature set for quick demo
    feature_set = GridFeatureSet(grid_size=8, num_features_per_square=12)

    model = NNUE(
        feature_set=feature_set,
        l1_size=256,  # Smaller for demo
        l2_size=8,
        l3_size=16,
        num_classes=10,
        visual_threshold=0.5,
    )

    print(
        f"✅ Created model with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model


def create_demo_dataset():
    """Create a small dataset for demonstration."""
    print("📊 Setting up demo dataset...")

    dataset = GenericVisionDataset(
        dataset_name="cifar10",
        split="test",
        target_size=(96, 96),
        max_samples=500,  # Small for demo
        binary_classification={
            "positive_classes": [
                "airplane",
                "bird",
            ]  # Person-like for Visual Wake Words demo
        },
    )

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    print(f"✅ Created dataset with {len(dataset)} samples")
    return data_loader


def main():
    """Run example MCU benchmark."""
    print("🚀 NNUE-Vision MCU Benchmark Example")
    print("=" * 50)

    # Check if we have a trained model, otherwise create a demo
    model_path = Path("visual_wake_words_model.pt")

    if model_path.exists():
        print(f"📥 Loading existing model: {model_path}")
        model = torch.load(model_path, map_location="cpu", weights_only=True)
        if hasattr(model, "state_dict"):
            # Handle model wrapper
            state_dict = model.state_dict() if hasattr(model, "state_dict") else model
            demo_model = create_example_model()
            try:
                demo_model.load_state_dict(state_dict, strict=False)
                model = demo_model
            except:
                print("⚠️  Model loading failed, using fresh model")
                model = demo_model
        else:
            model = create_example_model()
    else:
        print("⚠️  No trained model found, using randomly initialized model")
        print(
            "   (For real benchmarks, train a model first with: python train.py nnue)"
        )
        model = create_example_model()

    model.eval()
    device = torch.device("cpu")  # Use CPU for demo

    # Create demo dataset
    data_loader = create_demo_dataset()

    # Select a few MCU platforms for demo
    demo_mcus = [MCU_SPECS["cortex_m4"], MCU_SPECS["cortex_m33"]]

    print(f"\n🔬 Running MCU Theoretical Analysis...")
    print("=" * 30)
    print("📝 Note: Not actual MCU execution - mathematical estimates only")

    # Run MCU benchmark
    mcu_results = run_mcu_benchmark(
        model=model,
        data_loader=data_loader,
        mcu_specs=demo_mcus,
        target_sparsity=0.6,  # Assume 60% sparsity
        verbose=True,
    )

    print(f"\n🎯 Generating TinyML Report...")
    print("=" * 30)
    print(
        "📝 Note: MCU metrics are theoretical calculations, not hardware measurements"
    )

    # Generate TinyML comparison report
    output_dir = Path("example_benchmark_results")
    output_dir.mkdir(exist_ok=True)

    tinyml_report = generate_tinyml_report(
        model=model,
        dataset_name="visual_wake_words",
        data_loader=data_loader,
        device=device,
        mcu_specs=demo_mcus,
        output_path=output_dir / "example_report.json",
    )

    print(f"\n📋 BENCHMARK SUMMARY")
    print("=" * 30)

    # Print key results
    summary = tinyml_report["summary"]
    print(f"🏆 Best Platform: {summary['best_platform']}")
    print(f"   Latency: {summary['best_latency_ms']:.1f} ms")
    print(f"   Throughput: {summary['best_throughput_fps']:.1f} FPS")
    print(f"   Energy: {summary['best_energy_uj']:.1f} μJ")

    print(f"\n🎯 Key NNUE Advantages:")
    for advantage in summary["key_advantages"]:
        print(f"   • {advantage}")

    # Show comparison with baselines
    if tinyml_report["baseline_references"]:
        print(f"\n📊 Published Baselines (Reference Only):")
        for baseline_name, baseline_info in tinyml_report[
            "baseline_references"
        ].items():
            baseline = baseline_info["baseline"]
            print(
                f"   {baseline_name}: {baseline['accuracy']:.3f} acc, {baseline['model_size_kb']:.1f} KB"
            )
            if baseline["latency_ms"] is not None:
                print(
                    f"      Latency: {baseline['latency_ms']:.1f} ms ({baseline.get('platform', 'Unknown')})"
                )
            print(f"      Note: Not directly comparable - different hardware/methods")

    # Show incremental update benefits
    incremental = tinyml_report["incremental_update_analysis"]
    print(f"\n🔄 Incremental Update Analysis:")
    print(
        f"   Full recompute: {incremental['full_recompute_avg_ms']:.2f} ms (measured on host)"
    )
    print(
        f"   Incremental: {incremental['incremental_avg_ms']:.2f} ms (theoretical simulation)"
    )
    print(
        f"   Speedup: {incremental['speedup']:.1f}x (potential with optimized C++ engine)"
    )

    print(f"\n💾 Results saved to: {output_dir}")
    print(f"\n✅ Example benchmark complete!")

    print(f"\n🚀 Next Steps:")
    print(
        f"   1. Train a real model: python train.py nnue --config config/train_nnue_default.py"
    )
    print(
        f"   2. Run full benchmark: python scripts/run_mcu_benchmarks.py --model model.pt"
    )
    print(f"   3. See benchmarks/README.md for detailed documentation")


if __name__ == "__main__":
    main()
