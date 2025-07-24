#!/usr/bin/env python3
"""
Benchmark Validation Script

This script validates the MCU benchmarking system against known values
and realistic expectations to ensure accuracy and reliability.

Usage:
    python scripts/validate_benchmarks.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarks.mcu_simulation import (
    MCU_SPECS,
    count_mac_operations,
    count_model_parameters,
    estimate_memory_usage,
    estimate_quantized_model_size,
    generate_model_stats,
    run_mcu_benchmark,
)
from benchmarks.tinyml_benchmarks import generate_tinyml_report
from data.datasets import GenericVisionDataset
from model import NNUE, GridFeatureSet


def validate_parameter_counting():
    """Validate parameter counting accuracy."""
    print("🔍 Validating parameter counting...")

    # Create a model with known architecture
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=4)  # 64 features
    model = NNUE(
        feature_set=feature_set, l1_size=32, l2_size=4, l3_size=8, num_ls_buckets=1
    )

    # Count parameters
    our_total, our_trainable = count_model_parameters(model)
    torch_total = sum(p.numel() for p in model.parameters())
    torch_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Validate exact match
    assert (
        our_total == torch_total
    ), f"Parameter count mismatch: {our_total} vs {torch_total}"
    assert (
        our_trainable == torch_trainable
    ), f"Trainable count mismatch: {our_trainable} vs {torch_trainable}"

    print(f"✅ Parameter counting: {our_total:,} parameters (matches PyTorch)")


def validate_mac_operations():
    """Validate MAC operation counting."""
    print("🔍 Validating MAC operations...")

    # Create model with known dimensions
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)  # 128 features
    model = NNUE(
        feature_set=feature_set, l1_size=64, l2_size=8, l3_size=16, num_ls_buckets=1
    )

    mac_counts = count_mac_operations(model, input_shape=(3, 96, 96))

    # Validate conv layer MACs manually
    conv = model.conv
    conv_input_h, conv_input_w = 96, 96
    conv_output_h = conv_input_h // conv.stride[0]
    conv_output_w = conv_input_w // conv.stride[1]

    expected_conv_macs = (
        conv.out_channels
        * conv.in_channels
        * conv.kernel_size[0]
        * conv.kernel_size[1]
        * conv_output_h
        * conv_output_w
    )

    assert (
        mac_counts["conv"] == expected_conv_macs
    ), f"Conv MAC mismatch: {mac_counts['conv']} vs {expected_conv_macs}"

    # Validate feature transformer MACs
    expected_ft_macs = feature_set.num_features * model.l1_size
    assert (
        mac_counts["feature_transformer"] == expected_ft_macs
    ), f"FT MAC mismatch: {mac_counts['feature_transformer']} vs {expected_ft_macs}"

    # Validate total
    expected_total = (
        mac_counts["conv"] + mac_counts["feature_transformer"] + mac_counts["linear"]
    )
    assert (
        mac_counts["total"] == expected_total
    ), f"Total MAC mismatch: {mac_counts['total']} vs {expected_total}"

    print(f"✅ MAC operations: {mac_counts['total']:,} total MACs")


def validate_quantization():
    """Validate quantization calculations."""
    print("🔍 Validating quantization...")

    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
    model = NNUE(feature_set=feature_set, l1_size=64, l2_size=8, l3_size=16)

    # Get sizes
    total_params, _ = count_model_parameters(model)
    float32_size = total_params * 4
    quantized_size = estimate_quantized_model_size(model)

    # Validate compression
    compression_ratio = float32_size / quantized_size
    assert (
        2.0 <= compression_ratio <= 8.0
    ), f"Unrealistic compression ratio: {compression_ratio:.2f}"
    assert quantized_size < float32_size, "Quantized model should be smaller"

    print(
        f"✅ Quantization: {compression_ratio:.1f}x compression ({quantized_size} vs {float32_size} bytes)"
    )


def validate_memory_estimation():
    """Validate memory usage estimation."""
    print("🔍 Validating memory estimation...")

    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
    model = NNUE(feature_set=feature_set, l1_size=64, l2_size=8, l3_size=16)

    memory_usage = estimate_memory_usage(model)

    # Validate parameter memory matches exactly
    total_params, _ = count_model_parameters(model)
    expected_param_memory = total_params * 4
    assert (
        memory_usage["model_parameters"] == expected_param_memory
    ), "Parameter memory mismatch"

    # Validate all components are positive
    for key, value in memory_usage.items():
        assert value >= 0, f"Negative memory value for {key}: {value}"

    # Validate total is reasonable sum
    component_sum = (
        memory_usage["model_parameters"]
        + memory_usage["activations"]
        + memory_usage["working_buffers"]
    )
    assert (
        abs(memory_usage["total"] - component_sum) < 1000
    ), "Total memory doesn't match components"

    print(f"✅ Memory estimation: {memory_usage['total']:,} bytes total")


def validate_latency_scaling():
    """Validate latency scaling behavior."""
    print("🔍 Validating latency scaling...")

    from benchmarks.mcu_simulation import ModelStats, simulate_mcu_latency

    # Create stats with different MAC counts
    stats_1x = ModelStats(
        total_parameters=1000,
        trainable_parameters=1000,
        total_macs=100000,
        conv_macs=50000,
        linear_macs=30000,
        feature_transformer_macs=20000,
        model_size_bytes=4000,
        activation_memory_bytes=1000,
        quantized_model_size_bytes=1000,
        quantization_bit_width=8,
        sparsity_ratio=0.0,
        effective_macs=100000,
    )

    stats_4x = ModelStats(
        total_parameters=4000,
        trainable_parameters=4000,
        total_macs=400000,
        conv_macs=200000,
        linear_macs=120000,
        feature_transformer_macs=80000,
        model_size_bytes=16000,
        activation_memory_bytes=4000,
        quantized_model_size_bytes=4000,
        quantization_bit_width=8,
        sparsity_ratio=0.0,
        effective_macs=400000,
    )

    mcu = MCU_SPECS["cortex_m33"]

    latency_1x = simulate_mcu_latency(stats_1x, mcu)
    latency_4x = simulate_mcu_latency(stats_4x, mcu)

    # Validate scaling
    latency_ratio = (
        latency_4x["quantized_latency_ms"] / latency_1x["quantized_latency_ms"]
    )
    mac_ratio = stats_4x.effective_macs / stats_1x.effective_macs

    # Should be roughly proportional (within 2x due to overheads)
    assert (
        0.5 * mac_ratio <= latency_ratio <= 2.0 * mac_ratio
    ), f"Poor latency scaling: {latency_ratio:.2f} vs expected ~{mac_ratio:.2f}"

    # Throughput should be inversely related
    fps_ratio = latency_1x["throughput_fps"] / latency_4x["throughput_fps"]
    assert (
        0.5 * mac_ratio <= fps_ratio <= 2.0 * mac_ratio
    ), f"Poor FPS scaling: {fps_ratio:.2f}"

    print(f"✅ Latency scaling: {latency_ratio:.2f}x for {mac_ratio:.2f}x MACs")


def validate_energy_calculations():
    """Validate energy consumption calculations."""
    print("🔍 Validating energy calculations...")

    from benchmarks.mcu_simulation import estimate_energy_consumption

    mcu = MCU_SPECS["cortex_m33"]

    # Test energy scaling with latency
    latencies = [10.0, 50.0, 100.0]  # ms
    energies = [estimate_energy_consumption(lat, mcu) for lat in latencies]

    # Validate energy increases with latency
    for i in range(1, len(energies)):
        assert (
            energies[i]["total_energy_uj"] > energies[i - 1]["total_energy_uj"]
        ), "Energy should increase with latency"

    # Validate energy calculation accuracy
    for i, lat in enumerate(latencies):
        energy = energies[i]

        # Basic physics: Energy = Power × Time
        expected_active = mcu.power_active_mw * lat  # mJ
        actual_active = energy["active_energy_mj"]

        # Should be close (within 10%)
        error = abs(actual_active - expected_active) / expected_active
        assert error < 0.1, f"Energy calculation error too high: {error:.1%}"

    print(f"✅ Energy calculations: Linear scaling validated")


def validate_realistic_benchmarks():
    """Validate benchmarks produce realistic results."""
    print("🔍 Validating realistic benchmark results...")

    # Create test model
    feature_set = GridFeatureSet(grid_size=8, num_features_per_square=12)
    model = NNUE(
        feature_set=feature_set, l1_size=128, l2_size=16, l3_size=32, num_ls_buckets=2
    )
    model.eval()

    # Create test dataset
    dataset = GenericVisionDataset(
        dataset_name="cifar10",
        split="test",
        target_size=(96, 96),
        max_samples=100,
        binary_classification={"positive_classes": ["airplane"]},
    )
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Run benchmark
    results = run_mcu_benchmark(
        model=model,
        data_loader=data_loader,
        mcu_specs=[MCU_SPECS["cortex_m33"]],
        verbose=False,
    )

    model_stats = results["model_stats"]
    mcu_result = results["mcu_results"]["ARM Cortex-M33"]

    # Validate model stats are reasonable
    assert (
        10000 <= model_stats.total_parameters <= 1000000
    ), f"Unrealistic parameter count: {model_stats.total_parameters}"
    assert (
        100000 <= model_stats.total_macs <= 100000000
    ), f"Unrealistic MAC count: {model_stats.total_macs}"
    assert (
        2.0 <= model_stats.compression_ratio <= 8.0
    ), f"Unrealistic compression: {model_stats.compression_ratio}"
    assert (
        0.0 <= model_stats.sparsity_ratio <= 1.0
    ), f"Invalid sparsity: {model_stats.sparsity_ratio}"

    # Validate MCU results are reasonable
    latency = mcu_result["latency"]["quantized_latency_ms"]
    energy = mcu_result["energy"]["total_energy_uj"]
    fps = mcu_result["efficiency_metrics"]["fps"]

    assert 0.1 <= latency <= 10000.0, f"Unrealistic latency: {latency} ms"
    assert 0.1 <= energy <= 100000.0, f"Unrealistic energy: {energy} μJ"
    assert 0.1 <= fps <= 10000.0, f"Unrealistic FPS: {fps}"

    # Validate memory fits
    memory_fit = mcu_result["memory_fit"]
    assert isinstance(memory_fit["flash"], bool), "Memory fit should be boolean"
    assert isinstance(memory_fit["ram"], bool), "Memory fit should be boolean"
    assert (
        0.0 <= memory_fit["flash_usage_percent"] <= 200.0
    ), "Flash usage percentage invalid"
    assert (
        0.0 <= memory_fit["ram_usage_percent"] <= 200.0
    ), "RAM usage percentage invalid"

    print(f"✅ Realistic results: {latency:.1f}ms, {energy:.0f}μJ, {fps:.1f}FPS")


def validate_tinyml_report():
    """Validate TinyML report generation."""
    print("🔍 Validating TinyML report generation...")

    # Create test model
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
    model = NNUE(feature_set=feature_set, l1_size=64, l2_size=8, l3_size=16)
    model.eval()

    # Create test dataset
    dataset = GenericVisionDataset(
        dataset_name="cifar10",
        split="test",
        max_samples=50,
        binary_classification={"positive_classes": ["airplane"]},
    )
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "validation_report.json"

        # Generate report
        report = generate_tinyml_report(
            model=model,
            dataset_name="visual_wake_words",
            data_loader=data_loader,
            device=torch.device("cpu"),
            mcu_specs=[MCU_SPECS["cortex_m4"]],
            output_path=output_path,
        )

        # Validate report structure
        required_sections = [
            "model_info",
            "mlperf_results",
            "baseline_comparisons",
            "incremental_update_analysis",
            "summary",
        ]
        for section in required_sections:
            assert section in report, f"Missing report section: {section}"

        # Validate MLPerf results
        mlperf_results = report["mlperf_results"]
        assert "ARM Cortex-M4" in mlperf_results, "Missing MCU results"

        result = mlperf_results["ARM Cortex-M4"]
        assert (
            0.0 <= result["accuracy"] <= 1.0
        ), f"Invalid accuracy: {result['accuracy']}"
        assert result["latency_ms"] > 0, f"Invalid latency: {result['latency_ms']}"
        assert result["energy_uj"] > 0, f"Invalid energy: {result['energy_uj']}"

        # Validate file was saved and is valid JSON
        assert output_path.exists(), "Report file not saved"
        with open(output_path) as f:
            saved_report = json.load(f)
        assert (
            saved_report["model_info"]["name"] == "NNUE-Vision"
        ), "Invalid saved report"

    print(f"✅ TinyML report: Complete with all sections")


def validate_sparsity_benefits():
    """Validate sparsity provides expected benefits."""
    print("🔍 Validating sparsity benefits...")

    from benchmarks.mcu_simulation import ModelStats, simulate_mcu_latency

    # Create dense vs sparse model stats
    base_stats = ModelStats(
        total_parameters=10000,
        trainable_parameters=10000,
        total_macs=1000000,
        conv_macs=500000,
        linear_macs=300000,
        feature_transformer_macs=200000,
        model_size_bytes=40000,
        activation_memory_bytes=8000,
        quantized_model_size_bytes=10000,
        quantization_bit_width=8,
        sparsity_ratio=0.0,
        effective_macs=1000000,  # Dense
    )

    sparse_stats = ModelStats(
        total_parameters=10000,
        trainable_parameters=10000,
        total_macs=1000000,
        conv_macs=500000,
        linear_macs=300000,
        feature_transformer_macs=200000,
        model_size_bytes=40000,
        activation_memory_bytes=8000,
        quantized_model_size_bytes=10000,
        quantization_bit_width=8,
        sparsity_ratio=0.8,
        effective_macs=200000,  # 80% sparse
    )

    mcu = MCU_SPECS["cortex_m33"]

    dense_latency = simulate_mcu_latency(base_stats, mcu, use_sparsity=False)
    sparse_latency = simulate_mcu_latency(sparse_stats, mcu, use_sparsity=True)

    # Validate sparsity provides speedup
    speedup = (
        dense_latency["quantized_latency_ms"] / sparse_latency["quantized_latency_ms"]
    )
    expected_speedup = base_stats.effective_macs / sparse_stats.effective_macs  # 5x

    assert speedup > 1.0, f"Sparsity should provide speedup, got {speedup:.2f}"
    assert (
        speedup >= 0.8 * expected_speedup
    ), f"Sparsity speedup too low: {speedup:.2f} vs expected ~{expected_speedup:.2f}"

    # Validate speedup calculation
    assert (
        sparse_stats.sparsity_speedup == 5.0
    ), f"Wrong sparsity speedup calculation: {sparse_stats.sparsity_speedup}"

    print(f"✅ Sparsity benefits: {speedup:.1f}x speedup from 80% sparsity")


def run_all_validations():
    """Run all validation tests."""
    print("🚀 Validating MCU Benchmarking System")
    print("=" * 50)

    try:
        validate_parameter_counting()
        validate_mac_operations()
        validate_quantization()
        validate_memory_estimation()
        validate_latency_scaling()
        validate_energy_calculations()
        validate_sparsity_benefits()
        validate_realistic_benchmarks()
        validate_tinyml_report()

        print("\n" + "=" * 50)
        print("✅ ALL VALIDATIONS PASSED!")
        print("🎯 MCU benchmarking system is accurate and reliable")

        return True

    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False


def main():
    """Main validation function."""
    success = run_all_validations()

    if success:
        print("\n🏆 Benchmark validation complete - system ready for use!")
        return 0
    else:
        print("\n⚠️  Validation failed - please check the implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
