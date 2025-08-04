"""
Tests for MCU Benchmarking System

This module tests the accuracy and reliability of the MCU simulation
and TinyML benchmarking capabilities.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from benchmarks.mcu_simulation import (
    MCUSpecs,
    ModelStats,
    analyze_sparsity,
    count_mac_operations,
    count_model_parameters,
    estimate_energy_consumption,
    estimate_memory_usage,
    estimate_quantized_model_size,
    generate_model_stats,
    run_mcu_benchmark,
    simulate_mcu_latency,
)
from benchmarks.tinyml_benchmarks import (
    TINYML_BASELINES,
    BaselineComparison,
    MLPerfTinyResult,
    benchmark_incremental_updates,
    compare_with_baselines,
    evaluate_model_accuracy,
    generate_tinyml_report,
    run_mlperf_tiny_benchmark,
)
from data.datasets import GenericVisionDataset
from model import NNUE, GridFeatureSet


@pytest.fixture
def small_nnue_model(device):
    """Create a small NNUE model for testing."""
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)

    model = NNUE(
        feature_set=feature_set,
        l1_size=64,
        l2_size=8,
        l3_size=16,
        num_classes=10,
        visual_threshold=0.5,
    )
    model.to(device)
    model.eval()

    return model


@pytest.fixture
def test_data_loader():
    """Create a small test data loader."""
    dataset = GenericVisionDataset(
        dataset_name="cifar10",
        split="test",
        target_size=(96, 96),
        max_samples=50,
        binary_classification={"positive_classes": ["airplane"]},
    )

    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def test_mcu_spec():
    """Create a test MCU specification."""
    return MCUSpecs(
        name="Test MCU",
        peak_ops_per_second=100e6,  # 100 MOPS
        memory_flash_kb=512,
        memory_ram_kb=128,
        power_active_mw=50,
        power_idle_mw=2,
    )


class TestParameterCounting:
    """Test parameter counting accuracy."""

    def test_count_model_parameters(self, small_nnue_model):
        """Test parameter counting matches PyTorch's count."""
        total_params, trainable_params = count_model_parameters(small_nnue_model)

        # Verify against PyTorch's built-in counting
        expected_total = sum(p.numel() for p in small_nnue_model.parameters())
        expected_trainable = sum(
            p.numel() for p in small_nnue_model.parameters() if p.requires_grad
        )

        assert total_params == expected_total
        assert trainable_params == expected_trainable
        assert trainable_params <= total_params

    def test_parameter_count_consistency(self, small_nnue_model):
        """Test parameter counting is consistent across calls."""
        count1 = count_model_parameters(small_nnue_model)
        count2 = count_model_parameters(small_nnue_model)

        assert count1 == count2


class TestMACOperations:
    """Test MAC operation counting."""

    def test_conv_mac_calculation(self, small_nnue_model):
        """Test convolutional layer MAC counting."""
        mac_counts = count_mac_operations(small_nnue_model)

        # Manually calculate expected conv MACs
        conv = small_nnue_model.conv
        input_h, input_w = 96, 96
        output_h = input_h // conv.stride[0]
        output_w = input_w // conv.stride[1]

        expected_conv_macs = (
            conv.out_channels
            * conv.in_channels
            * conv.kernel_size[0]
            * conv.kernel_size[1]
            * output_h
            * output_w
        )

        assert mac_counts["conv"] == expected_conv_macs
        assert mac_counts["conv"] > 0

    def test_feature_transformer_mac_calculation(self, small_nnue_model):
        """Test feature transformer MAC counting."""
        mac_counts = count_mac_operations(small_nnue_model)

        # Expected FT MACs
        expected_ft_macs = (
            small_nnue_model.feature_set.num_features * small_nnue_model.l1_size
        )

        assert mac_counts["feature_transformer"] == expected_ft_macs
        assert mac_counts["feature_transformer"] > 0

    def test_total_mac_calculation(self, small_nnue_model):
        """Test total MAC calculation is sum of components."""
        mac_counts = count_mac_operations(small_nnue_model)

        expected_total = (
            mac_counts["conv"]
            + mac_counts["feature_transformer"]
            + mac_counts["linear"]
        )

        assert mac_counts["total"] == expected_total
        assert mac_counts["total"] > 0

    def test_mac_counts_positive(self, small_nnue_model):
        """Test all MAC counts are positive."""
        mac_counts = count_mac_operations(small_nnue_model)

        for key, count in mac_counts.items():
            assert count >= 0, f"MAC count for {key} should be non-negative"


class TestMemoryAnalysis:
    """Test memory usage analysis."""

    def test_memory_estimation_structure(self, small_nnue_model):
        """Test memory estimation returns correct structure."""
        memory_usage = estimate_memory_usage(small_nnue_model)

        required_keys = ["model_parameters", "activations", "working_buffers", "total"]
        for key in required_keys:
            assert key in memory_usage
            assert memory_usage[key] >= 0

        # Total should be sum of components (approximately)
        component_sum = (
            memory_usage["model_parameters"]
            + memory_usage["activations"]
            + memory_usage["working_buffers"]
        )
        assert abs(memory_usage["total"] - component_sum) < 100  # Small tolerance

    def test_model_parameter_memory(self, small_nnue_model):
        """Test model parameter memory calculation."""
        memory_usage = estimate_memory_usage(small_nnue_model)
        total_params, _ = count_model_parameters(small_nnue_model)

        expected_param_memory = total_params * 4  # 4 bytes per float32
        assert memory_usage["model_parameters"] == expected_param_memory

    def test_memory_scales_with_model_size(self, device):
        """Test memory usage scales with model size."""
        # Small model
        small_feature_set = GridFeatureSet(grid_size=4, num_features_per_square=4)
        small_model = NNUE(
            feature_set=small_feature_set, l1_size=32, l2_size=4, l3_size=8
        )
        small_model.to(device)

        # Large model
        large_feature_set = GridFeatureSet(grid_size=8, num_features_per_square=8)
        large_model = NNUE(
            feature_set=large_feature_set, l1_size=128, l2_size=16, l3_size=32
        )
        large_model.to(device)

        small_memory = estimate_memory_usage(small_model)
        large_memory = estimate_memory_usage(large_model)

        assert large_memory["total"] > small_memory["total"]
        assert large_memory["model_parameters"] > small_memory["model_parameters"]


class TestQuantization:
    """Test quantization analysis."""

    def test_quantized_model_size(self, small_nnue_model):
        """Test quantized model size calculation."""
        quantized_size = estimate_quantized_model_size(small_nnue_model)

        assert quantized_size > 0

        # Should be smaller than float32 model
        total_params, _ = count_model_parameters(small_nnue_model)
        float32_size = total_params * 4

        assert quantized_size < float32_size

    def test_quantization_compression_ratio(self, small_nnue_model):
        """Test quantization provides expected compression."""
        model_stats = generate_model_stats(small_nnue_model)

        assert model_stats.compression_ratio > 1.0  # Should compress
        assert model_stats.compression_ratio < 10.0  # Reasonable upper bound

    def test_quantized_data_consistency(self, small_nnue_model):
        """Test quantized model data is consistent."""
        quantized_data = small_nnue_model.get_quantized_model_data()

        # Check required keys exist
        assert "feature_transformer" in quantized_data
        assert "conv_layer" in quantized_data
        assert "metadata" in quantized_data

        # Check data types (our implementation returns float32 numpy arrays)
        ft_data = quantized_data["feature_transformer"]
        assert ft_data["weight"].dtype == np.float32
        assert ft_data["bias"].dtype == np.float32

        conv_data = quantized_data["conv_layer"]
        assert conv_data["weight"].dtype == np.float32
        # Conv bias is None in our implementation
        assert conv_data["bias"] is None


class TestSparsityAnalysis:
    """Test sparsity analysis."""

    def test_sparsity_analysis_structure(self, small_nnue_model, test_data_loader):
        """Test sparsity analysis returns correct structure."""
        sparsity_stats = analyze_sparsity(
            small_nnue_model, test_data_loader, num_samples=10
        )

        required_keys = ["avg_sparsity", "min_sparsity", "max_sparsity", "std_sparsity"]
        for key in required_keys:
            assert key in sparsity_stats
            assert 0.0 <= sparsity_stats[key] <= 1.0

    def test_sparsity_bounds(self, small_nnue_model, test_data_loader):
        """Test sparsity values are within valid bounds."""
        sparsity_stats = analyze_sparsity(
            small_nnue_model, test_data_loader, num_samples=5
        )

        assert sparsity_stats["min_sparsity"] <= sparsity_stats["avg_sparsity"]
        assert sparsity_stats["avg_sparsity"] <= sparsity_stats["max_sparsity"]
        assert sparsity_stats["std_sparsity"] >= 0.0

    def test_sparsity_with_different_thresholds(self, test_data_loader, device):
        """Test sparsity changes with visual threshold."""
        feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)

        # Model with low threshold (more features active)
        low_thresh_model = NNUE(feature_set=feature_set, visual_threshold=-0.5)
        low_thresh_model.to(device).eval()

        # Model with high threshold (fewer features active)
        high_thresh_model = NNUE(feature_set=feature_set, visual_threshold=0.5)
        high_thresh_model.to(device).eval()

        low_sparsity = analyze_sparsity(
            low_thresh_model, test_data_loader, num_samples=5
        )
        high_sparsity = analyze_sparsity(
            high_thresh_model, test_data_loader, num_samples=5
        )

        # Higher threshold should lead to higher sparsity
        assert high_sparsity["avg_sparsity"] >= low_sparsity["avg_sparsity"]


class TestLatencySimulation:
    """Test MCU latency simulation."""

    def test_latency_simulation_structure(self, test_mcu_spec):
        """Test latency simulation returns correct structure."""
        model_stats = ModelStats(
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
            sparsity_ratio=0.5,
            effective_macs=50000,
        )

        latency_results = simulate_mcu_latency(model_stats, test_mcu_spec)

        required_keys = [
            "base_latency_ms",
            "with_overhead_ms",
            "quantized_latency_ms",
            "throughput_fps",
        ]
        for key in required_keys:
            assert key in latency_results
            assert latency_results[key] > 0

    def test_latency_scaling(self, test_mcu_spec):
        """Test latency scales with MAC count."""
        low_mac_stats = ModelStats(
            total_parameters=1000,
            trainable_parameters=1000,
            total_macs=50000,
            conv_macs=25000,
            linear_macs=15000,
            feature_transformer_macs=10000,
            model_size_bytes=4000,
            activation_memory_bytes=1000,
            quantized_model_size_bytes=1000,
            quantization_bit_width=8,
            sparsity_ratio=0.5,
            effective_macs=25000,
        )

        high_mac_stats = ModelStats(
            total_parameters=2000,
            trainable_parameters=2000,
            total_macs=200000,
            conv_macs=100000,
            linear_macs=60000,
            feature_transformer_macs=40000,
            model_size_bytes=8000,
            activation_memory_bytes=2000,
            quantized_model_size_bytes=2000,
            quantization_bit_width=8,
            sparsity_ratio=0.5,
            effective_macs=100000,
        )

        low_latency = simulate_mcu_latency(low_mac_stats, test_mcu_spec)
        high_latency = simulate_mcu_latency(high_mac_stats, test_mcu_spec)

        assert (
            high_latency["quantized_latency_ms"] > low_latency["quantized_latency_ms"]
        )
        assert low_latency["throughput_fps"] > high_latency["throughput_fps"]

    def test_sparsity_benefits(self, test_mcu_spec):
        """Test sparsity provides latency benefits."""
        model_stats = ModelStats(
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
            sparsity_ratio=0.8,
            effective_macs=20000,  # 80% sparsity
        )

        sparse_latency = simulate_mcu_latency(
            model_stats, test_mcu_spec, use_sparsity=True
        )
        dense_latency = simulate_mcu_latency(
            model_stats, test_mcu_spec, use_sparsity=False
        )

        assert (
            sparse_latency["quantized_latency_ms"]
            < dense_latency["quantized_latency_ms"]
        )
        assert sparse_latency["throughput_fps"] > dense_latency["throughput_fps"]


class TestEnergyEstimation:
    """Test energy consumption estimation."""

    def test_energy_estimation_structure(self, test_mcu_spec):
        """Test energy estimation returns correct structure."""
        latency_ms = 50.0
        energy_results = estimate_energy_consumption(latency_ms, test_mcu_spec)

        required_keys = [
            "active_energy_mj",
            "idle_energy_mj",
            "total_energy_mj",
            "total_energy_uj",
        ]
        for key in required_keys:
            assert key in energy_results
            assert energy_results[key] >= 0

    def test_energy_scales_with_latency(self, test_mcu_spec):
        """Test energy consumption scales with latency."""
        short_latency = 10.0
        long_latency = 100.0

        short_energy = estimate_energy_consumption(short_latency, test_mcu_spec)
        long_energy = estimate_energy_consumption(long_latency, test_mcu_spec)

        assert long_energy["total_energy_mj"] > short_energy["total_energy_mj"]
        assert long_energy["total_energy_uj"] > short_energy["total_energy_uj"]

    def test_energy_unit_conversion(self, test_mcu_spec):
        """Test energy unit conversion is correct."""
        latency_ms = 50.0
        energy_results = estimate_energy_consumption(latency_ms, test_mcu_spec)

        # Verify unit conversion
        expected_uj = energy_results["total_energy_mj"] * 1000
        assert abs(energy_results["total_energy_uj"] - expected_uj) < 0.001


class TestAccuracyEvaluation:
    """Test accuracy evaluation."""

    def test_accuracy_evaluation_structure(
        self, small_nnue_model, test_data_loader, device
    ):
        """Test accuracy evaluation returns correct structure."""
        accuracy_metrics = evaluate_model_accuracy(
            small_nnue_model, test_data_loader, device, num_samples=20
        )

        assert "accuracy" in accuracy_metrics
        assert "total_samples" in accuracy_metrics
        assert 0.0 <= accuracy_metrics["accuracy"] <= 1.0
        assert accuracy_metrics["total_samples"] > 0

    def test_accuracy_with_different_sample_counts(
        self, small_nnue_model, test_data_loader, device
    ):
        """Test accuracy evaluation with different sample counts."""
        small_result = evaluate_model_accuracy(
            small_nnue_model, test_data_loader, device, num_samples=5
        )
        large_result = evaluate_model_accuracy(
            small_nnue_model, test_data_loader, device, num_samples=20
        )

        assert small_result["total_samples"] <= large_result["total_samples"]
        assert small_result["total_samples"] <= 5
        assert large_result["total_samples"] <= 20

    def test_accuracy_binary_classification_metrics(self, test_data_loader, device):
        """Test binary classification returns additional metrics."""
        # Create model for binary classification
        feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
        model = NNUE(feature_set=feature_set, l1_size=32, l2_size=4, l3_size=8)
        model.to(device).eval()

        accuracy_metrics = evaluate_model_accuracy(
            model, test_data_loader, device, num_samples=10
        )

        # Should have binary classification metrics
        expected_keys = ["accuracy", "precision", "recall", "f1_score", "total_samples"]
        for key in expected_keys:
            assert key in accuracy_metrics
            if key != "total_samples":
                assert 0.0 <= accuracy_metrics[key] <= 1.0


class TestMLPerfIntegration:
    """Test MLPerf Tiny integration."""

    def test_mlperf_result_structure(
        self, small_nnue_model, test_data_loader, device, test_mcu_spec
    ):
        """Test MLPerf result has correct structure."""
        result = run_mlperf_tiny_benchmark(
            small_nnue_model,
            "test_dataset",
            test_data_loader,
            device,
            test_mcu_spec,
            num_samples=10,
        )

        assert isinstance(result, MLPerfTinyResult)
        assert result.dataset == "test_dataset"
        assert 0.0 <= result.accuracy <= 1.0
        assert result.latency_ms > 0
        assert result.energy_uj > 0
        assert result.model_size_kb > 0
        assert result.throughput_fps > 0

    def test_mlperf_result_serialization(
        self, small_nnue_model, test_data_loader, device, test_mcu_spec
    ):
        """Test MLPerf result can be serialized."""
        result = run_mlperf_tiny_benchmark(
            small_nnue_model,
            "test_dataset",
            test_data_loader,
            device,
            test_mcu_spec,
            num_samples=5,
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)

        # Should be JSON serializable
        json_str = json.dumps(result_dict, default=str)
        assert len(json_str) > 0


class TestBaselineComparison:
    """Test baseline comparison functionality."""

    def test_baseline_comparison_structure(self):
        """Test baseline comparison data structure."""
        baseline = BaselineComparison(
            model_name="Test Model",
            dataset="test_dataset",
            accuracy=0.85,
            parameters=50000,
            macs=1000000,
            model_size_kb=200,
            latency_ms=50,
            paper="Test Paper",
        )

        assert baseline.model_name == "Test Model"
        assert baseline.accuracy == 0.85
        assert baseline.parameters == 50000

    def test_compare_with_baselines(self):
        """Test baseline comparison logic."""
        # Mock MLPerf result
        nnue_result = MLPerfTinyResult(
            dataset="test_dataset",
            accuracy=0.90,
            latency_ms=25.0,
            energy_uj=60.0,
            memory_peak_kb=150.0,
            model_size_kb=120.0,
            throughput_fps=40.0,
            sparsity_ratio=0.6,
            incremental_speedup=8.0,
        )

        # Create temporary baseline
        original_baselines = TINYML_BASELINES.get("test_dataset", [])
        test_baseline = BaselineComparison(
            model_name="Test Baseline",
            dataset="test_dataset",
            accuracy=0.85,
            parameters=60000,
            macs=2000000,
            model_size_kb=240,
            latency_ms=50,
            energy_uj=120,
            paper="Test Paper",
        )

        TINYML_BASELINES["test_dataset"] = [test_baseline]

        try:
            comparisons = compare_with_baselines(nnue_result, "test_dataset")

            assert "Test Baseline" in comparisons
            comparison = comparisons["Test Baseline"]

            assert "nnue_vs_baseline" in comparison
            ratios = comparison["nnue_vs_baseline"]

            # NNUE should be better in this test case
            assert ratios["accuracy_ratio"] > 1.0  # Better accuracy (higher is better)
            assert ratios["latency_ratio"] < 1.0  # Faster (lower latency is better)
            assert (
                ratios["energy_ratio"] < 1.0
            )  # More efficient (lower energy is better)

        finally:
            # Restore original baselines
            if original_baselines:
                TINYML_BASELINES["test_dataset"] = original_baselines
            else:
                TINYML_BASELINES.pop("test_dataset", None)


class TestIncrementalUpdates:
    """Test incremental update benchmarking."""

    def test_incremental_update_structure(
        self, small_nnue_model, test_data_loader, device
    ):
        """Test incremental update benchmark structure."""
        result = benchmark_incremental_updates(
            small_nnue_model,
            test_data_loader,
            device,
            num_sequences=2,
            sequence_length=3,
        )

        required_keys = [
            "full_recompute_avg_ms",
            "incremental_avg_ms",
            "speedup",
            "sequences_tested",
        ]
        for key in required_keys:
            assert key in result

        assert result["sequences_tested"] >= 0
        assert result["speedup"] >= 1.0  # Should never be slower

    def test_incremental_update_speedup(
        self, small_nnue_model, test_data_loader, device
    ):
        """Test incremental updates provide speedup."""
        result = benchmark_incremental_updates(
            small_nnue_model,
            test_data_loader,
            device,
            num_sequences=1,
            sequence_length=2,
        )

        if result["sequences_tested"] > 0:
            assert result["incremental_avg_ms"] <= result["full_recompute_avg_ms"]
            assert result["speedup"] >= 1.0


class TestEndToEndBenchmark:
    """Test end-to-end benchmark pipeline."""

    def test_mcu_benchmark_integration(
        self, small_nnue_model, test_data_loader, test_mcu_spec
    ):
        """Test full MCU benchmark pipeline."""
        results = run_mcu_benchmark(
            model=small_nnue_model,
            data_loader=test_data_loader,
            mcu_specs=[test_mcu_spec],
            target_sparsity=0.5,
            verbose=False,
        )

        assert "model_stats" in results
        assert "mcu_results" in results
        assert test_mcu_spec.name in results["mcu_results"]

        mcu_result = results["mcu_results"][test_mcu_spec.name]
        assert "latency" in mcu_result
        assert "energy" in mcu_result
        assert "memory_fit" in mcu_result
        assert "efficiency_metrics" in mcu_result

    def test_tinyml_report_generation(self, small_nnue_model, test_data_loader, device):
        """Test TinyML report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.json"

            report = generate_tinyml_report(
                model=small_nnue_model,
                dataset_name="test_dataset",
                data_loader=test_data_loader,
                device=device,
                mcu_specs=[MCUSpecs("Test MCU", 100e6, 512, 128, 50, 2)],
                output_path=output_path,
            )

            # Check report structure
            assert "model_info" in report
            assert "mlperf_results" in report
            assert "incremental_update_analysis" in report
            assert "summary" in report

            # Check file was created
            assert output_path.exists()

            # Check file content
            with open(output_path) as f:
                saved_report = json.load(f)

            assert saved_report["model_info"]["name"] == "NNUE-Vision"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_data_loader(self, small_nnue_model, device):
        """Test handling of empty data loader."""
        from torch.utils.data import TensorDataset

        # Create empty dataset
        empty_dataset = TensorDataset(
            torch.empty(0, 3, 96, 96), torch.empty(0, dtype=torch.long)
        )
        empty_loader = DataLoader(empty_dataset, batch_size=1)

        accuracy_metrics = evaluate_model_accuracy(
            small_nnue_model, empty_loader, device
        )

        # Should handle gracefully
        assert accuracy_metrics["accuracy"] == 0.0
        assert accuracy_metrics["total_samples"] == 0

    def test_invalid_mcu_specs(self, small_nnue_model, test_data_loader):
        """Test handling of invalid MCU specifications."""
        invalid_mcu = MCUSpecs(
            name="Invalid MCU",
            peak_ops_per_second=0,  # Invalid
            memory_flash_kb=0,  # Invalid
            memory_ram_kb=0,  # Invalid
            power_active_mw=0,
            power_idle_mw=0,
        )

        # Should not crash, but may give unrealistic results
        results = run_mcu_benchmark(
            model=small_nnue_model,
            data_loader=test_data_loader,
            mcu_specs=[invalid_mcu],
            verbose=False,
        )

        assert "mcu_results" in results
        assert invalid_mcu.name in results["mcu_results"]

    def test_model_stats_edge_cases(self, device):
        """Test model stats with edge case models."""
        # Tiny model
        tiny_feature_set = GridFeatureSet(grid_size=2, num_features_per_square=2)
        tiny_model = NNUE(
            feature_set=tiny_feature_set,
            l1_size=8,
            l2_size=2,
            l3_size=4,
            num_classes=10,
        )
        tiny_model.to(device).eval()

        stats = generate_model_stats(tiny_model, target_sparsity=0.0)

        assert stats.total_parameters > 0
        assert stats.total_macs > 0
        assert stats.quantized_model_size_bytes > 0
        assert stats.sparsity_ratio == 0.0
        assert stats.effective_macs == stats.total_macs


class TestNumericalAccuracy:
    """Test numerical accuracy of calculations."""

    def test_mac_calculation_precision(self, small_nnue_model):
        """Test MAC calculations are precise."""
        mac_counts = count_mac_operations(small_nnue_model)

        # Run multiple times to check consistency
        for _ in range(5):
            new_counts = count_mac_operations(small_nnue_model)
            assert new_counts == mac_counts

    def test_memory_calculation_precision(self, small_nnue_model):
        """Test memory calculations are precise."""
        memory_usage = estimate_memory_usage(small_nnue_model)

        # Run multiple times to check consistency
        for _ in range(5):
            new_usage = estimate_memory_usage(small_nnue_model)
            assert new_usage == memory_usage

    def test_quantization_deterministic(self, small_nnue_model):
        """Test quantization is deterministic."""
        size1 = estimate_quantized_model_size(small_nnue_model)
        size2 = estimate_quantized_model_size(small_nnue_model)

        assert size1 == size2

    def test_energy_calculation_precision(self, test_mcu_spec):
        """Test energy calculations are numerically stable."""
        latency_ms = 42.5

        energy1 = estimate_energy_consumption(latency_ms, test_mcu_spec)
        energy2 = estimate_energy_consumption(latency_ms, test_mcu_spec)

        for key in energy1:
            assert abs(energy1[key] - energy2[key]) < 1e-10


# Benchmark validation tests
class TestBenchmarkValidation:
    """Validate benchmark results against known values."""

    def test_known_model_metrics(self, device):
        """Test metrics against a known model configuration."""
        # Create a very specific model for validation
        feature_set = GridFeatureSet(
            grid_size=4, num_features_per_square=4
        )  # 64 features
        model = NNUE(
            feature_set=feature_set, l1_size=32, l2_size=4, l3_size=8, num_classes=10
        )
        model.to(device).eval()

        # Calculate metrics
        total_params, _ = count_model_parameters(model)
        mac_counts = count_mac_operations(model)

        # Validate parameter count makes sense
        # Conv: 3*4*3*3 + 4 bias = 112 params
        # FT: 64*32 + 32 bias = 2080 params
        # L1: 32*4 + 4 bias = 132 params
        # L2: (4*2)*8 + 8 bias = 72 params
        # Output: 8*1 + 1 bias = 9 params
        # Total ≈ 2405 params (approximately, layer stacks more complex)

        assert 2000 < total_params < 5000  # Reasonable range

        # Validate MAC counts make sense
        assert mac_counts["conv"] > 0
        assert mac_counts["feature_transformer"] == 64 * 32  # Exactly this
        assert mac_counts["linear"] > 0
        assert mac_counts["total"] == (
            mac_counts["conv"]
            + mac_counts["feature_transformer"]
            + mac_counts["linear"]
        )

    def test_sparsity_impact_validation(self, small_nnue_model, test_data_loader):
        """Validate sparsity impact is realistic."""
        model_stats = generate_model_stats(small_nnue_model, test_data_loader)

        # Sparsity should reduce effective MACs
        if model_stats.sparsity_ratio > 0:
            assert model_stats.effective_macs < model_stats.total_macs

            expected_effective = int(
                model_stats.total_macs * (1 - model_stats.sparsity_ratio)
            )
            assert (
                abs(model_stats.effective_macs - expected_effective) < 1000
            )  # Small tolerance

    def test_compression_ratio_realistic(self, small_nnue_model):
        """Test quantization compression ratio is realistic."""
        model_stats = generate_model_stats(small_nnue_model)

        # Should compress by 2x-8x (reasonable for mixed INT8/INT16)
        assert 2.0 <= model_stats.compression_ratio <= 8.0


if __name__ == "__main__":
    pytest.main([__file__])
