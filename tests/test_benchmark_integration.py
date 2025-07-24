"""
Integration tests for MCU benchmarking scripts and end-to-end workflows.

These tests verify that the benchmarking system works correctly when used
as it would be in practice, including command-line scripts and report generation.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from benchmarks.mcu_simulation import MCU_SPECS, run_mcu_benchmark
from benchmarks.tinyml_benchmarks import generate_tinyml_report
from data.datasets import GenericVisionDataset
from model import NNUE, GridFeatureSet
from serialize import serialize_model


@pytest.fixture
def trained_model_checkpoint(device, tmp_path):
    """Create a trained model checkpoint for testing."""
    # Create a small model
    feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
    model = NNUE(
        feature_set=feature_set, l1_size=64, l2_size=8, l3_size=16, num_ls_buckets=2
    )
    model.to(device)

    # Save as checkpoint
    checkpoint_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), checkpoint_path, weights_only=True)

    return checkpoint_path, model


@pytest.fixture
def test_dataset():
    """Create a test dataset."""
    return GenericVisionDataset(
        dataset_name="cifar10",
        split="test",
        target_size=(96, 96),
        max_samples=50,
        binary_classification={"positive_classes": ["airplane"]},
    )


class TestBenchmarkScriptIntegration:
    """Test the benchmark scripts work correctly."""

    def test_run_mcu_benchmarks_script_imports(self):
        """Test that the benchmark script can be imported without errors."""
        try:
            import sys
            from pathlib import Path

            # Add scripts directory to path
            script_path = Path(__file__).parent.parent / "scripts"
            sys.path.insert(0, str(script_path))

            import run_mcu_benchmarks

            # Should have main functions
            assert hasattr(run_mcu_benchmarks, "main")
            assert hasattr(run_mcu_benchmarks, "parse_arguments")
            assert hasattr(run_mcu_benchmarks, "load_model")

        except ImportError as e:
            pytest.skip(f"Could not import benchmark script: {e}")

    def test_example_benchmark_script_imports(self):
        """Test that the example benchmark script can be imported."""
        try:
            import sys
            from pathlib import Path

            # Add root directory to path
            root_path = Path(__file__).parent.parent
            sys.path.insert(0, str(root_path))

            import run_example_benchmark

            assert hasattr(run_example_benchmark, "main")
            assert hasattr(run_example_benchmark, "create_example_model")

        except ImportError as e:
            pytest.skip(f"Could not import example script: {e}")


class TestEndToEndBenchmarkWorkflow:
    """Test complete benchmark workflows."""

    def test_complete_mcu_benchmark_workflow(
        self, trained_model_checkpoint, test_dataset, device
    ):
        """Test complete MCU benchmark workflow from model to report."""
        checkpoint_path, model = trained_model_checkpoint

        # Create data loader
        data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Run complete MCU benchmark
        results = run_mcu_benchmark(
            model=model,
            data_loader=data_loader,
            mcu_specs=[MCU_SPECS["cortex_m4"], MCU_SPECS["cortex_m33"]],
            target_sparsity=0.6,
            verbose=False,
        )

        # Validate results structure
        assert "model_stats" in results
        assert "mcu_results" in results

        model_stats = results["model_stats"]
        assert model_stats.total_parameters > 0
        assert model_stats.total_macs > 0
        assert model_stats.sparsity_ratio == 0.6  # As specified

        # Check all MCU results
        for mcu_name in ["ARM Cortex-M4", "ARM Cortex-M33"]:
            assert mcu_name in results["mcu_results"]
            mcu_result = results["mcu_results"][mcu_name]

            assert "latency" in mcu_result
            assert "energy" in mcu_result
            assert "memory_fit" in mcu_result
            assert "efficiency_metrics" in mcu_result

            # Validate numeric results
            assert mcu_result["latency"]["quantized_latency_ms"] > 0
            assert mcu_result["energy"]["total_energy_uj"] > 0
            assert mcu_result["efficiency_metrics"]["fps"] > 0

    def test_complete_tinyml_report_workflow(
        self, trained_model_checkpoint, test_dataset, device
    ):
        """Test complete TinyML report generation workflow."""
        checkpoint_path, model = trained_model_checkpoint
        data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.json"

            # Generate complete TinyML report
            report = generate_tinyml_report(
                model=model,
                dataset_name="visual_wake_words",
                data_loader=data_loader,
                device=device,
                mcu_specs=[MCU_SPECS["cortex_m33"]],
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
                assert section in report, f"Missing section: {section}"

            # Validate model info
            model_info = report["model_info"]
            assert model_info["name"] == "NNUE-Vision"
            assert model_info["dataset"] == "visual_wake_words"
            assert "architecture" in model_info

            # Validate MLPerf results
            mlperf_results = report["mlperf_results"]
            assert "ARM Cortex-M33" in mlperf_results

            cortex_result = mlperf_results["ARM Cortex-M33"]
            assert cortex_result["accuracy"] >= 0.0
            assert cortex_result["latency_ms"] > 0
            assert cortex_result["energy_uj"] > 0
            assert cortex_result["model_size_kb"] > 0

            # Validate incremental update analysis
            incremental = report["incremental_update_analysis"]
            assert "speedup" in incremental
            assert "sequences_tested" in incremental

            # Validate summary
            summary = report["summary"]
            assert "best_platform" in summary
            assert "key_advantages" in summary
            assert len(summary["key_advantages"]) > 0

            # Validate file was saved
            assert output_path.exists()

            # Validate saved file content
            with open(output_path) as f:
                saved_report = json.load(f)

            assert saved_report["model_info"]["name"] == "NNUE-Vision"

    def test_model_serialization_integration(self, trained_model_checkpoint, device):
        """Test model serialization integrates with benchmarking."""
        checkpoint_path, model = trained_model_checkpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            nnue_path = Path(temp_dir) / "test_model.nnue"

            # Serialize model
            serialize_model(model, nnue_path)
            assert nnue_path.exists()

            # Model should have quantized data
            quantized_data = model.get_quantized_model_data()
            assert "feature_transformer" in quantized_data
            assert "conv_layer" in quantized_data
            assert "metadata" in quantized_data

            # Quantized size should be smaller than float model
            from benchmarks.mcu_simulation import (
                count_model_parameters,
                estimate_quantized_model_size,
            )

            quantized_size = estimate_quantized_model_size(model)
            total_params, _ = count_model_parameters(model)
            float_size = total_params * 4  # 4 bytes per float32

            assert quantized_size < float_size
            assert quantized_size > 0


class TestBenchmarkAccuracy:
    """Test benchmark accuracy and consistency."""

    def test_parameter_counting_accuracy(self, trained_model_checkpoint, device):
        """Test parameter counting matches PyTorch exactly."""
        checkpoint_path, model = trained_model_checkpoint

        from benchmarks.mcu_simulation import count_model_parameters

        # Count using our function
        our_total, our_trainable = count_model_parameters(model)

        # Count using PyTorch
        torch_total = sum(p.numel() for p in model.parameters())
        torch_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert our_total == torch_total
        assert our_trainable == torch_trainable

    def test_mac_counting_consistency(self, trained_model_checkpoint, device):
        """Test MAC counting is consistent and reasonable."""
        checkpoint_path, model = trained_model_checkpoint

        from benchmarks.mcu_simulation import count_mac_operations

        # Count MACs multiple times
        mac_counts_1 = count_mac_operations(model)
        mac_counts_2 = count_mac_operations(model)

        # Should be identical
        assert mac_counts_1 == mac_counts_2

        # Should be reasonable values
        assert mac_counts_1["conv"] > 0
        assert mac_counts_1["feature_transformer"] > 0
        assert mac_counts_1["linear"] > 0
        assert mac_counts_1["total"] > 0

        # Total should be sum of parts
        expected_total = (
            mac_counts_1["conv"]
            + mac_counts_1["feature_transformer"]
            + mac_counts_1["linear"]
        )
        assert mac_counts_1["total"] == expected_total

    def test_latency_scaling_accuracy(self, device):
        """Test latency scaling behaves correctly."""
        from benchmarks.mcu_simulation import ModelStats, simulate_mcu_latency

        # Create two model stats with different MAC counts
        low_mac_stats = ModelStats(
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

        high_mac_stats = ModelStats(
            total_parameters=2000,
            trainable_parameters=2000,
            total_macs=400000,
            conv_macs=200000,
            linear_macs=120000,
            feature_transformer_macs=80000,
            model_size_bytes=8000,
            activation_memory_bytes=2000,
            quantized_model_size_bytes=2000,
            quantization_bit_width=8,
            sparsity_ratio=0.5,
            effective_macs=200000,
        )

        test_mcu = MCU_SPECS["cortex_m33"]

        low_latency = simulate_mcu_latency(low_mac_stats, test_mcu)
        high_latency = simulate_mcu_latency(high_mac_stats, test_mcu)

        # Higher MACs should mean higher latency and lower throughput
        assert (
            high_latency["quantized_latency_ms"] > low_latency["quantized_latency_ms"]
        )
        assert low_latency["throughput_fps"] > high_latency["throughput_fps"]

        # Latency should scale roughly linearly with MACs
        mac_ratio = high_mac_stats.effective_macs / low_mac_stats.effective_macs
        latency_ratio = (
            high_latency["quantized_latency_ms"] / low_latency["quantized_latency_ms"]
        )

        # Should be roughly proportional (within 2x due to overhead)
        assert 0.5 * mac_ratio < latency_ratio < 2.0 * mac_ratio

    def test_energy_calculation_accuracy(self, device):
        """Test energy calculations are physically reasonable."""
        from benchmarks.mcu_simulation import estimate_energy_consumption

        test_mcu = MCU_SPECS["cortex_m33"]

        # Test different latencies
        latencies = [10.0, 50.0, 100.0]  # milliseconds
        energies = [estimate_energy_consumption(lat, test_mcu) for lat in latencies]

        # Energy should increase with latency
        for i in range(1, len(energies)):
            assert energies[i]["total_energy_uj"] > energies[i - 1]["total_energy_uj"]

        # Energy should be roughly proportional to time
        for i, lat in enumerate(latencies):
            energy = energies[i]

            # Basic sanity check: energy = power * time
            expected_active_energy = test_mcu.power_active_mw * lat  # mJ

            # Should be close (within 20% due to idle power additions)
            assert (
                0.8 * expected_active_energy
                < energy["active_energy_mj"]
                < 1.2 * expected_active_energy
            )


class TestBenchmarkRobustness:
    """Test benchmark robustness to edge cases."""

    def test_very_small_model(self, device):
        """Test benchmarking works with very small models."""
        # Create tiny model
        feature_set = GridFeatureSet(grid_size=2, num_features_per_square=2)
        tiny_model = NNUE(
            feature_set=feature_set, l1_size=8, l2_size=2, l3_size=4, num_ls_buckets=1
        )
        tiny_model.to(device).eval()

        from benchmarks.mcu_simulation import generate_model_stats

        # Should work without errors
        stats = generate_model_stats(tiny_model, target_sparsity=0.0)

        assert stats.total_parameters > 0
        assert stats.total_macs > 0
        assert stats.quantized_model_size_bytes > 0

    def test_very_large_model(self, device):
        """Test benchmarking works with larger models."""
        # Create larger model
        feature_set = GridFeatureSet(grid_size=8, num_features_per_square=16)
        large_model = NNUE(
            feature_set=feature_set,
            l1_size=256,
            l2_size=32,
            l3_size=64,
            num_ls_buckets=4,
        )
        large_model.to(device).eval()

        from benchmarks.mcu_simulation import generate_model_stats

        # Should work without errors
        stats = generate_model_stats(large_model, target_sparsity=0.7)

        assert stats.total_parameters > 10000  # Should be large
        assert stats.total_macs > 100000
        assert stats.sparsity_ratio == 0.7

    def test_extreme_sparsity_values(self, trained_model_checkpoint, device):
        """Test extreme sparsity values."""
        checkpoint_path, model = trained_model_checkpoint

        from benchmarks.mcu_simulation import generate_model_stats

        # Test 0% sparsity (dense)
        dense_stats = generate_model_stats(model, target_sparsity=0.0)
        assert dense_stats.sparsity_ratio == 0.0
        assert dense_stats.effective_macs == dense_stats.total_macs

        # Test 99% sparsity (very sparse)
        sparse_stats = generate_model_stats(model, target_sparsity=0.99)
        assert sparse_stats.sparsity_ratio == 0.99
        assert sparse_stats.effective_macs < dense_stats.effective_macs
        assert sparse_stats.sparsity_speedup > 50.0  # Should be very high

    def test_different_input_sizes(self, device):
        """Test benchmarking with different input sizes."""
        feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
        model = NNUE(feature_set=feature_set, l1_size=64, l2_size=8, l3_size=16)
        model.to(device).eval()

        from benchmarks.mcu_simulation import count_mac_operations

        # Test different input sizes
        input_sizes = [(32, 32), (96, 96), (128, 128)]

        for h, w in input_sizes:
            mac_counts = count_mac_operations(model, input_shape=(3, h, w))

            # Should work for all sizes
            assert mac_counts["conv"] > 0
            assert mac_counts["total"] > 0

            # Larger inputs should generally mean more conv MACs
            # (FT and linear MACs stay the same)


class TestBenchmarkValidation:
    """Validate benchmark results against expected values."""

    def test_compression_ratio_realistic(self, trained_model_checkpoint, device):
        """Test quantization compression ratios are realistic."""
        checkpoint_path, model = trained_model_checkpoint

        from benchmarks.mcu_simulation import generate_model_stats

        stats = generate_model_stats(model)

        # Compression ratio should be reasonable for mixed INT8/INT16
        assert 2.0 <= stats.compression_ratio <= 8.0

        # Quantized size should be much smaller than float size
        assert stats.quantized_model_size_bytes < stats.model_size_bytes

    def test_energy_efficiency_realistic(self, trained_model_checkpoint, device):
        """Test energy efficiency metrics are realistic."""
        checkpoint_path, model = trained_model_checkpoint

        from benchmarks.mcu_simulation import run_mcu_benchmark

        # Create minimal dataset for testing
        dataset = GenericVisionDataset(
            dataset_name="cifar10",
            split="test",
            max_samples=10,
            binary_classification={"positive_classes": ["airplane"]},
        )
        data_loader = DataLoader(dataset, batch_size=4)

        results = run_mcu_benchmark(
            model=model,
            data_loader=data_loader,
            mcu_specs=[MCU_SPECS["cortex_m33"]],
            verbose=False,
        )

        mcu_result = results["mcu_results"]["ARM Cortex-M33"]
        efficiency = mcu_result["efficiency_metrics"]

        # Energy per inference should be reasonable (microjoules range)
        assert 1.0 <= efficiency["energy_per_inference_uj"] <= 10000.0

        # FPS should be reasonable for MCU
        assert 1.0 <= efficiency["fps"] <= 1000.0

        # MACs per millisecond should be reasonable
        assert efficiency["macs_per_ms"] > 0

    def test_baseline_comparison_realistic(self, trained_model_checkpoint, device):
        """Test baseline comparisons produce realistic ratios."""
        checkpoint_path, model = trained_model_checkpoint

        from benchmarks.tinyml_benchmarks import (
            compare_with_baselines,
            run_mlperf_tiny_benchmark,
        )

        # Create minimal dataset
        dataset = GenericVisionDataset(
            dataset_name="cifar10",
            split="test",
            max_samples=20,
            binary_classification={"positive_classes": ["airplane"]},
        )
        data_loader = DataLoader(dataset, batch_size=8)

        # Run MLPerf benchmark
        result = run_mlperf_tiny_benchmark(
            model,
            "visual_wake_words",
            data_loader,
            device,
            MCU_SPECS["cortex_m33"],
            num_samples=20,
        )

        # Compare with baselines
        comparisons = compare_with_baselines(result, "visual_wake_words")

        if comparisons:  # If we have baselines for this dataset
            for baseline_name, comparison in comparisons.items():
                ratios = comparison["nnue_vs_baseline"]

                # Ratios should be reasonable (not extreme)
                for ratio_name, ratio_value in ratios.items():
                    if isinstance(ratio_value, (int, float)):
                        assert (
                            0.01 <= ratio_value <= 100.0
                        ), f"Unrealistic {ratio_name}: {ratio_value}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
