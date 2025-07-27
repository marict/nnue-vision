"""Tests for weight decay functionality in NNUE and EtinyNet models."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from config.config_loader import load_config
from etinynet_adapter import EtinyNetAdapter
from model import NNUE, EtinyNet, GridFeatureSet, LossParams
from nnue_adapter import NNUEAdapter


class TestWeightDecayConfiguration:
    """Test weight decay parameter configuration and usage."""

    def test_etinynet_weight_decay_parameter_acceptance(self):
        """Test that EtinyNet accepts and stores weight_decay parameter."""
        weight_decay_values = [1e-5, 1e-4, 2e-4, 5e-4, 1e-3]

        for weight_decay in weight_decay_values:
            model = EtinyNet(
                variant="0.75", num_classes=10, input_size=32, weight_decay=weight_decay
            )

            assert hasattr(
                model, "weight_decay"
            ), "Model should have weight_decay attribute"
            assert (
                model.weight_decay == weight_decay
            ), f"Expected weight_decay {weight_decay}, got {model.weight_decay}"

    def test_nnue_weight_decay_parameter_acceptance(self):
        """Test that NNUE accepts and stores weight_decay parameter."""
        weight_decay_values = [1e-5, 1e-4, 2e-4, 5e-4, 1e-3]

        for weight_decay in weight_decay_values:
            model = NNUE(
                feature_set=GridFeatureSet(grid_size=8, num_features_per_square=12),
                l1_size=128,
                l2_size=16,
                l3_size=32,
                num_ls_buckets=2,
                weight_decay=weight_decay,
            )

            assert hasattr(
                model, "weight_decay"
            ), "Model should have weight_decay attribute"
            assert (
                model.weight_decay == weight_decay
            ), f"Expected weight_decay {weight_decay}, got {model.weight_decay}"

    def test_etinynet_default_weight_decay(self):
        """Test that EtinyNet uses default weight decay when not specified."""
        model = EtinyNet(variant="0.75", num_classes=10, input_size=32)

        assert hasattr(
            model, "weight_decay"
        ), "Model should have weight_decay attribute"
        assert (
            model.weight_decay == 1e-4
        ), f"Expected default weight_decay 1e-4, got {model.weight_decay}"

    def test_nnue_default_weight_decay(self):
        """Test that NNUE uses default weight decay when not specified."""
        model = NNUE(
            feature_set=GridFeatureSet(grid_size=8, num_features_per_square=12),
            l1_size=128,
            l2_size=16,
            l3_size=32,
            num_ls_buckets=2,
        )

        assert hasattr(
            model, "weight_decay"
        ), "Model should have weight_decay attribute"
        assert (
            model.weight_decay == 5e-4
        ), f"Expected default weight_decay 5e-4, got {model.weight_decay}"


class TestOptimizerWeightDecayUsage:
    """Test that weight decay is properly used in optimizers."""

    def test_etinynet_optimizer_uses_weight_decay(self):
        """Test that EtinyNet optimizer uses the configured weight_decay."""
        weight_decay = 3e-4
        model = EtinyNet(
            variant="0.75", num_classes=10, input_size=32, weight_decay=weight_decay
        )

        optimizers, schedulers = model.configure_optimizers()

        assert len(optimizers) >= 1, "Should have at least one optimizer"
        optimizer = optimizers[0]

        # Check that weight_decay is set correctly in optimizer
        assert hasattr(optimizer, "param_groups"), "Optimizer should have param_groups"

        # For SGD optimizer, all param groups should have the same weight_decay
        for param_group in optimizer.param_groups:
            assert "weight_decay" in param_group, "Param group should have weight_decay"
            assert (
                param_group["weight_decay"] == weight_decay
            ), f"Expected weight_decay {weight_decay}, got {param_group['weight_decay']}"

    def test_nnue_optimizer_uses_weight_decay(self):
        """Test that NNUE optimizer uses the configured weight_decay for appropriate parameters."""
        weight_decay = 3e-4
        model = NNUE(
            feature_set=GridFeatureSet(grid_size=8, num_features_per_square=12),
            l1_size=128,
            l2_size=16,
            l3_size=32,
            num_ls_buckets=2,
            weight_decay=weight_decay,
        )

        optimizers, schedulers = model.configure_optimizers()

        assert len(optimizers) >= 1, "Should have at least one optimizer"
        optimizer = optimizers[0]

        # Check that weight_decay is set correctly in optimizer param groups
        assert hasattr(optimizer, "param_groups"), "Optimizer should have param_groups"
        assert (
            len(optimizer.param_groups) == 2
        ), "NNUE should have 2 param groups (weight decay and no weight decay)"

        # First param group should have weight decay
        weight_decay_group = optimizer.param_groups[0]
        assert (
            weight_decay_group["weight_decay"] == weight_decay
        ), f"Expected weight_decay {weight_decay}, got {weight_decay_group['weight_decay']}"

        # Second param group should have no weight decay
        no_weight_decay_group = optimizer.param_groups[1]
        assert (
            no_weight_decay_group["weight_decay"] == 0.0
        ), f"Expected no weight_decay (0.0), got {no_weight_decay_group['weight_decay']}"

    def test_etinynet_different_weight_decay_values_in_optimizer(self):
        """Test that different weight_decay values are correctly reflected in optimizer."""
        weight_decay_values = [1e-5, 1e-4, 5e-4, 1e-3]

        for weight_decay in weight_decay_values:
            model = EtinyNet(
                variant="0.75", num_classes=10, input_size=32, weight_decay=weight_decay
            )

            optimizers, _ = model.configure_optimizers()
            optimizer = optimizers[0]

            for param_group in optimizer.param_groups:
                assert (
                    param_group["weight_decay"] == weight_decay
                ), f"Expected weight_decay {weight_decay}, got {param_group['weight_decay']}"

    def test_nnue_different_weight_decay_values_in_optimizer(self):
        """Test that different weight_decay values are correctly reflected in NNUE optimizer."""
        weight_decay_values = [1e-5, 1e-4, 5e-4, 1e-3]

        for weight_decay in weight_decay_values:
            model = NNUE(
                feature_set=GridFeatureSet(grid_size=8, num_features_per_square=12),
                l1_size=128,
                l2_size=16,
                l3_size=32,
                num_ls_buckets=2,
                weight_decay=weight_decay,
            )

            optimizers, _ = model.configure_optimizers()
            optimizer = optimizers[0]

            # Check weight decay param group
            weight_decay_group = optimizer.param_groups[0]
            assert (
                weight_decay_group["weight_decay"] == weight_decay
            ), f"Expected weight_decay {weight_decay}, got {weight_decay_group['weight_decay']}"


class TestAdapterWeightDecayIntegration:
    """Test that adapters correctly pass weight_decay from config to models."""

    def test_etinynet_adapter_passes_weight_decay(self):
        """Test that EtinyNetAdapter passes weight_decay from config to model."""
        weight_decay = 3e-4

        # Create a mock config
        config = SimpleNamespace()
        config.etinynet_variant = "0.75"
        config.dataset_name = "cifar10"
        config.input_size = (32, 32)
        config.use_asq = False
        config.asq_bits = 4
        config.learning_rate = 0.1
        config.max_epochs = 200
        config.weight_decay = weight_decay

        adapter = EtinyNetAdapter()
        model = adapter.create_model(config)

        # The adapter wraps EtinyNet, so we need to access the underlying model
        assert hasattr(
            model, "weight_decay"
        ), "Model should have weight_decay attribute"
        assert (
            model.weight_decay == weight_decay
        ), f"Expected weight_decay {weight_decay}, got {model.weight_decay}"

    def test_nnue_adapter_passes_weight_decay(self):
        """Test that NNUEAdapter passes weight_decay from config to model."""
        weight_decay = 3e-4

        # Create a mock config
        config = SimpleNamespace()
        config.max_epochs = 50
        config.learning_rate = 1e-3
        config.start_lambda = 1.0
        config.end_lambda = 1.0
        config.num_ls_buckets = 8
        config.visual_threshold = 0.0
        config.num_classes = 1
        config.weight_decay = weight_decay

        adapter = NNUEAdapter()
        wrapped_model = adapter.create_model(config)

        # The adapter returns NNUEWrapper, so access the underlying NNUE model
        assert hasattr(
            wrapped_model, "nnue"
        ), "Wrapped model should have nnue attribute"
        nnue_model = wrapped_model.nnue

        assert hasattr(
            nnue_model, "weight_decay"
        ), "NNUE model should have weight_decay attribute"
        assert (
            nnue_model.weight_decay == weight_decay
        ), f"Expected weight_decay {weight_decay}, got {nnue_model.weight_decay}"

    def test_etinynet_adapter_uses_default_weight_decay(self):
        """Test that EtinyNetAdapter uses default weight_decay when not specified in config."""
        # Create a mock config without weight_decay
        config = SimpleNamespace()
        config.etinynet_variant = "0.75"
        config.dataset_name = "cifar10"
        config.input_size = (32, 32)
        config.use_asq = False
        config.asq_bits = 4
        config.learning_rate = 0.1
        config.max_epochs = 200
        # No weight_decay attribute

        adapter = EtinyNetAdapter()
        model = adapter.create_model(config)

        assert hasattr(
            model, "weight_decay"
        ), "Model should have weight_decay attribute"
        assert (
            model.weight_decay == 1e-4
        ), f"Expected default weight_decay 1e-4, got {model.weight_decay}"

    def test_nnue_adapter_uses_default_weight_decay(self):
        """Test that NNUEAdapter uses default weight_decay when not specified in config."""
        # Create a mock config without weight_decay
        config = SimpleNamespace()
        config.max_epochs = 50
        config.learning_rate = 1e-3
        config.start_lambda = 1.0
        config.end_lambda = 1.0
        config.num_ls_buckets = 8
        config.visual_threshold = 0.0
        config.num_classes = 1
        # No weight_decay attribute

        adapter = NNUEAdapter()
        wrapped_model = adapter.create_model(config)
        nnue_model = wrapped_model.nnue

        assert hasattr(
            nnue_model, "weight_decay"
        ), "NNUE model should have weight_decay attribute"
        assert (
            nnue_model.weight_decay == 5e-4
        ), f"Expected default weight_decay 5e-4, got {nnue_model.weight_decay}"


class TestConfigurationFileWeightDecay:
    """Test weight decay configuration loading from config files."""

    def test_etinynet_config_contains_weight_decay(self):
        """Test that EtinyNet configuration files contain weight_decay parameter."""
        # Test main config
        config = load_config("config/train_etinynet.py")
        assert hasattr(
            config, "weight_decay"
        ), "EtinyNet config should have weight_decay"
        assert isinstance(
            config.weight_decay, (int, float)
        ), "weight_decay should be numeric"
        assert config.weight_decay > 0, "weight_decay should be positive"

        # Test default config
        default_config = load_config("config/train_etinynet_default.py")
        assert hasattr(
            default_config, "weight_decay"
        ), "EtinyNet default config should have weight_decay"
        assert isinstance(
            default_config.weight_decay, (int, float)
        ), "weight_decay should be numeric"
        assert default_config.weight_decay > 0, "weight_decay should be positive"

    def test_nnue_config_contains_weight_decay(self):
        """Test that NNUE configuration files contain weight_decay parameter."""
        # Test main config
        config = load_config("config/train_nnue.py")
        assert hasattr(config, "weight_decay"), "NNUE config should have weight_decay"
        assert isinstance(
            config.weight_decay, (int, float)
        ), "weight_decay should be numeric"
        assert config.weight_decay > 0, "weight_decay should be positive"

        # Test default config
        default_config = load_config("config/train_nnue_default.py")
        assert hasattr(
            default_config, "weight_decay"
        ), "NNUE default config should have weight_decay"
        assert isinstance(
            default_config.weight_decay, (int, float)
        ), "weight_decay should be numeric"
        assert default_config.weight_decay > 0, "weight_decay should be positive"

    def test_weight_decay_values_are_reasonable(self):
        """Test that weight_decay values in configs are in reasonable ranges."""
        # Common weight decay values are typically between 1e-5 and 1e-2
        min_reasonable = 1e-6
        max_reasonable = 1e-2

        configs_to_test = [
            "config/train_etinynet.py",
            "config/train_etinynet_default.py",
            "config/train_nnue.py",
            "config/train_nnue_default.py",
        ]

        for config_path in configs_to_test:
            config = load_config(config_path)
            weight_decay = config.weight_decay

            assert (
                min_reasonable <= weight_decay <= max_reasonable
            ), f"weight_decay {weight_decay} in {config_path} is outside reasonable range [{min_reasonable}, {max_reasonable}]"


class TestWeightDecayBackwardCompatibility:
    """Test that weight decay implementation maintains backward compatibility."""

    def test_etinynet_works_without_weight_decay_parameter(self):
        """Test that EtinyNet still works when weight_decay is not provided (backward compatibility)."""
        # This should use the default value
        model = EtinyNet(variant="0.75", num_classes=10, input_size=32)

        # Should have default weight_decay
        assert hasattr(model, "weight_decay")
        assert model.weight_decay == 1e-4

        # Should be able to configure optimizers
        optimizers, schedulers = model.configure_optimizers()
        assert len(optimizers) >= 1

        # Optimizer should use the default weight_decay
        optimizer = optimizers[0]
        for param_group in optimizer.param_groups:
            assert param_group["weight_decay"] == 1e-4

    def test_nnue_works_without_weight_decay_parameter(self):
        """Test that NNUE still works when weight_decay is not provided (backward compatibility)."""
        # This should use the default value
        model = NNUE(
            feature_set=GridFeatureSet(grid_size=8, num_features_per_square=12),
            l1_size=128,
            l2_size=16,
            l3_size=32,
            num_ls_buckets=2,
        )

        # Should have default weight_decay
        assert hasattr(model, "weight_decay")
        assert model.weight_decay == 5e-4

        # Should be able to configure optimizers
        optimizers, schedulers = model.configure_optimizers()
        assert len(optimizers) >= 1

        # Optimizer should use the default weight_decay for weight decay group
        optimizer = optimizers[0]
        weight_decay_group = optimizer.param_groups[0]
        assert weight_decay_group["weight_decay"] == 5e-4
