"""
Tests for NNUE model forward and backward passes.
"""

import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from model import NNUE, FeatureTransformer, GridFeatureSet, LayerStacks
from tests.conftest import (assert_gradients_exist, assert_gradients_nonzero,
                            assert_quantized_weights_valid,
                            assert_sparse_features_valid, assert_tensor_shape)


class TestGridFeatureSet:
    """Test the GridFeatureSet dataclass."""

    def test_default_feature_set(self):
        """Test default GridFeatureSet values."""
        feature_set = GridFeatureSet()
        assert feature_set.grid_size == 8
        assert feature_set.num_features_per_square == 12
        assert feature_set.num_features == 8 * 8 * 12
        assert feature_set.name == "Grid8x8_12"

    def test_custom_feature_set(self):
        """Test custom GridFeatureSet values."""
        feature_set = GridFeatureSet(grid_size=16, num_features_per_square=24)
        assert feature_set.grid_size == 16
        assert feature_set.num_features_per_square == 24
        assert feature_set.num_features == 16 * 16 * 24
        assert feature_set.name == "Grid16x16_24"

    def test_feature_count_calculation(self):
        """Test that feature count is calculated correctly."""
        test_cases = [
            (4, 6, 4 * 4 * 6),
            (8, 12, 8 * 8 * 12),
            (16, 8, 16 * 16 * 8),
        ]

        for grid_size, features_per_square, expected_total in test_cases:
            feature_set = GridFeatureSet(grid_size, features_per_square)
            assert feature_set.num_features == expected_total


class TestFeatureTransformer:
    """Test the FeatureTransformer component."""

    def test_feature_transformer_initialization(self, _grid_feature_set):
        """Test FeatureTransformer initialization."""
        feature_transformer = FeatureTransformer(
            num_features=_grid_feature_set.num_features, output_size=256
        )

        assert feature_transformer.num_features == _grid_feature_set.num_features
        assert feature_transformer.output_size == 256

        # Check weight and bias dimensions
        assert feature_transformer.weight.shape == (256, _grid_feature_set.num_features)
        assert feature_transformer.bias.shape == (256,)

    def test_sparse_feature_forward(self, _grid_feature_set, _device):
        """Test FeatureTransformer forward pass with sparse features."""
        feature_transformer = FeatureTransformer(
            num_features=_grid_feature_set.num_features, output_size=32
        )
        feature_transformer.to(_device)

        batch_size = 4
        max_active_features = 8

        # Create sparse feature indices (some active, some -1 for padding)
        feature_indices = torch.randint(
            0,
            _grid_feature_set.num_features,
            (batch_size, max_active_features),
            device=_device,
        )
        mask = torch.rand(batch_size, max_active_features) < 0.7
        feature_indices = feature_indices * mask.to(_device) + (-1) * (~mask).to(
            _device
        )

        output = feature_transformer(feature_indices)

        assert_tensor_shape(output, (batch_size, 32))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_empty_features(self, _grid_feature_set, _device):
        """Test FeatureTransformer with completely empty feature sets."""
        feature_transformer = FeatureTransformer(
            num_features=_grid_feature_set.num_features, output_size=16
        )
        feature_transformer.to(_device)

        batch_size = 2
        max_active_features = 4

        # All features are -1 (empty)
        feature_indices = torch.full(
            (batch_size, max_active_features), -1, device=_device
        )

        output = feature_transformer(feature_indices)

        assert_tensor_shape(output, (batch_size, 16))

        # With no active features, output should be just the bias
        expected_output = feature_transformer.bias.unsqueeze(0).expand(batch_size, -1)
        assert torch.allclose(output, expected_output, atol=1e-6)

    def test_single_feature_per_sample(self, _grid_feature_set, _device):
        """Test FeatureTransformer with single active feature per sample."""
        feature_transformer = FeatureTransformer(
            num_features=_grid_feature_set.num_features, output_size=8
        )
        feature_transformer.to(_device)

        batch_size = 3
        max_active_features = 1

        # Each sample has exactly one active feature
        feature_indices = torch.randint(
            0,
            _grid_feature_set.num_features,
            (batch_size, max_active_features),
            device=_device,
        )

        output = feature_transformer(feature_indices)

        assert_tensor_shape(output, (batch_size, 8))
        assert not torch.isnan(output).any()

        # Verify that each output is bias + corresponding weight column
        for i in range(batch_size):
            feature_idx = feature_indices[i, 0].item()
            expected = (
                feature_transformer.bias + feature_transformer.weight[:, feature_idx]
            )
            assert torch.allclose(output[i], expected, atol=1e-6)


class TestLayerStacks:
    """Test the LayerStacks (bucketed NNUE layers)."""

    def test_layer_stacks_initialization(self):
        """Test LayerStacks initialization."""
        num_buckets = 4
        layer_stacks = LayerStacks(num_buckets)

        assert layer_stacks.count == num_buckets

        # Check layer dimensions
        assert layer_stacks.l1.in_features == 3072  # L1
        assert (
            layer_stacks.l1.out_features == (15 + 1) * num_buckets
        )  # (L2 + 1) * count
        assert layer_stacks.l1_fact.in_features == 3072
        assert layer_stacks.l1_fact.out_features == 15 + 1
        assert layer_stacks.l2.in_features == 15 * 2  # L2 * 2
        assert layer_stacks.l2.out_features == 32 * num_buckets  # L3 * count
        assert layer_stacks.output.in_features == 32  # L3
        assert layer_stacks.output.out_features == 1 * num_buckets

    def test_layer_stacks_forward(self, _device):
        """Test LayerStacks forward pass."""
        num_buckets = 2
        layer_stacks = LayerStacks(num_buckets)
        layer_stacks.to(_device)

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3072, device=_device)  # L1 size
        bucket_indices = torch.randint(0, num_buckets, (batch_size,), device=_device)

        output = layer_stacks(input_tensor, bucket_indices)

        assert_tensor_shape(output, (batch_size, 1))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_bucket_selection(self, _device):
        """Test that different bucket indices produce different outputs."""
        num_buckets = 3
        layer_stacks = LayerStacks(num_buckets)
        layer_stacks.to(_device)

        # Use the same input but different bucket indices
        input_tensor = torch.randn(1, 3072, device=_device)
        bucket_0 = torch.tensor([0], device=_device)
        bucket_1 = torch.tensor([1], device=_device)
        bucket_2 = torch.tensor([2], device=_device)

        output_0 = layer_stacks(input_tensor, bucket_0)
        output_1 = layer_stacks(input_tensor, bucket_1)
        output_2 = layer_stacks(input_tensor, bucket_2)

        # Different buckets should produce different outputs
        assert not torch.allclose(output_0, output_1, atol=1e-6)
        assert not torch.allclose(output_1, output_2, atol=1e-6)
        assert not torch.allclose(output_0, output_2, atol=1e-6)

        # But same bucket should produce same output
        output_0_repeat = layer_stacks(input_tensor, bucket_0)
        assert torch.allclose(output_0, output_0_repeat, atol=1e-6)

    def test_coalesced_layer_stacks(self):
        """Test LayerStacks with coalesced=True."""
        num_buckets = 2
        layer_stacks = LayerStacks(num_buckets, coalesced=True)

        # Should have the same structure as non-coalesced
        assert layer_stacks.count == num_buckets
        assert layer_stacks.l1.in_features == 3072
        assert layer_stacks.l1.out_features == (15 + 1) * num_buckets

        # Should still function (basic smoke test)
        input_tensor = torch.randn(1, 3072)
        bucket_indices = torch.tensor([0])
        output = layer_stacks(input_tensor, bucket_indices)

        assert_tensor_shape(output, (1, 1))
        assert not torch.isnan(output).any()


class TestNNUEArchitecture:
    """Test the main NNUE model architecture."""

    def test_nnue_initialization(self):
        """Test NNUE model initialization."""
        # Test with custom feature set and layer sizes
        feature_set = GridFeatureSet(grid_size=8, num_features_per_square=12)
        model = NNUE(
            feature_set=feature_set,
            l1_size=256,
            l2_size=8,
            l3_size=16,
            num_ls_buckets=4,
        )

        assert model.feature_set.grid_size == 8
        assert model.feature_set.num_features_per_square == 12
        assert model.feature_set.num_features == 8 * 8 * 12  # 768 features
        assert model.l1_size == 256
        assert model.l2_size == 8
        assert model.l3_size == 16
        assert model.num_ls_buckets == 4
        assert isinstance(model.input, FeatureTransformer)
        assert isinstance(model.layer_stacks, LayerStacks)

        # Check conv layer
        assert model.conv.in_channels == 3
        assert model.conv.out_channels == 12  # matches num_features_per_square
        assert model.conv.kernel_size == (3, 3)

    def test_nnue_default_initialization(self):
        """Test NNUE model initialization with defaults."""
        # Test with default parameters (large model)
        model = NNUE(num_ls_buckets=2)

        assert model.feature_set.grid_size == 32
        assert model.feature_set.num_features_per_square == 64
        assert model.feature_set.num_features == 32 * 32 * 64  # 65536 features
        assert model.l1_size == 3072  # Default
        assert model.l2_size == 15  # Default
        assert model.l3_size == 32  # Default
        assert model.num_ls_buckets == 2

    def test_nnue_tiny_initialization(self):
        """Test NNUE model initialization with tiny architecture."""
        feature_set = GridFeatureSet(grid_size=4, num_features_per_square=8)
        model = NNUE(
            feature_set=feature_set, l1_size=32, l2_size=4, l3_size=4, num_ls_buckets=2
        )

        assert model.feature_set.grid_size == 4
        assert model.feature_set.num_features_per_square == 8
        assert model.feature_set.num_features == 4 * 4 * 8  # 128 features
        assert model.l1_size == 32
        assert model.l2_size == 4
        assert model.l3_size == 4
        assert model.num_ls_buckets == 2

        # Check conv layer matches feature set
        assert model.conv.out_channels == 8  # matches num_features_per_square

    def test_nnue_forward_pass(self, _nnue_model, _sample_image_batch, _device):
        """Test NNUE forward pass with images."""
        _nnue_model.to(_device)
        _nnue_model.eval()

        images, _, _, layer_stack_indices = _sample_image_batch

        with torch.no_grad():
            output = _nnue_model(images, layer_stack_indices)

        batch_size = images.shape[0]
        assert_tensor_shape(output, (batch_size, 1))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_nnue_different_batch_sizes(self, _small_nnue_model, _device):
        """Test NNUE with different batch sizes."""
        _small_nnue_model.to(_device)
        _small_nnue_model.eval()

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            images = torch.randn(batch_size, 3, 96, 96, device=_device)
            layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)

            with torch.no_grad():
                output = _small_nnue_model(images, layer_stack_indices)

            assert_tensor_shape(output, (batch_size, 1))
            assert not torch.isnan(output).any()

    def test_conv_layer_output_shape(self, _tiny_nnue_model, _device):
        """Test that conv layer produces correct feature grid shape."""
        _tiny_nnue_model.to(_device)
        _tiny_nnue_model.eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, 96, 96, device=_device)

        with torch.no_grad():
            # Access conv layer output through forward hooks
            conv_outputs = []

            def hook(module, input, output):
                conv_outputs.append(output)

            handle = _tiny_nnue_model.conv.register_forward_hook(hook)

            try:
                layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)
                _ = _tiny_nnue_model(images, layer_stack_indices)

                assert len(conv_outputs) == 1
                conv_output = conv_outputs[0]

                # Should be (batch_size, num_features_per_square, grid_size, grid_size)
                expected_shape = (batch_size, 8, 4, 4)  # For tiny model
                assert_tensor_shape(conv_output, expected_shape)

            finally:
                handle.remove()

    def test_binary_feature_conversion(self, _tiny_nnue_model, _device):
        """Test that conv outputs are properly converted to binary features."""
        _tiny_nnue_model.to(_device)
        _tiny_nnue_model.eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, 96, 96, device=_device)
        layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)

        with torch.no_grad():
            # Store intermediate outputs
            conv_output = None
            sparse_features = None

            def conv_hook(module, input, output):
                nonlocal conv_output
                conv_output = output

            def ft_hook(module, input, output):
                nonlocal sparse_features
                sparse_features = input[0]  # Get the sparse feature input

            conv_handle = _tiny_nnue_model.conv.register_forward_hook(conv_hook)
            ft_handle = _tiny_nnue_model.input.register_forward_hook(ft_hook)

            try:
                _ = _tiny_nnue_model(images, layer_stack_indices)

                # Check that conv output was properly converted to sparse features
                assert conv_output is not None
                assert sparse_features is not None

                # Sparse features should be 2D: (batch_size, max_active_features)
                assert len(sparse_features.shape) == 2
                assert sparse_features.shape[0] == batch_size

                # All feature indices should be either valid (>= 0) or padding (-1)
                valid_mask = sparse_features >= 0
                padding_mask = sparse_features == -1
                assert torch.all(valid_mask | padding_mask)

                # Valid indices should be within feature range
                num_features = _tiny_nnue_model.feature_set.num_features
                assert torch.all(sparse_features[valid_mask] < num_features)

            finally:
                conv_handle.remove()
                ft_handle.remove()

    def test_weight_clipping(self, _tiny_nnue_model):
        """Test weight clipping functionality."""
        # Set some weights to extreme values
        with torch.no_grad():
            _tiny_nnue_model.input.weight.data.fill_(10.0)  # Too large
            _tiny_nnue_model.layer_stacks.l1.weight.data.fill_(-5.0)  # Too negative

        # Apply clipping
        _tiny_nnue_model.clip_weights()

        # Check that weights are now within expected ranges
        # Feature transformer weights should be clipped
        assert torch.all(_tiny_nnue_model.input.weight.data >= -2.0)
        assert torch.all(_tiny_nnue_model.input.weight.data <= 2.0)

        # Layer stack weights should be clipped
        assert torch.all(_tiny_nnue_model.layer_stacks.l1.weight.data >= -2.0)
        assert torch.all(_tiny_nnue_model.layer_stacks.l1.weight.data <= 2.0)

    def test_configurable_feature_sets(self, _device):
        """Test NNUE with different feature set configurations."""
        test_configs = [
            (GridFeatureSet(grid_size=4, num_features_per_square=4), 32, 4, 8),
            (GridFeatureSet(grid_size=8, num_features_per_square=8), 64, 8, 16),
            (GridFeatureSet(grid_size=6, num_features_per_square=12), 128, 16, 32),
        ]

        for feature_set, l1_size, l2_size, l3_size in test_configs:
            model = NNUE(
                feature_set=feature_set,
                l1_size=l1_size,
                l2_size=l2_size,
                l3_size=l3_size,
                num_ls_buckets=2,
            )
            model.to(_device)
            model.eval()

            # Test forward pass
            batch_size = 2
            images = torch.randn(batch_size, 3, 96, 96, device=_device)
            layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)

            with torch.no_grad():
                output = model(images, layer_stack_indices)

            assert_tensor_shape(output, (batch_size, 1))
            assert not torch.isnan(output).any()


class TestNNUEForwardBackward:
    """Test NNUE forward and backward passes."""

    def test_forward_backward_basic(
        self, _small_nnue_model, _small_image_batch, _device
    ):
        """Test basic forward and backward passes."""
        _small_nnue_model.to(_device)
        _small_nnue_model.train()

        images, targets, scores, layer_stack_indices = _small_image_batch

        # Forward pass
        output = _small_nnue_model(images, layer_stack_indices)

        # Simple loss for testing
        loss = torch.mean((output - targets) ** 2)

        # Backward pass
        loss.backward()

        # Check gradients exist and are reasonable
        assert_gradients_exist(_small_nnue_model)
        assert_gradients_nonzero(_small_nnue_model)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_gradient_shapes(self, _small_nnue_model, _small_image_batch, _device):
        """Test that gradients have correct shapes."""
        _small_nnue_model.to(_device)
        _small_nnue_model.train()

        images, targets, scores, layer_stack_indices = _small_image_batch

        output = _small_nnue_model(images, layer_stack_indices)
        loss = torch.mean((output - targets) ** 2)
        loss.backward()

        for name, param in _small_nnue_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert (
                    param.grad.shape == param.shape
                ), f"Gradient shape mismatch for {name}"

    def test_conv_layer_gradients(self, _small_nnue_model, _device):
        """Test that conv layer gradients are computed correctly."""
        _small_nnue_model.to(_device)
        _small_nnue_model.train()

        batch_size = 2
        images = torch.randn(batch_size, 3, 96, 96, device=_device)
        targets = torch.rand(batch_size, 1, device=_device)
        layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)

        output = _small_nnue_model(images, layer_stack_indices)
        loss = torch.mean((output - targets) ** 2)
        loss.backward()

        # Conv layer should have gradients
        assert _small_nnue_model.conv.weight.grad is not None
        assert _small_nnue_model.conv.bias.grad is not None

        # Gradients should have same shape as parameters
        assert (
            _small_nnue_model.conv.weight.grad.shape
            == _small_nnue_model.conv.weight.shape
        )
        assert (
            _small_nnue_model.conv.bias.grad.shape == _small_nnue_model.conv.bias.shape
        )

        # Gradients should be non-zero (assuming loss is meaningful)
        assert torch.any(_small_nnue_model.conv.weight.grad != 0)
        assert torch.any(_small_nnue_model.conv.bias.grad != 0)

    def test_gradient_accumulation(self, _small_nnue_model, _device):
        """Test gradient accumulation across multiple backward passes."""
        _small_nnue_model.to(_device)
        _small_nnue_model.train()

        batch_size = 2
        images = torch.randn(batch_size, 3, 96, 96, device=_device)
        targets = torch.rand(batch_size, 1, device=_device)
        layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)

        # First backward pass
        output1 = _small_nnue_model(images, layer_stack_indices)
        loss1 = torch.mean((output1 - targets) ** 2)
        loss1.backward()

        # Store gradients after first pass
        first_grad = _small_nnue_model.conv.weight.grad.clone()

        # Second backward pass (without clearing gradients)
        output2 = _small_nnue_model(images, layer_stack_indices)
        loss2 = torch.mean((output2 - targets) ** 2)
        loss2.backward()

        # Gradients should be accumulated (doubled assuming similar losses)
        accumulated_grad = _small_nnue_model.conv.weight.grad

        # The accumulated gradient should be approximately 2x the first gradient
        # (allowing for small numerical differences)
        expected_grad = 2 * first_grad
        assert torch.allclose(accumulated_grad, expected_grad, rtol=1e-3, atol=1e-6)


class TestQuantization:
    """Test model quantization for export."""

    def test_quantized_data_export(self, _trained_nnue_model):
        """Test that quantized data can be exported."""
        quantized_data = _trained_nnue_model.get_quantized_data()

        # Should have all required components
        assert_quantized_weights_valid(quantized_data)

    def test_feature_transformer_quantization(self, _trained_nnue_model):
        """Test feature transformer quantization specifically."""
        quantized_data = _trained_nnue_model.get_quantized_data()

        ft_data = quantized_data["feature_transformer"]

        # Check data types
        assert ft_data["weight"].dtype == torch.int16
        assert ft_data["bias"].dtype == torch.int32

        # Check shapes
        assert ft_data["weight"].shape == _trained_nnue_model.input.weight.shape
        assert ft_data["bias"].shape == _trained_nnue_model.input.bias.shape

        # Check scale is reasonable
        assert 0 < ft_data["scale"] < 1000

    def test_layer_stack_quantization(self, _trained_nnue_model):
        """Test layer stack quantization."""
        quantized_data = _trained_nnue_model.get_quantized_data()

        # Should have layer stack data for each bucket
        for i in range(_trained_nnue_model.num_ls_buckets):
            ls_key = f"layer_stack_{i}"
            assert ls_key in quantized_data

            ls_data = quantized_data[ls_key]

            # Check required fields
            assert "l1_weight" in ls_data
            assert "l2_weight" in ls_data
            assert "output_weight" in ls_data
            assert "scales" in ls_data

            # Check data types
            assert ls_data["l1_weight"].dtype == torch.int8
            assert ls_data["l2_weight"].dtype == torch.int8
            assert ls_data["output_weight"].dtype == torch.int8

    def test_conv_layer_quantization(self, _trained_nnue_model):
        """Test conv layer quantization."""
        quantized_data = _trained_nnue_model.get_quantized_data()

        conv_data = quantized_data["conv_layer"]

        # Check data types
        assert conv_data["weight"].dtype == torch.int8
        assert conv_data["bias"].dtype == torch.int32

        # Check shapes match original
        assert conv_data["weight"].shape == _trained_nnue_model.conv.weight.shape
        assert conv_data["bias"].shape == _trained_nnue_model.conv.bias.shape

        # Check scale
        assert conv_data["scale"] > 0


class TestPyTorchLightningIntegration:
    """Test PyTorch Lightning integration."""

    def test_training_step(self, _small_nnue_model, _small_image_batch, _device):
        """Test PyTorch Lightning training step."""
        _small_nnue_model.to(_device)

        batch_idx = 0
        loss = _small_nnue_model.training_step(_small_image_batch, batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_validation_step(self, _small_nnue_model, _small_image_batch, _device):
        """Test PyTorch Lightning validation step."""
        _small_nnue_model.to(_device)

        batch_idx = 0
        loss = _small_nnue_model.validation_step(_small_image_batch, batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_test_step(self, _small_nnue_model, _small_image_batch, _device):
        """Test PyTorch Lightning test step."""
        _small_nnue_model.to(_device)

        batch_idx = 0
        loss = _small_nnue_model.test_step(_small_image_batch, batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_configure_optimizers(self, _nnue_model):
        """Test optimizer configuration."""
        config = _nnue_model.configure_optimizers()

        assert isinstance(config, (list, tuple))
        assert len(config) == 2  # [optimizers], [schedulers]

        optimizers, schedulers = config
        assert len(optimizers) == 1
        assert len(schedulers) == 1

        optimizer = optimizers[0]
        scheduler = schedulers[0]

        # Should be Adam (fallback) or Ranger21 if available
        assert hasattr(optimizer, "param_groups")
        assert hasattr(scheduler, "step")

    def test_loss_computation(self, _small_nnue_model, _device):
        """Test loss computation details."""
        _small_nnue_model.to(_device)
        _small_nnue_model.eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, 96, 96, device=_device)
        targets = torch.rand(batch_size, 1, device=_device)
        scores = torch.randn(batch_size, 1, device=_device) * 50
        layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)

        batch = (images, targets, scores, layer_stack_indices)

        with torch.no_grad():
            loss = _small_nnue_model.training_step(batch, 0)

            # Loss should be reasonable
            assert 0 <= loss.item() <= 1000  # Reasonable upper bound

        # Test with different target values
        targets_zero = torch.zeros(batch_size, 1, device=_device)
        targets_one = torch.ones(batch_size, 1, device=_device)

        batch_zero = (images, targets_zero, scores, layer_stack_indices)
        batch_one = (images, targets_one, scores, layer_stack_indices)

        with torch.no_grad():
            loss_zero = _small_nnue_model.training_step(batch_zero, 0)
            loss_one = _small_nnue_model.training_step(batch_one, 0)

            # Different targets should generally produce different losses
            # (unless model output happens to be exactly 0.5)
            assert loss_zero.item() != loss_one.item()


class TestModelPersistence:
    """Test model saving and loading."""

    def test_model_state_dict_save_load(self, _small_nnue_model, _temp_model_path):
        """Test saving and loading model state dict."""
        # Save model
        torch.save(_small_nnue_model.state_dict(), _temp_model_path)

        # Create new model with same architecture
        feature_set = _small_nnue_model.feature_set
        new_model = NNUE(
            feature_set=feature_set,
            l1_size=_small_nnue_model.l1_size,
            l2_size=_small_nnue_model.l2_size,
            l3_size=_small_nnue_model.l3_size,
            num_ls_buckets=_small_nnue_model.num_ls_buckets,
        )

        # Load state dict
        new_model.load_state_dict(torch.load(_temp_model_path))

        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            _small_nnue_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6)

        # Test that models produce same output
        batch_size = 2
        images = torch.randn(batch_size, 3, 96, 96)
        layer_stack_indices = torch.randint(0, 2, (batch_size,))

        _small_nnue_model.eval()
        new_model.eval()

        with torch.no_grad():
            output1 = _small_nnue_model(images, layer_stack_indices)
            output2 = new_model(images, layer_stack_indices)

        assert torch.allclose(output1, output2, atol=1e-6)


class TestRobustness:
    """Test model robustness to edge cases."""

    def test_extreme_image_values(self, _small_nnue_model, _device):
        """Test model with extreme input values."""
        _small_nnue_model.to(_device)
        _small_nnue_model.eval()

        batch_size = 2
        layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)

        test_cases = [
            torch.zeros(batch_size, 3, 96, 96, device=_device),  # All zeros
            torch.ones(batch_size, 3, 96, 96, device=_device),  # All ones
            torch.full(
                (batch_size, 3, 96, 96), 1000.0, device=_device
            ),  # Very large values
            torch.full(
                (batch_size, 3, 96, 96), -1000.0, device=_device
            ),  # Very negative values
        ]

        for images in test_cases:
            with torch.no_grad():
                output = _small_nnue_model(images, layer_stack_indices)

            # Model should still produce finite outputs
            assert torch.isfinite(output).all(), f"Non-finite output for extreme inputs"
            assert_tensor_shape(output, (batch_size, 1))

    def test_model_determinism(self, _small_nnue_model, _device):
        """Test that model produces deterministic outputs."""
        _small_nnue_model.to(_device)
        _small_nnue_model.eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, 96, 96, device=_device)
        layer_stack_indices = torch.randint(0, 2, (batch_size,), device=_device)

        # Run multiple times with same input
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = _small_nnue_model(images, layer_stack_indices)
            outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_invalid_bucket_indices(self, _small_nnue_model, _device):
        """Test model behavior with invalid bucket indices."""
        _small_nnue_model.to(_device)
        _small_nnue_model.eval()

        batch_size = 2
        images = torch.randn(batch_size, 3, 96, 96, device=_device)

        # Test with out-of-range bucket indices
        invalid_indices = torch.tensor([99, 100], device=_device)  # Way out of range

        # Model should handle gracefully (likely with clamping)
        with torch.no_grad():
            try:
                output = _small_nnue_model(images, invalid_indices)
                # If it doesn't crash, output should still be valid
                assert_tensor_shape(output, (batch_size, 1))
                assert torch.isfinite(output).all()
            except (IndexError, RuntimeError):
                # It's also acceptable for model to raise an error for invalid indices
                pass

    def test_sparse_feature_validation(self, _grid_feature_set):
        """Test sparse feature validation utility."""
        batch_size = 3
        max_features = 8

        # Valid sparse features
        valid_indices = torch.randint(
            0, _grid_feature_set.num_features, (batch_size, max_features)
        )
        valid_values = torch.ones(batch_size, max_features)

        # Add some padding (-1)
        valid_indices[0, -2:] = -1  # Last 2 features are padding
        valid_indices[1, -1:] = -1  # Last feature is padding

        # Should pass validation
        assert_sparse_features_valid(
            valid_indices, valid_values, _grid_feature_set.num_features
        )

    def test_invalid_sparse_features(self, _grid_feature_set):
        """Test detection of invalid sparse features."""
        batch_size = 2
        max_features = 4

        # Test case 1: Feature indices out of range
        invalid_indices = torch.tensor(
            [
                [0, 1, _grid_feature_set.num_features + 1, -1],
                [2, 3, 4, -1],
            ]  # Out of range
        )
        valid_values = torch.ones(batch_size, max_features)

        with pytest.raises(AssertionError):
            assert_sparse_features_valid(
                invalid_indices, valid_values, _grid_feature_set.num_features
            )

        # Test case 2: Invalid padding value
        invalid_indices = torch.tensor(
            [[0, 1, 2, -2], [3, 4, 5, -1]]
        )  # -2 is not valid padding
        valid_values = torch.ones(batch_size, max_features)

        with pytest.raises(AssertionError):
            assert_sparse_features_valid(
                invalid_indices, valid_values, _grid_feature_set.num_features
            )


class TestSerialization:
    """Test model serialization to .nnue format."""

    @pytest.fixture
    def temp_nnue_path(self, _tmp_path):
        """Return a temporary path for NNUE files."""
        return _tmp_path / "test_model.nnue"

    @pytest.fixture
    def temp_checkpoint_path(self, _tmp_path):
        """Return a temporary path for checkpoint files."""
        return _tmp_path / "test_checkpoint.pt"

    def test_model_quantized_data_export(self, _small_nnue_model):
        """Test quantized data export for C++ engine."""
        quantized_data = _small_nnue_model.get_quantized_data()

        # Validate structure
        assert_quantized_weights_valid(quantized_data)

        # Check metadata
        metadata = quantized_data["metadata"]
        assert (
            metadata["feature_set"]["grid_size"]
            == _small_nnue_model.feature_set.grid_size
        )
        assert (
            metadata["feature_set"]["num_features_per_square"]
            == _small_nnue_model.feature_set.num_features_per_square
        )
        assert metadata["l1_size"] == _small_nnue_model.l1_size
        assert metadata["l2_size"] == _small_nnue_model.l2_size
        assert metadata["l3_size"] == _small_nnue_model.l3_size
        assert metadata["num_ls_buckets"] == _small_nnue_model.num_ls_buckets

        # Check that quantized weights have reasonable ranges
        ft_weight = quantized_data["feature_transformer"]["weight"]
        assert ft_weight.min() >= -32768 and ft_weight.max() <= 32767  # int16 range

        conv_weight = quantized_data["conv_layer"]["weight"]
        assert conv_weight.min() >= -128 and conv_weight.max() <= 127  # int8 range

        # Check layer stacks
        for i in range(_small_nnue_model.num_ls_buckets):
            ls_data = quantized_data[f"layer_stack_{i}"]

            l1_weight = ls_data["l1_weight"]
            assert l1_weight.min() >= -128 and l1_weight.max() <= 127

            l2_weight = ls_data["l2_weight"]
            assert l2_weight.min() >= -128 and l2_weight.max() <= 127

            output_weight = ls_data["output_weight"]
            assert output_weight.min() >= -128 and output_weight.max() <= 127

        # Additional checks for scale factors
        for i in range(_small_nnue_model.num_ls_buckets):
            ls_data = quantized_data[f"layer_stack_{i}"]
            scales = ls_data["scales"]

            for scale_name, scale_value in scales.items():
                assert scale_value > 0, f"Scale {scale_name} should be positive"
                assert (
                    scale_value < 1000
                ), f"Scale {scale_name} seems too large: {scale_value}"

    def test_nnue_file_structure_validation(self, _small_nnue_model, _temp_nnue_path):
        """Test that serialized .nnue file has correct structure."""
        # Import serialize module for testing
        import serialize

        # Serialize model
        serialize.serialize_model(_small_nnue_model, _temp_nnue_path)

        # Verify file was created
        assert _temp_nnue_path.exists()
        assert _temp_nnue_path.stat().st_size > 0

        # Read and validate file structure
        with open(_temp_nnue_path, "rb") as f:
            # Check magic header
            magic = f.read(4)
            assert len(magic) == 4

            # Check version
            version = struct.unpack("I", f.read(4))[0]
            assert version > 0

            # Check feature set info
            grid_size = struct.unpack("I", f.read(4))[0]
            assert grid_size == _small_nnue_model.feature_set.grid_size

            features_per_square = struct.unpack("I", f.read(4))[0]
            assert (
                features_per_square
                == _small_nnue_model.feature_set.num_features_per_square
            )

            # Check layer sizes
            l1_size = struct.unpack("I", f.read(4))[0]
            assert l1_size == _small_nnue_model.l1_size

            l2_size = struct.unpack("I", f.read(4))[0]
            assert l2_size == _small_nnue_model.l2_size

            l3_size = struct.unpack("I", f.read(4))[0]
            assert l3_size == _small_nnue_model.l3_size

            num_buckets = struct.unpack("I", f.read(4))[0]
            assert num_buckets == _small_nnue_model.num_ls_buckets

            # More detailed validation could be added here
            # For now, just check that we can read the basic header

    def test_serialization_consistency_across_runs(self, _small_nnue_model, _tmp_path):
        """Test that serialization produces identical files across multiple runs."""
        import serialize

        path1 = _tmp_path / "model1.nnue"
        path2 = _tmp_path / "model2.nnue"

        # Serialize same model twice
        serialize.serialize_model(_small_nnue_model, path1)
        serialize.serialize_model(_small_nnue_model, path2)

        # Files should be identical
        with open(path1, "rb") as f1, open(path2, "rb") as f2:
            data1 = f1.read()
            data2 = f2.read()

        assert data1 == data2, "Serialization should be deterministic"

    def test_serialize_different_model_sizes(self, _tmp_path):
        """Test serialization with different model architectures."""
        import serialize

        test_configs = [
            # (grid_size, features_per_square, l1, l2, l3, buckets)
            (4, 4, 32, 4, 8, 2),
            (8, 8, 64, 8, 16, 4),
            (6, 12, 128, 16, 32, 3),
        ]

        for i, (grid_size, features_per_square, l1, l2, l3, buckets) in enumerate(
            test_configs
        ):
            feature_set = GridFeatureSet(grid_size, features_per_square)
            model = NNUE(
                feature_set=feature_set,
                l1_size=l1,
                l2_size=l2,
                l3_size=l3,
                num_ls_buckets=buckets,
            )

            path = _tmp_path / f"model_{i}.nnue"
            serialize.serialize_model(model, path)

            # File should exist and have reasonable size
            assert path.exists()
            assert path.stat().st_size > 1000  # Should be at least 1KB

    def test_error_handling_invalid_input(self, _tmp_path):
        """Test error handling in serialization."""
        import serialize

        # Test with invalid path
        invalid_path = Path("/invalid/path/model.nnue")

        feature_set = GridFeatureSet(4, 4)
        model = NNUE(
            feature_set=feature_set, l1_size=32, l2_size=4, l3_size=8, num_ls_buckets=2
        )

        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            serialize.serialize_model(model, invalid_path)

    def test_quantization_weight_clipping(self, _small_nnue_model):
        """Test that quantization properly clips extreme weights."""
        # Set some weights to extreme values
        with torch.no_grad():
            _small_nnue_model.input.weight.data.fill_(100.0)  # Very large
            _small_nnue_model.layer_stacks.l1.weight.data.fill_(-100.0)  # Very negative

        # Get quantized data
        quantized_data = _small_nnue_model.get_quantized_data()

        # Check that quantized weights are within valid ranges
        ft_weight = quantized_data["feature_transformer"]["weight"]
        assert ft_weight.min() >= -32768
        assert ft_weight.max() <= 32767

        # Check layer stack weights
        for i in range(_small_nnue_model.num_ls_buckets):
            ls_data = quantized_data[f"layer_stack_{i}"]

            for weight_name in ["l1_weight", "l2_weight", "output_weight"]:
                weight = ls_data[weight_name]
                assert weight.min() >= -128
                assert weight.max() <= 127

        # Check conv layer weights
        conv_weight = quantized_data["conv_layer"]["weight"]
        assert conv_weight.min() >= -128
        assert conv_weight.max() <= 127

    def test_conv_layer_serialization(self, _small_nnue_model, _temp_nnue_path):
        """Test that conv layer is properly serialized."""
        import serialize

        # Serialize model
        serialize.serialize_model(_small_nnue_model, _temp_nnue_path)

        # Verify that conv layer data is in quantized export
        quantized_data = _small_nnue_model.get_quantized_data()
        conv_data = quantized_data["conv_layer"]

        # Check that conv layer has all required fields
        assert "weight" in conv_data
        assert "bias" in conv_data
        assert "scale" in conv_data

        # Check shapes match original conv layer
        original_conv = _small_nnue_model.conv
        assert conv_data["weight"].shape == original_conv.weight.shape
        assert conv_data["bias"].shape == original_conv.bias.shape

        # Check quantization is reasonable
        assert conv_data["weight"].dtype == torch.int8
        assert conv_data["bias"].dtype == torch.int32
        assert conv_data["scale"] > 0

    def test_model_size_estimation(self, _small_nnue_model, _temp_nnue_path):
        """Test that serialized model has expected file size."""
        import serialize

        # Serialize model
        serialize.serialize_model(_small_nnue_model, _temp_nnue_path)

        file_size = _temp_nnue_path.stat().st_size

        # Estimate expected size based on model parameters
        # This is a rough estimate - actual size may vary due to headers, alignment, etc.

        # Feature transformer: weights (int16) + biases (int32)
        ft_weights = _small_nnue_model.input.weight.numel() * 2  # int16 = 2 bytes
        ft_biases = _small_nnue_model.input.bias.numel() * 4  # int32 = 4 bytes

        # Conv layer: weights (int8) + biases (int32)
        conv_weights = _small_nnue_model.conv.weight.numel() * 1  # int8 = 1 byte
        conv_biases = _small_nnue_model.conv.bias.numel() * 4  # int32 = 4 bytes

        # Layer stacks: all weights (int8) + biases (int32)
        ls_size = 0
        for i in range(_small_nnue_model.num_ls_buckets):
            # Rough estimate of layer stack size
            ls_size += (
                _small_nnue_model.l1_size * (_small_nnue_model.l2_size + 1)
            ) * 1  # L1 weights
            ls_size += (
                _small_nnue_model.l2_size * 2 * _small_nnue_model.l3_size
            ) * 1  # L2 weights
            ls_size += (_small_nnue_model.l3_size * 1) * 1  # Output weights
            ls_size += (
                _small_nnue_model.l2_size + 1 + _small_nnue_model.l3_size + 1
            ) * 4  # Biases

        estimated_size = ft_weights + ft_biases + conv_weights + conv_biases + ls_size

        # Add some overhead for headers, metadata, alignment
        estimated_size_with_overhead = estimated_size + 1000

        # File size should be reasonably close to estimate
        assert file_size >= estimated_size * 0.5  # At least 50% of estimate
        assert (
            file_size <= estimated_size_with_overhead * 2
        )  # At most 2x estimate with overhead
