"""
Tests for NNUE model forward and backward passes.

This module tests:
- GridFeatureSet functionality
- FeatureTransformer architecture and sparse feature processing
- LayerStacks and bucketed layer functionality
- NNUE model forward/backward passes
- Weight clipping for quantization readiness
- Quantized model export for C++ deployment
- PyTorch Lightning integration
- Model persistence and robustness
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from model import (NNUE, FeatureTransformer, GridFeatureSet, LayerStacks,
                   LossParams)
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
    """Test the FeatureTransformer for sparse feature processing."""

    def test_feature_transformer_initialization(self, grid_feature_set):
        """Test FeatureTransformer initialization."""
        ft = FeatureTransformer(grid_feature_set.num_features, 256)

        assert ft.num_features == grid_feature_set.num_features
        assert ft.output_size == 256
        assert ft.weight.shape == (grid_feature_set.num_features, 256)
        assert ft.bias.shape == (256,)

    def test_sparse_feature_forward(self, grid_feature_set, device):
        """Test forward pass with sparse features."""
        ft = FeatureTransformer(grid_feature_set.num_features, 128)
        ft.to(device)

        batch_size = 4
        max_features = 16

        # Create valid sparse features
        feature_indices = torch.randint(
            0, grid_feature_set.num_features, (batch_size, max_features), device=device
        )
        feature_values = torch.ones(batch_size, max_features, device=device)

        # Some samples have fewer features (use -1 for padding)
        mask = torch.rand(batch_size, max_features) < 0.8
        feature_indices = feature_indices * mask.to(device) + (-1) * (~mask).to(device)

        output = ft(feature_indices, feature_values)

        assert_tensor_shape(output, (batch_size, 128))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_empty_features(self, grid_feature_set, device):
        """Test forward pass with no active features."""
        ft = FeatureTransformer(grid_feature_set.num_features, 64)
        ft.to(device)

        batch_size = 2
        max_features = 8

        # All features are inactive (-1)
        feature_indices = torch.full((batch_size, max_features), -1, device=device)
        feature_values = torch.ones(batch_size, max_features, device=device)

        output = ft(feature_indices, feature_values)

        # Should return bias only
        assert_tensor_shape(output, (batch_size, 64))
        for i in range(batch_size):
            assert torch.allclose(output[i], ft.bias, atol=1e-6)

    def test_single_feature_per_sample(self, grid_feature_set, device):
        """Test with exactly one active feature per sample."""
        ft = FeatureTransformer(grid_feature_set.num_features, 32)
        ft.to(device)

        batch_size = 3

        # One feature per sample, rest are padding
        feature_indices = torch.tensor(
            [[0, -1, -1], [10, -1, -1], [50, -1, -1]], device=device
        )
        feature_values = torch.ones(batch_size, 3, device=device)

        output = ft(feature_indices, feature_values)

        assert_tensor_shape(output, (batch_size, 32))

        # Each output should be bias + corresponding weight
        for i in range(batch_size):
            active_idx = feature_indices[i, 0].item()
            expected = ft.bias + ft.weight[active_idx]
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

    def test_layer_stacks_forward(self, device):
        """Test LayerStacks forward pass."""
        num_buckets = 2
        layer_stacks = LayerStacks(num_buckets)
        layer_stacks.to(device)

        batch_size = 4
        input_tensor = torch.randn(batch_size, 3072, device=device)  # L1 size
        bucket_indices = torch.randint(0, num_buckets, (batch_size,), device=device)

        output = layer_stacks(input_tensor, bucket_indices)

        assert_tensor_shape(output, (batch_size, 1))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_bucket_selection(self, device):
        """Test that different buckets can produce different outputs."""
        num_buckets = 3
        layer_stacks = LayerStacks(num_buckets)
        layer_stacks.to(device)
        layer_stacks.eval()  # Deterministic behavior

        # Modify weights to make buckets different
        with torch.no_grad():
            for i in range(num_buckets):
                # Make each bucket's weights slightly different
                start_idx = i * (15 + 1)  # L2 + 1
                end_idx = (i + 1) * (15 + 1)
                layer_stacks.l1.weight[start_idx:end_idx] += (i + 1) * 0.1

        input_tensor = torch.randn(1, 3072, device=device)

        outputs = []
        for bucket_idx in range(num_buckets):
            bucket_indices = torch.tensor([bucket_idx], device=device)
            output = layer_stacks(input_tensor, bucket_indices)
            outputs.append(output)

        # After modifying weights, outputs should be different for different buckets
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j], atol=1e-6)

    def test_coalesced_layer_stacks(self):
        """Test that coalesced layer stacks can be extracted."""
        num_buckets = 2
        layer_stacks = LayerStacks(num_buckets)

        coalesced = list(layer_stacks.get_coalesced_layer_stacks())

        assert len(coalesced) == num_buckets

        for l1, l2, output in coalesced:
            assert isinstance(l1, nn.Linear)
            assert isinstance(l2, nn.Linear)
            assert isinstance(output, nn.Linear)

            assert l1.in_features == 3072  # L1
            assert l1.out_features == 15 + 1  # L2 + 1
            assert l2.in_features == 15 * 2  # L2 * 2
            assert l2.out_features == 32  # L3
            assert output.in_features == 32  # L3
            assert output.out_features == 1


class TestNNUEArchitecture:
    """Test the main NNUE model architecture."""

    def test_nnue_initialization(self, grid_feature_set):
        """Test NNUE model initialization."""
        model = NNUE(grid_feature_set, num_ls_buckets=4)

        assert model.feature_set == grid_feature_set
        assert model.num_ls_buckets == 4
        assert isinstance(model.input, FeatureTransformer)
        assert isinstance(model.layer_stacks, LayerStacks)
        assert model.input.num_features == grid_feature_set.num_features
        assert model.layer_stacks.count == 4

    def test_nnue_forward_pass(self, nnue_model, sample_sparse_batch, device):
        """Test NNUE forward pass."""
        nnue_model.to(device)
        nnue_model.eval()

        feature_indices, feature_values, _, _, layer_stack_indices = sample_sparse_batch

        with torch.no_grad():
            output = nnue_model(feature_indices, feature_values, layer_stack_indices)

        batch_size = feature_indices.shape[0]
        assert_tensor_shape(output, (batch_size, 1))
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_nnue_different_batch_sizes(self, small_nnue_model, device):
        """Test NNUE with different batch sizes."""
        small_nnue_model.to(device)
        small_nnue_model.eval()

        batch_sizes = [1, 2, 4, 8]
        max_features = 16

        for batch_size in batch_sizes:
            feature_indices = torch.randint(
                0,
                small_nnue_model.feature_set.num_features,
                (batch_size, max_features),
                device=device,
            )
            feature_values = torch.ones(batch_size, max_features, device=device)
            layer_stack_indices = torch.randint(0, 2, (batch_size,), device=device)

            with torch.no_grad():
                output = small_nnue_model(
                    feature_indices, feature_values, layer_stack_indices
                )

            assert_tensor_shape(output, (batch_size, 1))
            assert not torch.isnan(output).any()

    def test_weight_clipping(self, nnue_model):
        """Test weight clipping for quantization."""
        # Get initial weight ranges
        initial_weights = {}
        for name, param in nnue_model.named_parameters():
            initial_weights[name] = param.clone()

        # Make some weights exceed clipping bounds
        with torch.no_grad():
            for param in nnue_model.layer_stacks.l1.weight:
                param.fill_(10.0)  # Large value

        # Apply weight clipping
        nnue_model._clip_weights()

        # Check weights are clipped
        max_hidden_weight = nnue_model.quantized_one / nnue_model.weight_scale_hidden
        l1_weights = nnue_model.layer_stacks.l1.weight
        assert torch.all(l1_weights >= -max_hidden_weight)
        assert torch.all(l1_weights <= max_hidden_weight)


class TestNNUEForwardBackward:
    """Test NNUE forward and backward passes."""

    def test_forward_backward_basic(self, small_nnue_model, small_sparse_batch, device):
        """Test basic forward and backward passes."""
        small_nnue_model.to(device)
        small_nnue_model.train()

        feature_indices, feature_values, targets, scores, layer_stack_indices = (
            small_sparse_batch
        )

        # Forward pass
        output = small_nnue_model(feature_indices, feature_values, layer_stack_indices)

        # Simple loss for testing
        loss = torch.mean((output - targets) ** 2)

        # Backward pass
        loss.backward()

        # Check gradients exist and are reasonable
        assert_gradients_exist(small_nnue_model)
        assert_gradients_nonzero(small_nnue_model)

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_gradient_shapes(self, small_nnue_model, small_sparse_batch, device):
        """Test that gradients have correct shapes."""
        small_nnue_model.to(device)
        small_nnue_model.train()

        feature_indices, feature_values, targets, scores, layer_stack_indices = (
            small_sparse_batch
        )

        output = small_nnue_model(feature_indices, feature_values, layer_stack_indices)
        loss = torch.mean((output - targets) ** 2)
        loss.backward()

        for name, param in small_nnue_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert (
                    param.grad.shape == param.shape
                ), f"Gradient shape mismatch for {name}"

    def test_gradient_accumulation(self, small_nnue_model, device):
        """Test gradient accumulation."""
        small_nnue_model.to(device)
        small_nnue_model.train()

        # First batch
        batch_size = 2
        max_features = 8

        feature_indices1 = torch.randint(
            0, 24, (batch_size, max_features), device=device
        )
        feature_values1 = torch.ones(batch_size, max_features, device=device)
        targets1 = torch.rand(batch_size, 1, device=device)
        layer_stack_indices1 = torch.randint(0, 2, (batch_size,), device=device)

        output1 = small_nnue_model(
            feature_indices1, feature_values1, layer_stack_indices1
        )
        loss1 = torch.mean((output1 - targets1) ** 2)
        loss1.backward()

        # Store first gradients
        first_grads = {}
        for name, param in small_nnue_model.named_parameters():
            if param.grad is not None:
                first_grads[name] = param.grad.clone()

        # Second batch (without zeroing gradients)
        feature_indices2 = torch.randint(
            0, 24, (batch_size, max_features), device=device
        )
        feature_values2 = torch.ones(batch_size, max_features, device=device)
        targets2 = torch.rand(batch_size, 1, device=device)
        layer_stack_indices2 = torch.randint(0, 2, (batch_size,), device=device)

        output2 = small_nnue_model(
            feature_indices2, feature_values2, layer_stack_indices2
        )
        loss2 = torch.mean((output2 - targets2) ** 2)
        loss2.backward()

        # Check gradients accumulated
        for name, param in small_nnue_model.named_parameters():
            if param.grad is not None and name in first_grads:
                assert not torch.allclose(param.grad, first_grads[name], atol=1e-6)


class TestQuantizedModelExport:
    """Test quantized model export for C++ deployment."""

    def test_quantized_data_export(self, trained_nnue_model):
        """Test exporting quantized model data."""
        quantized_data = trained_nnue_model.get_quantized_model_data()

        # Use helper function to validate structure
        assert_quantized_weights_valid(quantized_data)

        # Check metadata
        metadata = quantized_data["metadata"]
        assert metadata["feature_set"] == trained_nnue_model.feature_set
        assert metadata["L1"] == 3072
        assert metadata["num_ls_buckets"] == trained_nnue_model.num_ls_buckets

    def test_feature_transformer_quantization(self, trained_nnue_model):
        """Test feature transformer quantization."""
        quantized_data = trained_nnue_model.get_quantized_model_data()
        ft_data = quantized_data["feature_transformer"]

        # Check quantization parameters
        assert ft_data["weight"].dtype == torch.int16
        assert ft_data["bias"].dtype == torch.int32
        assert ft_data["scale"] == 64.0

        # Check weight ranges for 16-bit
        weights = ft_data["weight"]
        assert torch.all(weights >= -32767)
        assert torch.all(weights <= 32767)

    def test_layer_stack_quantization(self, trained_nnue_model):
        """Test layer stack quantization."""
        quantized_data = trained_nnue_model.get_quantized_model_data()

        layer_stack_keys = [
            k for k in quantized_data.keys() if k.startswith("layer_stack_")
        ]
        assert len(layer_stack_keys) == trained_nnue_model.num_ls_buckets

        for key in layer_stack_keys:
            ls_data = quantized_data[key]

            # Check 8-bit quantization
            for weight_key in ["l1_weight", "l2_weight", "output_weight"]:
                weights = ls_data[weight_key]
                assert weights.dtype == torch.int8
                assert torch.all(weights >= -127)
                assert torch.all(weights <= 127)


class TestPyTorchLightningIntegration:
    """Test PyTorch Lightning integration."""

    def test_training_step(self, small_nnue_model, small_sparse_batch, device):
        """Test PyTorch Lightning training step."""
        small_nnue_model.to(device)

        batch_idx = 0
        loss = small_nnue_model.training_step(small_sparse_batch, batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_validation_step(self, small_nnue_model, small_sparse_batch, device):
        """Test PyTorch Lightning validation step."""
        small_nnue_model.to(device)

        batch_idx = 0
        loss = small_nnue_model.validation_step(small_sparse_batch, batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_test_step(self, small_nnue_model, small_sparse_batch, device):
        """Test PyTorch Lightning test step."""
        small_nnue_model.to(device)

        batch_idx = 0
        loss = small_nnue_model.test_step(small_sparse_batch, batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_configure_optimizers(self, nnue_model):
        """Test optimizer configuration."""
        config = nnue_model.configure_optimizers()

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

    def test_loss_computation(self, small_nnue_model, device):
        """Test NNUE loss computation."""
        small_nnue_model.to(device)

        batch_size = 2
        max_features = 8

        # Create batch with known values for testing
        feature_indices = torch.randint(
            0, 24, (batch_size, max_features), device=device
        )
        feature_values = torch.ones(batch_size, max_features, device=device)
        targets = torch.tensor([[0.3], [0.8]], device=device)  # Known targets
        scores = torch.tensor([[50.0], [-30.0]], device=device)  # Known scores
        layer_stack_indices = torch.zeros(batch_size, device=device, dtype=torch.long)

        batch = (feature_indices, feature_values, targets, scores, layer_stack_indices)

        loss = small_nnue_model.step_(batch, 0, "test_loss")

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestModelPersistence:
    """Test model saving and loading."""

    def test_model_state_dict_save_load(self, small_nnue_model, temp_model_path):
        """Test saving and loading model state dict."""
        # Save model
        torch.save(small_nnue_model.state_dict(), temp_model_path)

        # Create new model and load state
        new_model = NNUE(small_nnue_model.feature_set, num_ls_buckets=2)
        new_model.load_state_dict(torch.load(temp_model_path, weights_only=True))

        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            small_nnue_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_model_consistency_after_loading(
        self, trained_nnue_model, temp_model_path, device
    ):
        """Test that loaded model produces same outputs."""
        trained_nnue_model.to(device)
        trained_nnue_model.eval()

        # Create test input
        batch_size = 2
        max_features = 8
        feature_indices = torch.randint(
            0, 24, (batch_size, max_features), device=device
        )
        feature_values = torch.ones(batch_size, max_features, device=device)
        layer_stack_indices = torch.zeros(batch_size, device=device, dtype=torch.long)

        # Get original output
        with torch.no_grad():
            original_output = trained_nnue_model(
                feature_indices, feature_values, layer_stack_indices
            )

        # Save and load model
        torch.save(trained_nnue_model.state_dict(), temp_model_path)

        new_model = NNUE(trained_nnue_model.feature_set, num_ls_buckets=2)
        new_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
        new_model.to(device)
        new_model.eval()

        # Get loaded output
        with torch.no_grad():
            loaded_output = new_model(
                feature_indices, feature_values, layer_stack_indices
            )

        assert torch.allclose(original_output, loaded_output, atol=1e-6)


class TestModelRobustness:
    """Test model robustness and edge cases."""

    def test_extreme_feature_values(self, small_nnue_model, device):
        """Test model with extreme feature values."""
        small_nnue_model.to(device)
        small_nnue_model.eval()

        batch_size = 2
        max_features = 8

        feature_indices = torch.randint(
            0, 24, (batch_size, max_features), device=device
        )
        layer_stack_indices = torch.zeros(batch_size, device=device, dtype=torch.long)

        # Test with very large values
        large_values = torch.ones(batch_size, max_features, device=device) * 1000
        with torch.no_grad():
            output = small_nnue_model(
                feature_indices, large_values, layer_stack_indices
            )
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # Test with very small values
        small_values = torch.ones(batch_size, max_features, device=device) * 1e-6
        with torch.no_grad():
            output = small_nnue_model(
                feature_indices, small_values, layer_stack_indices
            )
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_determinism(self, small_nnue_model, device):
        """Test model determinism."""
        small_nnue_model.to(device)
        small_nnue_model.eval()

        torch.manual_seed(42)
        batch_size = 2
        max_features = 8

        feature_indices = torch.randint(
            0, 24, (batch_size, max_features), device=device
        )
        feature_values = torch.ones(batch_size, max_features, device=device)
        layer_stack_indices = torch.zeros(batch_size, device=device, dtype=torch.long)

        # Run multiple times
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = small_nnue_model(
                    feature_indices, feature_values, layer_stack_indices
                )
            outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)

    def test_invalid_bucket_indices(self, small_nnue_model, device):
        """Test model behavior with invalid bucket indices."""
        small_nnue_model.to(device)

        batch_size = 2
        max_features = 8

        feature_indices = torch.randint(
            0, 24, (batch_size, max_features), device=device
        )
        feature_values = torch.ones(batch_size, max_features, device=device)

        # Invalid bucket indices (should cause error)
        invalid_indices = torch.tensor(
            [5, 10], device=device
        )  # Only 2 buckets available (0, 1)

        with pytest.raises(IndexError):
            small_nnue_model(feature_indices, feature_values, invalid_indices)


class TestSparseFeatureValidation:
    """Test sparse feature input validation."""

    def test_sparse_feature_validation(self, grid_feature_set):
        """Test that sparse features are validated correctly."""
        batch_size = 3
        max_features = 8

        # Valid features
        valid_indices = torch.randint(
            0, grid_feature_set.num_features, (batch_size, max_features)
        )
        valid_values = torch.ones(batch_size, max_features)

        # Should not raise error
        assert_sparse_features_valid(
            valid_indices, valid_values, grid_feature_set.num_features
        )

        # Test with padding
        padded_indices = valid_indices.clone()
        padded_indices[:, -2:] = -1  # Last 2 features are padding

        assert_sparse_features_valid(
            padded_indices, valid_values, grid_feature_set.num_features
        )

    def test_invalid_sparse_features(self, grid_feature_set):
        """Test detection of invalid sparse features."""
        batch_size = 2
        max_features = 4

        # Invalid: feature index exceeds max
        invalid_indices = torch.tensor([[0, 1, grid_feature_set.num_features, -1]])
        valid_values = torch.ones(1, max_features)

        with pytest.raises(AssertionError):
            assert_sparse_features_valid(
                invalid_indices, valid_values, grid_feature_set.num_features
            )

        # Invalid: negative values for valid indices
        valid_indices = torch.randint(
            0, grid_feature_set.num_features, (batch_size, max_features)
        )
        invalid_values = torch.ones(batch_size, max_features)
        invalid_values[0, 0] = -1.0  # Negative value

        with pytest.raises(AssertionError):
            assert_sparse_features_valid(
                valid_indices, invalid_values, grid_feature_set.num_features
            )
