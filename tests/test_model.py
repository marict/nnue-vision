"""
Tests for NNUE model forward and backward passes.

This module tests:
- Model initialization and architecture
- Forward pass functionality
- Backward pass and gradient computation
- Model training and validation steps
- Model saving and loading
- PyTorch Lightning integration
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from model import ModelParams, SimpleCNN
from tests.conftest import (assert_gradients_exist, assert_gradients_nonzero,
                            assert_tensor_shape)


class TestModelParams:
    """Test the ModelParams dataclass."""

    def test_default_params(self):
        """Test default parameter values."""
        params = ModelParams()
        assert params.input_size == (96, 96)
        assert params.num_classes == 2
        assert params.learning_rate == 1e-3

    def test_custom_params(self):
        """Test custom parameter values."""
        params = ModelParams(input_size=(224, 224), num_classes=10, learning_rate=1e-4)
        assert params.input_size == (224, 224)
        assert params.num_classes == 10
        assert params.learning_rate == 1e-4


class TestSimpleCNNArchitecture:
    """Test the SimpleCNN model architecture."""

    def test_model_initialization(self, model_params):
        """Test that model initializes correctly."""
        model = SimpleCNN(model_params)

        # Check that all layers exist
        assert hasattr(model, "conv1")
        assert hasattr(model, "bn1")
        assert hasattr(model, "conv2")
        assert hasattr(model, "bn2")
        assert hasattr(model, "conv3")
        assert hasattr(model, "bn3")
        assert hasattr(model, "global_pool")
        assert hasattr(model, "classifier")
        assert hasattr(model, "loss_fn")

        # Check layer types
        assert isinstance(model.conv1, nn.Conv2d)
        assert isinstance(model.bn1, nn.BatchNorm2d)
        assert isinstance(model.global_pool, nn.AdaptiveAvgPool2d)
        assert isinstance(model.classifier, nn.Linear)
        assert isinstance(model.loss_fn, nn.CrossEntropyLoss)

    def test_layer_dimensions(self, simple_model):
        """Test that layers have correct dimensions."""
        # Conv layers
        assert simple_model.conv1.in_channels == 3
        assert simple_model.conv1.out_channels == 32

        assert simple_model.conv2.in_channels == 32
        assert simple_model.conv2.out_channels == 64

        assert simple_model.conv3.in_channels == 64
        assert simple_model.conv3.out_channels == 128

        # Classifier
        assert simple_model.classifier.in_features == 128
        assert simple_model.classifier.out_features == 2

    def test_parameter_count(self, simple_model):
        """Test that model has reasonable number of parameters."""
        total_params = sum(p.numel() for p in simple_model.parameters())

        # Should be reasonable for a simple CNN
        assert (
            10000 < total_params < 100000
        ), f"Expected 10k-100k parameters, got {total_params}"

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in simple_model.parameters() if p.requires_grad
        )
        assert trainable_params == total_params  # All parameters should be trainable


class TestForwardPass:
    """Test the forward pass functionality."""

    def test_forward_pass_basic(self, simple_model, sample_batch, device):
        """Test basic forward pass functionality."""
        simple_model.to(device)
        simple_model.eval()

        images, _ = sample_batch

        with torch.no_grad():
            logits = simple_model(images)

        # Check output shape and type
        assert_tensor_shape(logits, (4, 2))  # batch_size=4, num_classes=2
        assert logits.dtype == torch.float32

        # Check that outputs are reasonable
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_forward_pass_different_batch_sizes(self, simple_model, device):
        """Test forward pass with different batch sizes."""
        simple_model.to(device)
        simple_model.eval()

        batch_sizes = [1, 2, 8, 16]

        for batch_size in batch_sizes:
            images = torch.randn(batch_size, 3, 96, 96, device=device)

            with torch.no_grad():
                logits = simple_model(images)

            assert_tensor_shape(logits, (batch_size, 2))
            assert not torch.isnan(logits).any()

    def test_forward_pass_different_input_sizes(self, device):
        """Test forward pass with different input sizes."""
        # Test with different input sizes
        input_sizes = [(64, 64), (128, 128), (224, 224)]

        for input_size in input_sizes:
            params = ModelParams(input_size=input_size)
            model = SimpleCNN(params)
            model.to(device)
            model.eval()

            images = torch.randn(2, 3, input_size[0], input_size[1], device=device)

            with torch.no_grad():
                logits = model(images)

            assert_tensor_shape(logits, (2, 2))

    def test_intermediate_activations(self, simple_model, sample_batch, device):
        """Test that intermediate activations have expected properties."""
        simple_model.to(device)
        simple_model.eval()

        images, _ = sample_batch

        # Hook to capture intermediate activations
        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output

            return hook

        # Register hooks
        simple_model.conv1.register_forward_hook(hook_fn("conv1"))
        simple_model.conv2.register_forward_hook(hook_fn("conv2"))
        simple_model.conv3.register_forward_hook(hook_fn("conv3"))
        simple_model.global_pool.register_forward_hook(hook_fn("global_pool"))

        with torch.no_grad():
            logits = simple_model(images)

        # Check intermediate activation shapes
        assert_tensor_shape(
            activations["conv1"], (4, 32, 48, 48)
        )  # stride=2, so 96/2=48
        assert_tensor_shape(activations["conv2"], (4, 64, 24, 24))  # 48/2=24
        assert_tensor_shape(activations["conv3"], (4, 128, 12, 12))  # 24/2=12
        assert_tensor_shape(
            activations["global_pool"], (4, 128, 1, 1)
        )  # global pooling

        # Check that activations are reasonable
        for name, activation in activations.items():
            assert not torch.isnan(activation).any(), f"NaN in {name}"
            assert not torch.isinf(activation).any(), f"Inf in {name}"


class TestBackwardPass:
    """Test the backward pass and gradient computation."""

    def test_backward_pass_basic(self, simple_model, sample_batch, device):
        """Test basic backward pass functionality."""
        simple_model.to(device)
        simple_model.train()

        images, labels = sample_batch

        # Forward pass
        logits = simple_model(images)
        loss = simple_model.loss_fn(logits, labels)

        # Backward pass
        loss.backward()

        # Check that loss is reasonable
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Check that gradients exist
        assert_gradients_exist(simple_model)
        assert_gradients_nonzero(simple_model)

    def test_gradient_shapes(self, simple_model, sample_batch, device):
        """Test that gradients have correct shapes."""
        simple_model.to(device)
        simple_model.train()

        images, labels = sample_batch

        # Forward and backward pass
        logits = simple_model(images)
        loss = simple_model.loss_fn(logits, labels)
        loss.backward()

        # Check gradient shapes match parameter shapes
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert (
                    param.grad.shape == param.shape
                ), f"Gradient shape mismatch for {name}"

    def test_gradient_accumulation(self, simple_model, device):
        """Test gradient accumulation across multiple batches."""
        simple_model.to(device)
        simple_model.train()

        # First batch
        images1 = torch.randn(2, 3, 96, 96, device=device)
        labels1 = torch.randint(0, 2, (2,), device=device)

        logits1 = simple_model(images1)
        loss1 = simple_model.loss_fn(logits1, labels1)
        loss1.backward()

        # Store first gradients
        first_grads = {}
        for name, param in simple_model.named_parameters():
            if param.grad is not None:
                first_grads[name] = param.grad.clone()

        # Second batch (without zeroing gradients)
        images2 = torch.randn(2, 3, 96, 96, device=device)
        labels2 = torch.randint(0, 2, (2,), device=device)

        logits2 = simple_model(images2)
        loss2 = simple_model.loss_fn(logits2, labels2)
        loss2.backward()

        # Check that gradients accumulated
        for name, param in simple_model.named_parameters():
            if param.grad is not None and name in first_grads:
                # Use relative tolerance for very small gradients
                if torch.max(torch.abs(first_grads[name])) > 1e-6:
                    assert not torch.allclose(
                        param.grad, first_grads[name], rtol=1e-3, atol=1e-6
                    ), f"Gradients did not accumulate for {name}"
                else:
                    # For very small gradients, just check they're not exactly equal
                    assert not torch.equal(
                        param.grad, first_grads[name]
                    ), f"Gradients did not accumulate for {name}"

    def test_gradient_zeroing(self, simple_model, sample_batch, device):
        """Test that gradients can be zeroed."""
        simple_model.to(device)
        simple_model.train()

        images, labels = sample_batch

        # First forward/backward pass
        logits = simple_model(images)
        loss = simple_model.loss_fn(logits, labels)
        loss.backward()

        # Zero gradients
        simple_model.zero_grad()

        # Check that all gradients are zero
        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    assert torch.allclose(
                        param.grad, torch.zeros_like(param.grad)
                    ), f"Gradient not zeroed for {name}"


class TestPyTorchLightningIntegration:
    """Test PyTorch Lightning specific functionality."""

    def test_training_step(self, simple_model, sample_batch, device):
        """Test the training_step method."""
        simple_model.to(device)

        images, labels = sample_batch

        # Mock batch_idx
        batch_idx = 0

        loss = simple_model.training_step((images, labels), batch_idx)

        # Check that loss is returned and reasonable
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_validation_step(self, simple_model, sample_batch, device):
        """Test the validation_step method."""
        simple_model.to(device)

        images, labels = sample_batch
        batch_idx = 0

        loss = simple_model.validation_step((images, labels), batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_test_step(self, simple_model, sample_batch, device):
        """Test the test_step method."""
        simple_model.to(device)

        images, labels = sample_batch
        batch_idx = 0

        loss = simple_model.test_step((images, labels), batch_idx)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_configure_optimizers(self, simple_model):
        """Test the configure_optimizers method."""
        config = simple_model.configure_optimizers()

        assert isinstance(config, dict)
        assert "optimizer" in config
        assert "lr_scheduler" in config

        optimizer = config["optimizer"]
        scheduler_config = config["lr_scheduler"]

        assert isinstance(optimizer, torch.optim.Adam)
        assert isinstance(
            scheduler_config["scheduler"], torch.optim.lr_scheduler.StepLR
        )


class TestModelPersistence:
    """Test model saving and loading."""

    def test_model_state_dict_save_load(self, simple_model, temp_model_path):
        """Test saving and loading model state dict."""
        # Save model
        torch.save(simple_model.state_dict(), temp_model_path)

        # Create new model and load state
        new_model = SimpleCNN(simple_model.params)
        new_model.load_state_dict(torch.load(temp_model_path, weights_only=True))

        # Compare parameters
        for (name1, param1), (name2, param2) in zip(
            simple_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)

    def test_model_consistency_after_loading(
        self, trained_model, temp_model_path, sample_batch, device
    ):
        """Test that loaded model produces same outputs as original."""
        trained_model.to(device)
        trained_model.eval()

        images, _ = sample_batch

        # Get outputs from original model
        with torch.no_grad():
            original_outputs = trained_model(images)

        # Save and load model
        torch.save(trained_model.state_dict(), temp_model_path)

        new_model = SimpleCNN(trained_model.params)
        new_model.load_state_dict(torch.load(temp_model_path, weights_only=True))
        new_model.to(device)
        new_model.eval()

        # Get outputs from loaded model
        with torch.no_grad():
            loaded_outputs = new_model(images)

        # Outputs should be identical
        assert torch.allclose(original_outputs, loaded_outputs, atol=1e-6)


class TestModelRobustness:
    """Test model robustness and edge cases."""

    def test_model_with_extreme_inputs(self, simple_model, device):
        """Test model behavior with extreme input values."""
        simple_model.to(device)
        simple_model.eval()

        # Test with very large values
        large_images = torch.ones(2, 3, 96, 96, device=device) * 1000
        with torch.no_grad():
            logits = simple_model(large_images)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

        # Test with very small values
        small_images = torch.ones(2, 3, 96, 96, device=device) * 1e-6
        with torch.no_grad():
            logits = simple_model(small_images)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

        # Test with negative values
        negative_images = torch.ones(2, 3, 96, 96, device=device) * -10
        with torch.no_grad():
            logits = simple_model(negative_images)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_model_determinism(self, simple_model, device):
        """Test that model is deterministic given same inputs."""
        simple_model.to(device)
        simple_model.eval()

        # Set manual seed for reproducibility
        torch.manual_seed(42)
        images = torch.randn(2, 3, 96, 96, device=device)

        # Run forward pass multiple times
        outputs = []
        for _ in range(3):
            with torch.no_grad():
                output = simple_model(images)
            outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)
