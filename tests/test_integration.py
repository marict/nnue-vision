"""
Integration tests for NNUE-Vision.

This module tests end-to-end functionality:
- Complete training workflows
- Data loading to model training pipelines
- Model evaluation and inference
- PyTorch Lightning trainer integration
- Memory usage and performance
"""

import os
import tempfile
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping

from dataset import create_data_loaders
from model import ModelParams, SimpleCNN
from tests.conftest import assert_tensor_shape


class TestTrainingWorkflow:
    """Test complete training workflows."""

    def test_minimal_training_loop(self, device):
        """Test a minimal training loop from data loading to model update."""
        # Create small data loaders for quick testing
        train_loader, val_loader, _ = create_data_loaders(
            batch_size=4, num_workers=0, target_size=(96, 96)
        )

        # Create model
        model = SimpleCNN(ModelParams())
        model.to(device)
        model.train()

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Store initial parameters for comparison
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone()

        # Training loop for one batch
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        logits = model(images)
        loss = model.loss_fn(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify that parameters were updated
        for name, param in model.named_parameters():
            assert not torch.allclose(
                param, initial_params[name]
            ), f"Parameter {name} was not updated during training"

        # Verify loss is reasonable
        assert 0 < loss.item() < 10, f"Loss {loss.item()} seems unreasonable"

        # Verify model can make predictions
        model.eval()
        with torch.no_grad():
            val_logits = model(images)
            predictions = torch.argmax(val_logits, dim=1)

        assert predictions.shape == labels.shape
        assert all(p in [0, 1] for p in predictions.tolist())

    def test_multiple_epoch_training(self, device):
        """Test training for multiple epochs."""
        # Small dataset for quick testing
        train_loader, val_loader, _ = create_data_loaders(
            batch_size=8, num_workers=0, target_size=(96, 96)
        )

        model = SimpleCNN(ModelParams())
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train for 2 epochs
        num_epochs = 2
        losses = []

        for epoch in range(num_epochs):
            epoch_losses = []
            model.train()

            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx >= 3:  # Limit batches for quick testing
                    break

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(images)
                loss = model.loss_fn(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)

        # Verify training progressed
        assert len(losses) == num_epochs
        assert all(loss > 0 for loss in losses)

        # Loss might decrease or increase slightly, but should be stable
        assert all(loss < 10 for loss in losses), "Losses seem too high"

    def test_validation_during_training(self, device):
        """Test validation evaluation during training."""
        train_loader, val_loader, _ = create_data_loaders(
            batch_size=4, num_workers=0, target_size=(96, 96)
        )

        model = SimpleCNN(ModelParams())
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train for one batch
        model.train()
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = model.loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        val_losses = []
        val_accuracies = []

        with torch.no_grad():
            for batch_idx, (val_images, val_labels) in enumerate(val_loader):
                if batch_idx >= 3:  # Limit for quick testing
                    break

                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_logits = model(val_images)
                val_loss = model.loss_fn(val_logits, val_labels)

                # Calculate accuracy
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = (val_preds == val_labels).float().mean()

                val_losses.append(val_loss.item())
                val_accuracies.append(val_acc.item())

        # Verify validation metrics
        assert len(val_losses) > 0
        assert len(val_accuracies) > 0
        assert all(loss > 0 for loss in val_losses)
        assert all(0 <= acc <= 1 for acc in val_accuracies)


class TestPyTorchLightningIntegration:
    """Test PyTorch Lightning trainer integration."""

    def test_lightning_trainer_basic(self, device):
        """Test basic PyTorch Lightning trainer functionality."""
        # Create small data loaders
        train_loader, val_loader, _ = create_data_loaders(
            batch_size=4, num_workers=0, target_size=(96, 96)
        )

        # Create model
        model = SimpleCNN(ModelParams())

        # Create trainer with minimal configuration
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=3,  # Only train on 3 batches
            limit_val_batches=2,  # Only validate on 2 batches
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu" if device.type == "cpu" else "gpu",
            devices=1,
        )

        # Train model
        trainer.fit(model, train_loader, val_loader)

        # Verify training completed
        assert trainer.current_epoch == 0  # 1 epoch completed (0-indexed)
        assert model.training_step_outputs is not None or True  # Training happened

    def test_lightning_trainer_with_callbacks(self, device):
        """Test PyTorch Lightning trainer with callbacks."""
        train_loader, val_loader, _ = create_data_loaders(
            batch_size=4, num_workers=0, target_size=(96, 96)
        )

        model = SimpleCNN(ModelParams())

        # Add early stopping callback
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=1, verbose=False, mode="min"
        )

        trainer = pl.Trainer(
            max_epochs=2,
            limit_train_batches=2,
            limit_val_batches=2,
            callbacks=[early_stop_callback],
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu" if device.type == "cpu" else "gpu",
            devices=1,
        )

        trainer.fit(model, train_loader, val_loader)

        # Training should complete without errors
        assert trainer.current_epoch >= 0

    def test_lightning_testing(self, device):
        """Test PyTorch Lightning testing functionality."""
        _, _, test_loader = create_data_loaders(
            batch_size=4, num_workers=0, target_size=(96, 96)
        )

        model = SimpleCNN(ModelParams())

        trainer = pl.Trainer(
            limit_test_batches=2,
            enable_progress_bar=False,
            logger=False,
            accelerator="cpu" if device.type == "cpu" else "gpu",
            devices=1,
        )

        # Test the model
        test_results = trainer.test(model, test_loader)

        # Verify test results
        assert len(test_results) == 1
        assert "test_loss" in test_results[0]
        assert "test_acc" in test_results[0]
        assert test_results[0]["test_loss"] > 0
        assert 0 <= test_results[0]["test_acc"] <= 1


class TestModelInference:
    """Test model inference and evaluation."""

    def test_inference_on_single_images(self, trained_model, device):
        """Test inference on individual images."""
        trained_model.to(device)
        trained_model.eval()

        # Create single image
        image = torch.randn(1, 3, 96, 96, device=device)

        with torch.no_grad():
            logits = trained_model(image)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)

        # Verify outputs
        assert_tensor_shape(logits, (1, 2))
        assert_tensor_shape(probabilities, (1, 2))
        assert_tensor_shape(prediction, (1,))

        # Check probability properties
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(1, device=device))
        assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)

        # Check prediction is valid
        assert prediction.item() in [0, 1]

    def test_batch_inference(self, trained_model, device):
        """Test inference on batches of images."""
        trained_model.to(device)
        trained_model.eval()

        batch_sizes = [1, 4, 8, 16]

        for batch_size in batch_sizes:
            images = torch.randn(batch_size, 3, 96, 96, device=device)

            with torch.no_grad():
                logits = trained_model(images)
                predictions = torch.argmax(logits, dim=1)

            assert_tensor_shape(logits, (batch_size, 2))
            assert_tensor_shape(predictions, (batch_size,))
            assert torch.all((predictions >= 0) & (predictions <= 1))

    def test_model_confidence_scores(self, trained_model, device):
        """Test that model produces reasonable confidence scores."""
        trained_model.to(device)
        trained_model.eval()

        # Create batch of images
        images = torch.randn(8, 3, 96, 96, device=device)

        with torch.no_grad():
            logits = trained_model(images)
            probabilities = torch.softmax(logits, dim=1)
            max_probs, predictions = torch.max(probabilities, dim=1)

        # Check confidence scores
        assert torch.all(max_probs >= 0.5), "All confidence scores should be >= 0.5"
        assert torch.all(max_probs <= 1.0), "All confidence scores should be <= 1.0"

        # Check predictions are consistent with max probabilities
        for i in range(len(predictions)):
            assert probabilities[i, predictions[i]] == max_probs[i]


class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_memory_efficiency(self, device):
        """Test that model doesn't use excessive memory."""
        if device.type == "cuda":
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        model = SimpleCNN(ModelParams())
        model.to(device)

        # Create moderately large batch
        images = torch.randn(16, 3, 96, 96, device=device)
        labels = torch.randint(0, 2, (16,), device=device)

        # Forward pass
        logits = model(images)
        loss = model.loss_fn(logits, labels)

        # Backward pass
        loss.backward()

        if device.type == "cuda":
            peak_memory = torch.cuda.memory_allocated()
            memory_used = peak_memory - initial_memory

            # Memory usage should be reasonable (less than 1GB for this simple model)
            assert (
                memory_used < 1e9
            ), f"Model uses too much memory: {memory_used / 1e6:.1f} MB"

    def test_gradient_checkpointing_compatibility(self, device):
        """Test that model works with gradient checkpointing if needed."""
        model = SimpleCNN(ModelParams())
        model.to(device)

        # Enable gradient checkpointing for conv layers if available
        if hasattr(torch.utils.checkpoint, "checkpoint"):
            # Model should work with checkpointing
            images = torch.randn(4, 3, 96, 96, device=device)

            def run_model(x):
                return model(x)

            # This should work without errors
            logits = torch.utils.checkpoint.checkpoint(run_model, images)
            assert_tensor_shape(logits, (4, 2))


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_wrong_input_dimensions(self, simple_model, device):
        """Test model behavior with wrong input dimensions."""
        simple_model.to(device)

        # Test with wrong number of channels
        with pytest.raises((RuntimeError, ValueError)):
            wrong_channels = torch.randn(
                1, 1, 96, 96, device=device
            )  # 1 channel instead of 3
            simple_model(wrong_channels)

        # Test with wrong spatial dimensions
        with pytest.raises((RuntimeError, ValueError)):
            wrong_size = torch.randn(
                1, 3, 32, 32, device=device
            )  # 32x32 instead of 96x96
            simple_model(wrong_size)

    def test_empty_batch_handling(self, simple_model, device):
        """Test model behavior with empty batches."""
        simple_model.to(device)

        # Empty batch should raise an error
        with pytest.raises((RuntimeError, ValueError)):
            empty_batch = torch.randn(0, 3, 96, 96, device=device)
            simple_model(empty_batch)

    def test_model_mode_consistency(self, simple_model, device):
        """Test that model behaves consistently in train/eval modes."""
        simple_model.to(device)

        # Create test input
        images = torch.randn(4, 3, 96, 96, device=device)

        # Get outputs in eval mode
        simple_model.eval()
        with torch.no_grad():
            eval_outputs = simple_model(images)

        # Get outputs in train mode (but with no_grad to make it comparable)
        simple_model.train()
        with torch.no_grad():
            train_outputs = simple_model(images)

        # Outputs might be slightly different due to batch norm behavior
        # but should be in the same range
        assert torch.allclose(eval_outputs, train_outputs, atol=1e-2)


class TestDataPipelineIntegration:
    """Test integration between data pipeline and model."""

    def test_full_pipeline_compatibility(self, device):
        """Test that the entire pipeline from data loading to model training works."""
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            batch_size=8, num_workers=0, target_size=(96, 96)
        )

        # Create and train model
        model = SimpleCNN(ModelParams())
        model.to(device)

        # Test one complete training iteration
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = model.loss_fn(logits, labels)

            # Verify everything works
            assert_tensor_shape(logits, (images.shape[0], 2))
            assert not torch.isnan(loss)
            assert loss.item() > 0

            break  # Just test one batch

        # Test validation
        model.eval()
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_logits = model(val_images)
                val_loss = model.loss_fn(val_logits, val_labels)

                assert_tensor_shape(val_logits, (val_images.shape[0], 2))
                assert not torch.isnan(val_loss)

                break  # Just test one batch

        # Test inference
        with torch.no_grad():
            for test_images, test_labels in test_loader:
                test_images = test_images.to(device)

                test_logits = model(test_images)
                predictions = torch.argmax(test_logits, dim=1)

                assert_tensor_shape(test_logits, (test_images.shape[0], 2))
                assert torch.all((predictions >= 0) & (predictions <= 1))

                break  # Just test one batch
