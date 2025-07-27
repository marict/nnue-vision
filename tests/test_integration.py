"""Fast integration tests using mock data."""

import pytest
import torch


# Use fast fixtures from conftest.py instead of slow real data
class TestBasicIntegration:
    """Test basic integration functionality."""

    def test_model_data_compatibility(self, tiny_nnue_model, fast_data_loaders, device):
        """Test that model works with data loaders."""
        model = tiny_nnue_model
        model.to(device)
        model.eval()

        train_loader, _, _ = fast_data_loaders
        batch = next(iter(train_loader))
        images, labels = batch

        # Create fake layer stack indices for the model
        layer_stack_indices = torch.randint(0, 2, (images.shape[0],), device=device)

        with torch.no_grad():
            output = model(images.to(device), layer_stack_indices)

        assert output.shape[0] == images.shape[0]
        assert not torch.isnan(output).any()

    def test_minimal_training_loop(self, tiny_nnue_model, fast_data_loaders, device):
        """Test a minimal training loop."""
        model = tiny_nnue_model
        model.to(device)
        model.train()

        train_loader, _, _ = fast_data_loaders
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Run one training step
        batch = next(iter(train_loader))
        images, labels = batch

        # Create batch in the format expected by the model
        targets = (
            labels.float().unsqueeze(1).to(device)
        )  # Convert to float and add dimension
        scores = torch.randn_like(targets) * 10  # Fake scores
        layer_stack_indices = torch.randint(0, 2, (images.shape[0],), device=device)

        model_batch = (images.to(device), targets, scores, layer_stack_indices)

        # Forward pass
        loss = model.training_step(model_batch, 0)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify loss is reasonable
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMemoryEfficiency:
    """Test memory efficiency."""

    def test_memory_efficiency(self, tiny_nnue_model, fast_data_loaders, device):
        """Test that training doesn't cause memory leaks."""
        model = tiny_nnue_model
        model.to(device)
        model.train()

        train_loader, _, _ = fast_data_loaders
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_memory = (
            torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        )

        # Run multiple training steps
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Just a few steps for memory test
                break

            images, labels = batch
            targets = labels.float().unsqueeze(1).to(device)
            scores = torch.randn_like(targets) * 5
            layer_stack_indices = torch.randint(0, 2, (images.shape[0],), device=device)

            model_batch = (images.to(device), targets, scores, layer_stack_indices)

            loss = model.training_step(model_batch, i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_memory = (
            torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        )

        # Memory usage shouldn't grow excessively (allow for some normal variation)
        if torch.cuda.is_available():
            memory_growth = final_memory - initial_memory
            # Allow up to 100MB growth for model and optimization states
            assert (
                memory_growth < 100 * 1024 * 1024
            ), f"Excessive memory growth: {memory_growth} bytes"


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_input_shape(self, tiny_nnue_model, device):
        """Test error handling for invalid input shapes."""
        model = tiny_nnue_model
        model.to(device)
        model.eval()

        # Wrong input shape (should be [B, 3, 96, 96])
        invalid_input = torch.randn(2, 4, 96, 96, device=device)
        layer_stack_indices = torch.tensor([0, 1], device=device)

        with pytest.raises((RuntimeError, ValueError)):
            model(invalid_input, layer_stack_indices)
