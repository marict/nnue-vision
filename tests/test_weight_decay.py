"""
Test weight decay functionality in NNUE and EtinyNet models.
"""

from types import SimpleNamespace

import pytest

from model import NNUE, EtinyNet, GridFeatureSet


class TestWeightDecayConfiguration:
    """Test weight decay parameter handling and defaults."""

    def test_etinynet_weight_decay_default(self):
        """Test that EtinyNet has the correct default weight_decay."""
        model = EtinyNet(variant="0.75", num_classes=10, input_size=32)

        # Check default weight_decay
        assert hasattr(
            model, "weight_decay"
        ), "EtinyNet should have weight_decay attribute"
        assert (
            model.weight_decay == 1e-4
        ), f"Expected default weight_decay 1e-4, got {model.weight_decay}"

    def test_etinynet_weight_decay_custom(self):
        """Test that EtinyNet accepts custom weight_decay."""
        custom_weight_decay = 5e-5
        model = EtinyNet(
            variant="0.75",
            num_classes=10,
            input_size=32,
            weight_decay=custom_weight_decay,
        )

        assert (
            model.weight_decay == custom_weight_decay
        ), f"Expected weight_decay {custom_weight_decay}, got {model.weight_decay}"

    def test_nnue_weight_decay_default(self):
        """Test that NNUE has the correct default weight_decay."""
        feature_set = GridFeatureSet(grid_size=8, num_features_per_square=12)
        model = NNUE(
            feature_set=feature_set,
            l1_size=128,
            l2_size=16,
            l3_size=32,
            num_ls_buckets=2,
        )

        # Check default weight_decay
        assert hasattr(model, "weight_decay"), "NNUE should have weight_decay attribute"
        assert (
            model.weight_decay == 5e-4
        ), f"Expected default weight_decay 5e-4, got {model.weight_decay}"

    def test_nnue_weight_decay_custom(self):
        """Test that NNUE accepts custom weight_decay."""
        custom_weight_decay = 1e-5
        feature_set = GridFeatureSet(grid_size=8, num_features_per_square=12)
        model = NNUE(
            feature_set=feature_set,
            l1_size=128,
            l2_size=16,
            l3_size=32,
            num_ls_buckets=2,
            weight_decay=custom_weight_decay,
        )

        assert (
            model.weight_decay == custom_weight_decay
        ), f"Expected weight_decay {custom_weight_decay}, got {model.weight_decay}"
