"""
Original NNUE Quantization Implementation

This module implements the quantization scheme from the original NNUE paper:
"Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi"
by Yu Nasu (2018)

Key differences from standard quantization:
- 16-bit weights for feature transformer (W₁)
- 8-bit weights for subsequent layers (W₂, W₃, W₄)
- Optimized for CPU SIMD instructions (AVX2)
- Supports difference-based updates
"""

from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from torch import nn

# 3 layer fully connected network dimensions
L1 = 3072
L2 = 15
L3 = 32


# parameters needed for the definition of the loss
@dataclass
class LossParams:
    in_offset: float = 270
    out_offset: float = 270
    in_scaling: float = 340
    out_scaling: float = 380
    start_lambda: float = 1.0
    end_lambda: float = 1.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0


@dataclass
class GridFeatureSet:
    """Feature set for NxN grid-based features."""

    grid_size: int = 8  # NxN grid
    num_features_per_square: int = 12  # Number of possible features per square

    @property
    def num_features(self) -> int:
        return self.grid_size * self.grid_size * self.num_features_per_square

    @property
    def name(self) -> str:
        return f"Grid{self.grid_size}x{self.grid_size}_{self.num_features_per_square}"


class FeatureTransformer(nn.Module):
    """
    Single-perspective feature transformer for grid-based features.
    Adapted from DoubleFeatureTransformerSlice to work with NxN grids.
    """

    def __init__(self, num_features: int, output_size: int):
        super().__init__()
        self.num_features = num_features
        self.output_size = output_size

        # Linear transformation from sparse features to dense representation
        self.weight = nn.Parameter(torch.randn(num_features, output_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(
        self, feature_indices: torch.Tensor, feature_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with sparse feature representation.

        Args:
            feature_indices: [batch_size, max_features] - indices of active features
            feature_values: [batch_size, max_features] - values of active features (typically 1.0)
        """
        batch_size = feature_indices.shape[0]

        # Initialize output tensor
        output = self.bias.unsqueeze(0).expand(batch_size, -1).clone()

        # Add contributions from active features
        for b in range(batch_size):
            # Get valid indices (non-negative)
            valid_mask = feature_indices[b] >= 0
            if valid_mask.any():
                valid_indices = feature_indices[b][valid_mask]
                valid_values = feature_values[b][valid_mask]

                # Accumulate weighted features
                feature_weights = self.weight[valid_indices]  # [num_valid, output_size]
                weighted_features = feature_weights * valid_values.unsqueeze(
                    -1
                )  # [num_valid, output_size]
                output[b] += weighted_features.sum(dim=0)

        return output


def get_parameters(layers):
    return [p for layer in layers for p in layer.parameters()]


class LayerStacks(nn.Module):
    def __init__(self, count):
        super(LayerStacks, self).__init__()

        self.count = count
        self.l1 = nn.Linear(L1, (L2 + 1) * count)
        # Factorizer only for the first layer because later
        # there's a non-linearity and factorization breaks.
        # This is by design. The weights in the further layers should be
        # able to diverge a lot.
        self.l1_fact = nn.Linear(L1, L2 + 1, bias=True)
        self.l2 = nn.Linear(L2 * 2, L3 * count)
        self.output = nn.Linear(L3, 1 * count)

        # Cached helper tensor for choosing outputs by bucket indices.
        # Initialized lazily in forward.
        self.idx_offset = None

        self._init_layers()

    def _init_layers(self):
        l1_weight = self.l1.weight
        l1_bias = self.l1.bias
        l1_fact_weight = self.l1_fact.weight
        l1_fact_bias = self.l1_fact.bias
        l2_weight = self.l2.weight
        l2_bias = self.l2.bias
        output_weight = self.output.weight
        output_bias = self.output.bias
        with torch.no_grad():
            l1_fact_weight.fill_(0.0)
            l1_fact_bias.fill_(0.0)
            output_bias.fill_(0.0)

            for i in range(1, self.count):
                # Force all layer stacks to be initialized in the same way.
                l1_weight[i * (L2 + 1) : (i + 1) * (L2 + 1), :] = l1_weight[
                    0 : (L2 + 1), :
                ]
                l1_bias[i * (L2 + 1) : (i + 1) * (L2 + 1)] = l1_bias[0 : (L2 + 1)]
                l2_weight[i * L3 : (i + 1) * L3, :] = l2_weight[0:L3, :]
                l2_bias[i * L3 : (i + 1) * L3] = l2_bias[0:L3]
                output_weight[i : i + 1, :] = output_weight[0:1, :]

        self.l1.weight = nn.Parameter(l1_weight)
        self.l1.bias = nn.Parameter(l1_bias)
        self.l1_fact.weight = nn.Parameter(l1_fact_weight)
        self.l1_fact.bias = nn.Parameter(l1_fact_bias)
        self.l2.weight = nn.Parameter(l2_weight)
        self.l2.bias = nn.Parameter(l2_bias)
        self.output.weight = nn.Parameter(output_weight)
        self.output.bias = nn.Parameter(output_bias)

    def forward(self, x, ls_indices):
        if self.idx_offset is None or self.idx_offset.shape[0] != x.shape[0]:
            batch_size = x.shape[0]
            self.idx_offset = torch.arange(0, batch_size, device=x.device) * self.count

        indices = ls_indices.flatten() + self.idx_offset

        l1s_ = self.l1(x).reshape((-1, self.count, L2 + 1))
        l1f_ = self.l1_fact(x)
        # Pick the appropriate bucket for each sample
        l1c_ = l1s_.view(-1, L2 + 1)[indices]
        l1c_, l1c_out = l1c_.split(L2, dim=1)
        l1f_, l1f_out = l1f_.split(L2, dim=1)
        l1x_ = l1c_ + l1f_
        # multiply sqr crelu result by (127/128) to match quantized version
        l1x_ = torch.clamp(
            torch.cat([torch.pow(l1x_, 2.0) * (127 / 128), l1x_], dim=1), 0.0, 1.0
        )

        l2s_ = self.l2(l1x_).reshape((-1, self.count, L3))
        l2c_ = l2s_.view(-1, L3)[indices]
        l2x_ = torch.clamp(l2c_, 0.0, 1.0)

        l3s_ = self.output(l2x_).reshape((-1, self.count, 1))
        l3c_ = l3s_.view(-1, 1)[indices]
        l3x_ = l3c_ + l1f_out + l1c_out

        return l3x_

    def get_coalesced_layer_stacks(self):
        # During training the buckets are represented by a single, wider, layer.
        # This representation needs to be transformed into individual layers
        # for the serializer, because the buckets are interpreted as separate layers.
        for i in range(self.count):
            with torch.no_grad():
                l1 = nn.Linear(L1, L2 + 1)
                l2 = nn.Linear(L2 * 2, L3)
                output = nn.Linear(L3, 1)
                l1.weight.data = (
                    self.l1.weight[i * (L2 + 1) : (i + 1) * (L2 + 1), :]
                    + self.l1_fact.weight.data
                )
                l1.bias.data = (
                    self.l1.bias[i * (L2 + 1) : (i + 1) * (L2 + 1)]
                    + self.l1_fact.bias.data
                )
                l2.weight.data = self.l2.weight[i * L3 : (i + 1) * L3, :]
                l2.bias.data = self.l2.bias[i * L3 : (i + 1) * L3]
                output.weight.data = self.output.weight[i : (i + 1), :]
                output.bias.data = self.output.bias[i : (i + 1)]
                yield l1, l2, output


class NNUE(pl.LightningModule):
    """
    Grid-based NNUE model adapted from chess NNUE.

    Features:
    - Single perspective (no white/black switching)
    - NxN grid feature space
    - Quantization-ready architecture for C++ conversion
    - Bucketed layer stacks for efficiency
    """

    def __init__(
        self,
        feature_set: Optional[GridFeatureSet] = None,
        max_epoch=800,
        num_batches_per_epoch=int(100_000_000 / 16384),
        gamma=0.992,
        lr=8.75e-4,
        param_index=0,
        num_ls_buckets=8,
        loss_params=LossParams(),
    ):
        super(NNUE, self).__init__()

        if feature_set is None:
            feature_set = GridFeatureSet()

        self.feature_set = feature_set
        self.num_ls_buckets = num_ls_buckets

        # Single feature transformer (no perspective switching)
        self.input = FeatureTransformer(feature_set.num_features, L1)
        self.layer_stacks = LayerStacks(self.num_ls_buckets)

        self.loss_params = loss_params
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.gamma = gamma
        self.lr = lr
        self.param_index = param_index

        # Quantization parameters (matching original NNUE)
        self.nnue2score = 600.0
        self.weight_scale_hidden = 64.0
        self.weight_scale_out = 16.0
        self.quantized_one = 127.0

        # Weight clipping for quantization readiness
        max_hidden_weight = self.quantized_one / self.weight_scale_hidden
        max_out_weight = (self.quantized_one * self.quantized_one) / (
            self.nnue2score * self.weight_scale_out
        )
        self.weight_clipping = [
            {
                "params": [self.layer_stacks.l1.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
                "virtual_params": self.layer_stacks.l1_fact.weight,
            },
            {
                "params": [self.layer_stacks.l2.weight],
                "min_weight": -max_hidden_weight,
                "max_weight": max_hidden_weight,
            },
            {
                "params": [self.layer_stacks.output.weight],
                "min_weight": -max_out_weight,
                "max_weight": max_out_weight,
            },
        ]

        self._init_layers()

    def _init_layers(self):
        # Initialize feature transformer weights
        with torch.no_grad():
            # Small random initialization for feature transformer
            self.input.weight.data.normal_(0, 0.1)
            self.input.bias.data.zero_()

    def _clip_weights(self):
        """
        Clips the weights of the model based on the min/max values allowed
        by the quantization scheme.
        """
        for group in self.weight_clipping:
            for p in group["params"]:
                if "min_weight" in group or "max_weight" in group:
                    p_data_fp32 = p.data
                    min_weight = group["min_weight"]
                    max_weight = group["max_weight"]
                    if "virtual_params" in group:
                        virtual_params = group["virtual_params"]
                        xs = p_data_fp32.shape[0] // virtual_params.shape[0]
                        ys = p_data_fp32.shape[1] // virtual_params.shape[1]
                        expanded_virtual_layer = virtual_params.repeat(xs, ys)
                        if min_weight is not None:
                            min_weight_t = (
                                p_data_fp32.new_full(p_data_fp32.shape, min_weight)
                                - expanded_virtual_layer
                            )
                            p_data_fp32 = torch.max(p_data_fp32, min_weight_t)
                        if max_weight is not None:
                            max_weight_t = (
                                p_data_fp32.new_full(p_data_fp32.shape, max_weight)
                                - expanded_virtual_layer
                            )
                            p_data_fp32 = torch.min(p_data_fp32, max_weight_t)
                    else:
                        if min_weight is not None and max_weight is not None:
                            p_data_fp32.clamp_(min_weight, max_weight)
                        else:
                            raise Exception("Not supported.")
                    p.data.copy_(p_data_fp32)

    def forward(
        self,
        feature_indices: torch.Tensor,
        feature_values: torch.Tensor,
        layer_stack_indices: torch.Tensor,
    ):
        """
        Forward pass for grid-based NNUE.

        Args:
            feature_indices: [batch_size, max_features] - indices of active features
            feature_values: [batch_size, max_features] - values of active features
            layer_stack_indices: [batch_size] - which bucket to use for each sample
        """
        # Transform sparse features to dense representation
        features = self.input(feature_indices, feature_values)

        # Clamp to [0, 1]
        l0_ = torch.clamp(features, 0.0, 1.0)

        # Split into halves and apply squared activation (original NNUE approach)
        l0_s = torch.split(l0_, L1 // 2, dim=1)
        l0_s1 = l0_s[0] * l0_s[1]  # Element-wise multiplication (squared crelu)

        # Concatenate original and squared parts to maintain L1 dimensions
        # This gives us the same input size that LayerStacks expects
        l0_ = torch.cat([l0_s1, l0_s[0]], dim=1) * (127 / 128)

        # Pass through layer stacks
        x = self.layer_stacks(l0_, layer_stack_indices)

        return x

    def step_(self, batch, batch_idx, loss_type):
        _ = batch_idx  # unused, but required by pytorch-lightning

        # We clip weights at the start of each step. This means that after
        # the last step the weights might be outside of the desired range.
        # They should be also clipped accordingly in the serializer.
        self._clip_weights()

        (
            feature_indices,
            feature_values,
            targets,
            scores,
            layer_stack_indices,
        ) = batch

        # Forward pass
        scorenet = (
            self(feature_indices, feature_values, layer_stack_indices) * self.nnue2score
        )

        p = self.loss_params
        # convert the network and search scores to an estimate match result
        # based on the win_rate_model, with scalings and offsets optimized
        q = (scorenet - p.in_offset) / p.in_scaling
        qm = (-scorenet - p.in_offset) / p.in_scaling
        qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

        s = (scores - p.out_offset) / p.out_scaling
        sm = (-scores - p.out_offset) / p.out_scaling
        pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

        # blend that eval based score with the actual targets
        t = targets
        actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * (
            self.current_epoch / self.max_epoch
        )
        pt = pf * actual_lambda + t * (1.0 - actual_lambda)

        # use a MSE-like loss function
        loss = torch.pow(torch.abs(pt - qf), p.pow_exp)
        if p.qp_asymmetry != 0.0:
            loss = loss * ((qf > pt) * p.qp_asymmetry + 1)
        loss = loss.mean()

        self.log(loss_type, loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    def validation_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "val_loss")

    def test_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "test_loss")

    def configure_optimizers(self):
        LR = self.lr

        # Try to use ranger21 if available, otherwise fall back to Adam
        try:
            import ranger21

            train_params = [
                {"params": get_parameters([self.input]), "lr": LR, "gc_dim": 0},
                {"params": [self.layer_stacks.l1_fact.weight], "lr": LR},
                {"params": [self.layer_stacks.l1_fact.bias], "lr": LR},
                {"params": [self.layer_stacks.l1.weight], "lr": LR},
                {"params": [self.layer_stacks.l1.bias], "lr": LR},
                {"params": [self.layer_stacks.l2.weight], "lr": LR},
                {"params": [self.layer_stacks.l2.bias], "lr": LR},
                {"params": [self.layer_stacks.output.weight], "lr": LR},
                {"params": [self.layer_stacks.output.bias], "lr": LR},
            ]

            optimizer = ranger21.Ranger21(
                train_params,
                lr=1.0,
                betas=(0.9, 0.999),
                eps=1.0e-7,
                using_gc=False,
                using_normgc=False,
                weight_decay=0.0,
                num_batches_per_epoch=self.num_batches_per_epoch,
                num_epochs=self.max_epoch,
                warmdown_active=False,
                use_warmup=False,
                use_adaptive_gradient_clipping=False,
                softplus=False,
                pnm_momentum_factor=0.0,
            )
        except ImportError:
            # Fallback to Adam if ranger21 is not available
            optimizer = torch.optim.Adam(self.parameters(), lr=LR)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.gamma
        )
        return [optimizer], [scheduler]

    def get_quantized_model_data(self):
        """Export quantized weights and metadata for C++ deployment."""
        quantized_data = {}

        # Export feature transformer weights
        ft_weight = self.input.weight.data
        ft_bias = self.input.bias.data

        # Quantize feature transformer (16-bit like original NNUE)
        ft_scale = 64.0
        ft_weight_q = (
            torch.round(ft_weight * ft_scale).clamp_(-32767, 32767).to(torch.int16)
        )
        ft_bias_q = torch.round(ft_bias * ft_scale).to(torch.int32)

        quantized_data["feature_transformer"] = {
            "weight": ft_weight_q,
            "bias": ft_bias_q,
            "scale": ft_scale,
        }

        # Export layer stack weights (8-bit for hidden layers)
        for i, (l1, l2, output) in enumerate(
            self.layer_stacks.get_coalesced_layer_stacks()
        ):
            l1_scale = self.weight_scale_hidden
            l2_scale = self.weight_scale_hidden
            out_scale = self.weight_scale_out

            quantized_data[f"layer_stack_{i}"] = {
                "l1_weight": torch.round(l1.weight.data * l1_scale)
                .clamp_(-127, 127)
                .to(torch.int8),
                "l1_bias": torch.round(l1.bias.data * l1_scale).to(torch.int32),
                "l2_weight": torch.round(l2.weight.data * l2_scale)
                .clamp_(-127, 127)
                .to(torch.int8),
                "l2_bias": torch.round(l2.bias.data * l2_scale).to(torch.int32),
                "output_weight": torch.round(output.weight.data * out_scale)
                .clamp_(-127, 127)
                .to(torch.int8),
                "output_bias": torch.round(output.bias.data * out_scale).to(
                    torch.int32
                ),
                "scales": {"l1": l1_scale, "l2": l2_scale, "output": out_scale},
            }

        quantized_data["metadata"] = {
            "feature_set": self.feature_set,
            "L1": L1,
            "L2": L2,
            "L3": L3,
            "num_ls_buckets": self.num_ls_buckets,
            "nnue2score": self.nnue2score,
            "quantized_one": self.quantized_one,
        }

        return quantized_data
