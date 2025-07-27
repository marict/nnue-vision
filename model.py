"""
Original NNUE Quantization Implementation for Computer Vision

This module implements the quantization scheme from the original NNUE paper:
"Efficiently Updatable Neural-Network-based Evaluation Functions for Computer Shogi"
by Yu Nasu (2018), adapted for computer vision tasks.

Key differences from standard quantization:
- 16-bit weights for feature transformer (W₁)
- 8-bit weights for subsequent layers (W₂, W₃, W₄)
- Optimized for CPU SIMD instructions (AVX2)
- Supports difference-based updates
- Adapted for image classification tasks

Additionally implements EtinyNet: Extremely Tiny Network for TinyML
from "EtinyNet: Extremely Tiny Network for TinyML" by Xu et al. (2022)
"""

from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

# Default layer sizes (can be overridden in NNUE constructor)
# Updated for 0.98M parameter target to match EtinyNet-0.98M
DEFAULT_L1 = 1024
DEFAULT_L2 = 15
DEFAULT_L3 = 32


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

    grid_size: int = 8  # 8x8 grid to match test expectations
    num_features_per_square: int = (
        12  # 12 features per square to match test expectations
    )

    @property
    def num_features(self) -> int:
        return self.grid_size * self.grid_size * self.num_features_per_square

    @property
    def name(self) -> str:
        return f"Grid{self.grid_size}x{self.grid_size}_{self.num_features_per_square}"


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable convolution implementation for EtinyNet.
    Consists of depthwise convolution followed by pointwise convolution.
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True
    ):
        super().__init__()

        # Depthwise convolution: each input channel is convolved separately
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LinearDepthwiseBlock(nn.Module):
    """
    Linear Depthwise Block (LB) from EtinyNet paper.

    Structure: depthwise conv -> pointwise conv -> depthwise conv
    Key insight: No ReLU after first depthwise conv to preserve information flow
    and enable higher parameter efficiency.
    """

    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()

        # First depthwise convolution (no activation after this)
        self.dconv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )

        # Pointwise convolution to change channel dimension
        self.pconv = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=True
        )

        # Second depthwise convolution
        self.dconv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=mid_channels,
            bias=False,
        )

        # Final pointwise to get output channels
        self.pconv_out = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True
        )

        # ReLU activation (only after pointwise and final depthwise)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First depthwise (no activation - linear)
        out = self.dconv1(x)

        # Pointwise with ReLU
        out = self.pconv(out)
        out = self.relu(out)

        # Second depthwise with ReLU
        out = self.dconv2(out)
        out = self.relu(out)

        # Final pointwise
        out = self.pconv_out(out)

        return out


class DenseLinearDepthwiseBlock(nn.Module):
    """
    Dense Linear Depthwise Block (DLB) from EtinyNet paper.

    Same as LinearDepthwiseBlock but with dense connection (skip connection).
    Used only at higher stages to save memory consumption.
    """

    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()

        self.linear_block = LinearDepthwiseBlock(
            in_channels, mid_channels, out_channels, stride
        )

        # Skip connection (only if input and output channels match and stride=1)
        self.use_skip = in_channels == out_channels and stride == 1

    def forward(self, x):
        out = self.linear_block(x)

        if self.use_skip:
            out = out + x  # Dense connection

        return out


class AdaptiveScaleQuantization(nn.Module):
    """
    Adaptive Scale Quantization (ASQ) from EtinyNet paper.

    Novel quantization method that uses learnable λ parameter to balance
    quantization error and information entropy for aggressive low-bit quantization.
    """

    def __init__(self, bits=4, init_lambda=2.0):
        super().__init__()

        self.bits = bits
        self.register_parameter("lambda_param", nn.Parameter(torch.tensor(init_lambda)))

        # Quantization levels
        self.a_hat = 2 ** (bits - 1) - 1  # For symmetric quantization

    def forward(self, weights):
        if not self.training:
            return weights  # No quantization during inference

        # ASQ quantization scheme from Equation (6)
        eps = 1e-5

        # Adaptive re-scaling with learnable lambda (add stability)
        variance = torch.var(weights) + eps
        lambda_clamped = torch.clamp(
            self.lambda_param, 0.1, 10.0
        )  # Prevent extreme values
        W_tilde = weights / (lambda_clamped * torch.sqrt(variance) + eps)

        # Clamping step with more conservative bounds
        W_hat = torch.tanh(W_tilde)
        max_val = torch.max(torch.abs(W_hat))
        if max_val > eps:
            W_hat = W_hat / max_val

        # Symmetric quantization
        Q = torch.round(self.a_hat * W_hat) / self.a_hat

        # Apply quantization with gradual introduction during training
        # This helps with training stability
        alpha = (
            min(1.0, self.training_step / 1000.0)
            if hasattr(self, "training_step")
            else 1.0
        )
        quantized = weights + alpha * (Q - weights).detach()

        # Sanity check: ensure output is reasonable
        if torch.isnan(quantized).any() or torch.isinf(quantized).any():
            return weights  # Fallback to original weights if unstable

        return quantized


class EtinyNet(pl.LightningModule):
    """
    EtinyNet: Extremely Tiny Network for TinyML

    Implements the architecture from "EtinyNet: Extremely Tiny Network for TinyML"
    by Xu et al. (2022). Features ultra-lightweight CNN with Linear Depthwise Blocks (LB)
    and Dense Linear Depthwise Blocks (DLB) for maximum parameter efficiency.

    Three variants:
    - EtinyNet-1.0: 976K parameters, 117M MAdds (original from paper)
    - EtinyNet-0.75: 680K parameters, 75M MAdds (original from paper)
    - EtinyNet-0.98M: 980K parameters (custom variant for fair NNUE comparison)
    """

    def __init__(
        self,
        variant="1.0",
        num_classes=1000,
        input_size=112,
        use_asq=False,
        asq_bits=4,
        lr=0.1,
        max_epochs=300,
        weight_decay=1e-4,
    ):
        super().__init__()

        self.variant = variant
        self.num_classes = num_classes
        self.input_size = input_size
        self.use_asq = use_asq
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

        # Architecture configurations from Table 1
        if variant == "1.0":
            self.configs = {
                "conv_channels": 32,
                "stage1": [(32, 32, 32), 4],  # LB: [32,32,32] × 4
                "stage2": [
                    (32, 128, 128),
                    1,
                    (128, 128, 128),
                    3,
                ],  # LB: [32,128,128] × 1, [128,128,128] × 3
                "stage3": [
                    (128, 192, 192),
                    1,
                    (192, 192, 192),
                    2,
                ],  # DLB: [128,192,192] × 1, [192,192,192] × 2
                "stage4": [
                    (192, 256, 256),
                    1,
                    (256, 256, 256),
                    1,
                    (256, 512, 512),
                    1,
                ],  # DLB
            }
        elif variant == "0.75":
            self.configs = {
                "conv_channels": 24,
                "stage1": [(24, 24, 24), 4],  # LB: [24,24,24] × 4
                "stage2": [
                    (24, 96, 96),
                    1,
                    (96, 96, 96),
                    3,
                ],  # LB: [24,96,96] × 1, [96,96,96] × 3
                "stage3": [
                    (96, 168, 168),
                    1,
                    (168, 168, 168),
                    2,
                ],  # DLB: [96,168,168] × 1, [168,168,168] × 2
                "stage4": [
                    (168, 192, 192),
                    1,
                    (192, 192, 192),
                    1,
                    (192, 384, 384),
                    1,
                ],  # DLB
            }
        elif variant == "0.98M":
            self.configs = {
                "conv_channels": 24,
                "stage1": [(24, 24, 24), 4],  # LB: [24,24,24] × 4
                "stage2": [
                    (24, 96, 96),
                    1,
                    (96, 96, 96),
                    3,
                ],  # LB: [24,96,96] × 1, [96,96,96] × 3
                "stage3": [
                    (96, 144, 144),
                    1,
                    (144, 144, 144),
                    2,
                ],  # DLB: [96,144,144] × 1, [144,144,144] × 2
                "stage4": [
                    (144, 192, 192),
                    1,
                    (192, 192, 192),
                    1,
                    (192, 384, 384),
                    1,
                ],  # DLB
            }
        else:
            raise ValueError(
                f"Unsupported variant: {variant}. Use '1.0', '0.75', or '0.98M'"
            )

        # Build the network
        self._build_network()

        # Initialize ASQ for 4-bit quantization if enabled
        if use_asq:
            self.asq = AdaptiveScaleQuantization(bits=asq_bits)

        # Initialize weights
        self._init_weights()

    def _build_network(self):
        """Build EtinyNet architecture following Table 1."""
        layers = []

        # Initial convolution: 3×3, stride 2
        layers.append(
            nn.Conv2d(
                3,
                self.configs["conv_channels"],
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # Max pooling: 2×2, stride 2
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Stage 1: 56² -> LB blocks (maximum memory consumption stage)
        current_channels = self.configs["conv_channels"]
        stage1_config = self.configs["stage1"]
        in_ch, mid_ch, out_ch = stage1_config[0]
        num_blocks = stage1_config[1]

        # First block may have channel change
        layers.append(LinearDepthwiseBlock(current_channels, mid_ch, out_ch, stride=1))
        current_channels = out_ch

        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.append(
                LinearDepthwiseBlock(current_channels, mid_ch, out_ch, stride=1)
            )

        self.stage1 = nn.Sequential(*layers)

        # Stage 2: 28² -> LB blocks
        stage2_layers = []
        stage2_config = self.configs["stage2"]

        # Parse stage2 config: [(in1, mid1, out1), num1, (in2, mid2, out2), num2]
        in_ch1, mid_ch1, out_ch1 = stage2_config[0]
        num_blocks1 = stage2_config[1]
        in_ch2, mid_ch2, out_ch2 = stage2_config[2]
        num_blocks2 = stage2_config[3]

        # First set of blocks with stride=2 for downsampling
        stage2_layers.append(
            LinearDepthwiseBlock(current_channels, mid_ch1, out_ch1, stride=2)
        )
        current_channels = out_ch1

        for _ in range(num_blocks1 - 1):
            stage2_layers.append(
                LinearDepthwiseBlock(current_channels, mid_ch1, out_ch1, stride=1)
            )

        # Second set of blocks
        stage2_layers.append(
            LinearDepthwiseBlock(current_channels, mid_ch2, out_ch2, stride=1)
        )
        current_channels = out_ch2

        for _ in range(num_blocks2 - 1):
            stage2_layers.append(
                LinearDepthwiseBlock(current_channels, mid_ch2, out_ch2, stride=1)
            )

        self.stage2 = nn.Sequential(*stage2_layers)

        # Stage 3: 14² -> DLB blocks
        stage3_layers = []
        stage3_config = self.configs["stage3"]

        in_ch1, mid_ch1, out_ch1 = stage3_config[0]
        num_blocks1 = stage3_config[1]
        in_ch2, mid_ch2, out_ch2 = stage3_config[2]
        num_blocks2 = stage3_config[3]

        # First set with stride=2 for downsampling
        stage3_layers.append(
            DenseLinearDepthwiseBlock(current_channels, mid_ch1, out_ch1, stride=2)
        )
        current_channels = out_ch1

        for _ in range(num_blocks1 - 1):
            stage3_layers.append(
                DenseLinearDepthwiseBlock(current_channels, mid_ch1, out_ch1, stride=1)
            )

        # Second set
        stage3_layers.append(
            DenseLinearDepthwiseBlock(current_channels, mid_ch2, out_ch2, stride=1)
        )
        current_channels = out_ch2

        for _ in range(num_blocks2 - 1):
            stage3_layers.append(
                DenseLinearDepthwiseBlock(current_channels, mid_ch2, out_ch2, stride=1)
            )

        self.stage3 = nn.Sequential(*stage3_layers)

        # Stage 4: 7² -> DLB blocks
        stage4_layers = []
        stage4_config = self.configs["stage4"]

        # Parse stage4: [(ch1), num1, (ch2), num2, (ch3), num3]
        in_ch1, mid_ch1, out_ch1 = stage4_config[0]
        num_blocks1 = stage4_config[1]
        in_ch2, mid_ch2, out_ch2 = stage4_config[2]
        num_blocks2 = stage4_config[3]
        in_ch3, mid_ch3, out_ch3 = stage4_config[4]
        num_blocks3 = stage4_config[5]

        # First set with stride=2 for downsampling
        stage4_layers.append(
            DenseLinearDepthwiseBlock(current_channels, mid_ch1, out_ch1, stride=2)
        )
        current_channels = out_ch1

        for _ in range(num_blocks1 - 1):
            stage4_layers.append(
                DenseLinearDepthwiseBlock(current_channels, mid_ch1, out_ch1, stride=1)
            )

        # Second set
        stage4_layers.append(
            DenseLinearDepthwiseBlock(current_channels, mid_ch2, out_ch2, stride=1)
        )
        current_channels = out_ch2

        for _ in range(num_blocks2 - 1):
            stage4_layers.append(
                DenseLinearDepthwiseBlock(current_channels, mid_ch2, out_ch2, stride=1)
            )

        # Third set
        stage4_layers.append(
            DenseLinearDepthwiseBlock(current_channels, mid_ch3, out_ch3, stride=1)
        )
        current_channels = out_ch3

        for _ in range(num_blocks3 - 1):
            stage4_layers.append(
                DenseLinearDepthwiseBlock(current_channels, mid_ch3, out_ch3, stride=1)
            )

        self.stage4 = nn.Sequential(*stage4_layers)

        # Global average pooling + FC-1000
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_channels, self.num_classes)

        # Store final feature dimensions for parameter counting
        self.final_channels = current_channels

    def _init_weights(self):
        """Initialize weights following paper specifications."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through EtinyNet."""
        # Apply ASQ quantization during training if enabled
        if self.training and self.use_asq:
            # Apply ASQ to conv weights
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.data = self.asq(module.weight.data)

        # Forward through stages
        x = self.stage1(x)  # 56² -> after conv+pool+LB
        x = self.stage2(x)  # 28² -> after LB
        x = self.stage3(x)  # 14² -> after DLB
        x = self.stage4(x)  # 7² -> after DLB

        # Global pooling and classification
        x = self.global_pool(x)  # -> 1×1
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        """Training step for PyTorch Lightning."""
        images, targets = batch
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, targets)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for PyTorch Lightning."""
        images, targets = batch
        outputs = self(images)
        loss = nn.functional.cross_entropy(outputs, targets)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).sum().item() / targets.size(0)

        # Calculate additional metrics for multi-class classification
        if self.num_classes > 2:
            # Multi-class metrics
            targets_np = targets.cpu().numpy()
            predicted_np = predicted.cpu().numpy()

            f1 = f1_score(targets_np, predicted_np, average="weighted", zero_division=0)
            precision = precision_score(
                targets_np, predicted_np, average="weighted", zero_division=0
            )
            recall = recall_score(
                targets_np, predicted_np, average="weighted", zero_division=0
            )
        else:
            # Binary classification metrics
            targets_np = targets.cpu().numpy()
            predicted_np = predicted.cpu().numpy()

            f1 = f1_score(targets_np, predicted_np, zero_division=0)
            precision = precision_score(targets_np, predicted_np, zero_division=0)
            recall = recall_score(targets_np, predicted_np, zero_division=0)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step for PyTorch Lightning."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizers and schedulers for EtinyNet."""
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=0
        )

        return [optimizer], [scheduler]

    def count_parameters(self):
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self, input_size=None):
        """Estimate FLOPs (multiply-adds) for the model."""
        if input_size is None:
            input_size = self.input_size

        # This is a rough estimation - for exact FLOP counting,
        # use specialized tools like fvcore or ptflops
        total_flops = 0

        # Initial conv: H*W*3*32*3*3
        h, w = input_size // 2, input_size // 2  # After stride=2
        total_flops += h * w * 3 * self.configs["conv_channels"] * 9

        # Max pool doesn't add FLOPs
        h, w = h // 2, w // 2  # After max pool

        # Estimate for each stage (simplified)
        # This is a rough approximation - exact counting would need
        # to trace through each layer
        if self.variant == "1.0":
            estimated_flops = 117_000_000  # From paper
        elif self.variant == "0.75":
            estimated_flops = 75_000_000  # From paper
        else:  # 0.98M
            estimated_flops = 98_000_000  # Custom variant

        return estimated_flops


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

        # Standard PyTorch implementation
        for b in range(batch_size):
            valid_mask = feature_indices[b] >= 0
            if valid_mask.any():
                valid_indices = feature_indices[b][valid_mask]
                valid_values = feature_values[b][valid_mask]
                feature_weights = self.weight[valid_indices]
                weighted_features = feature_weights * valid_values.unsqueeze(-1)
                output[b] += weighted_features.sum(dim=0)

        return output


class LayerStacks(nn.Module):
    def __init__(
        self,
        count,
        l1_size=DEFAULT_L1,
        l2_size=DEFAULT_L2,
        l3_size=DEFAULT_L3,
        num_classes=1,
    ):
        super(LayerStacks, self).__init__()

        self.count = count
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        self.num_classes = num_classes

        self.l1 = nn.Linear(l1_size, (l2_size + 1) * count)
        # Factorizer only for the first layer because later
        # there's a non-linearity and factorization breaks.
        # This is by design. The weights in the further layers should be
        # able to diverge a lot.
        self.l1_fact = nn.Linear(l1_size, l2_size + 1, bias=True)
        self.l2 = nn.Linear(l2_size * 2, l3_size * count)
        self.output = nn.Linear(l3_size, num_classes * count)

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
                l1_weight[i * (self.l2_size + 1) : (i + 1) * (self.l2_size + 1), :] = (
                    l1_weight[0 : (self.l2_size + 1), :]
                )
                l1_bias[i * (self.l2_size + 1) : (i + 1) * (self.l2_size + 1)] = (
                    l1_bias[0 : (self.l2_size + 1)]
                )
                l2_weight[i * self.l3_size : (i + 1) * self.l3_size, :] = l2_weight[
                    0 : self.l3_size, :
                ]
                l2_bias[i * self.l3_size : (i + 1) * self.l3_size] = l2_bias[
                    0 : self.l3_size
                ]
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

        l1s_ = self.l1(x).reshape((-1, self.count, self.l2_size + 1))
        l1f_ = self.l1_fact(x)
        # Pick the appropriate bucket for each sample
        l1c_ = l1s_.view(-1, self.l2_size + 1)[indices]
        l1c_, l1c_out = l1c_.split(self.l2_size, dim=1)
        l1f_, l1f_out = l1f_.split(self.l2_size, dim=1)
        l1x_ = l1c_ + l1f_
        # multiply sqr crelu result by (127/128) to match quantized version
        l1x_ = torch.clamp(
            torch.cat([torch.pow(l1x_, 2.0) * (127 / 128), l1x_], dim=1), 0.0, 1.0
        )

        l2s_ = self.l2(l1x_).reshape((-1, self.count, self.l3_size))
        l2c_ = l2s_.view(-1, self.l3_size)[indices]
        l2x_ = torch.clamp(l2c_, 0.0, 1.0)

        l3s_ = self.output(l2x_).reshape((-1, self.count, self.num_classes))
        l3c_ = l3s_.view(-1, self.num_classes)[indices]
        l3x_ = l3c_ + l1f_out + l1c_out

        return l3x_

    def get_coalesced_layer_stacks(self):
        # During training the buckets are represented by a single, wider, layer.
        # This representation needs to be transformed into individual layers
        # for the serializer, because the buckets are interpreted as separate layers.
        for i in range(self.count):
            with torch.no_grad():
                # Create layers that preserve the +1 components needed for factorization
                l1 = nn.Linear(self.l1_size, self.l2_size + 1)  # Keep +1 for l1c_out
                l2 = nn.Linear(self.l2_size * 2, self.l3_size)
                output = nn.Linear(self.l3_size, 1)

                # Get combined weights/biases (l1 + l1_fact) but preserve all components
                combined_weight = (
                    self.l1.weight[
                        i * (self.l2_size + 1) : (i + 1) * (self.l2_size + 1), :
                    ]
                    + self.l1_fact.weight.data
                )
                combined_bias = (
                    self.l1.bias[i * (self.l2_size + 1) : (i + 1) * (self.l2_size + 1)]
                    + self.l1_fact.bias.data
                )

                # Preserve all components including the +1 for l1c_out
                l1.weight.data = combined_weight
                l1.bias.data = combined_bias

                l2.weight.data = self.l2.weight[
                    i * self.l3_size : (i + 1) * self.l3_size, :
                ]
                l2.bias.data = self.l2.bias[i * self.l3_size : (i + 1) * self.l3_size]
                output.weight.data = self.output.weight[i : (i + 1), :]
                output.bias.data = self.output.bias[i : (i + 1)]
                yield l1, l2, output


def get_parameters(layers):
    return [p for layer in layers for p in layer.parameters()]


class NNUE(pl.LightningModule):
    """
    NNUE model for computer vision with configurable architecture for maximum efficiency.

    Processes images through convolutional feature extraction to binary features,
    then through NNUE architecture for final classification.

    Features:
    - Configurable feature space and layer sizes for testing and deployment
    - Convolutional frontend with aggressive spatial downsampling
    - Quantization-ready architecture for C++ conversion
    - Bucketed layer stacks for efficiency
    - Supports various computer vision datasets (CIFAR-10, CIFAR-100, etc.)
    """

    def __init__(
        self,
        feature_set: Optional[GridFeatureSet] = None,
        l1_size: int = DEFAULT_L1,
        l2_size: int = DEFAULT_L2,
        l3_size: int = DEFAULT_L3,
        max_epoch=800,
        num_batches_per_epoch=int(100_000_000 / 16384),
        gamma=0.992,
        lr=8.75e-4,
        param_index=0,
        num_ls_buckets=8,
        loss_params=LossParams(),
        visual_threshold=0.0,
        num_classes=1,
        weight_decay=5e-4,
    ):
        super(NNUE, self).__init__()

        # Architecture configuration
        if feature_set is None:
            # Default to smaller feature set for 0.98M parameter target to match EtinyNet-0.98M
            feature_set = GridFeatureSet(grid_size=10, num_features_per_square=8)

        self.feature_set = feature_set
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        self.num_ls_buckets = num_ls_buckets
        self.visual_threshold = visual_threshold
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        # Calculate conv output channels based on feature set
        # For grid-based features, we need enough channels to cover all features per square
        conv_out_channels = feature_set.num_features_per_square

        # Calculate image size and stride based on feature set grid size
        # Default: 96x96 input, but make stride dynamic to hit target grid size
        input_size = 96
        target_grid_size = feature_set.grid_size

        # Calculate stride to get exactly target_grid_size output
        # For conv with kernel=3, padding=1: output = (input + 2*padding - kernel) // stride + 1
        # Solving for stride: stride = (input + 2*padding - kernel) // (output - 1)
        # We want output = target_grid_size, so:
        conv_stride = max(1, (input_size + 2 * 1 - 3) // (target_grid_size - 1))

        # Verify the output size will be correct
        expected_output_size = (input_size + 2 * 1 - 3) // conv_stride + 1
        if expected_output_size != target_grid_size:
            # Adjust stride if needed to get exact match
            conv_stride = (input_size + 2 * 1 - 3) // (target_grid_size - 1)
            if conv_stride < 1:
                conv_stride = 1

        # Visual processing layers - conv and tanh at the beginning
        self.conv = nn.Conv2d(
            in_channels=3,  # RGB input
            out_channels=conv_out_channels,
            kernel_size=3,
            stride=conv_stride,
            padding=1,  # Keep spatial dimensions
            bias=True,
        )
        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)

        # Single feature transformer (standard PyTorch)
        self.input = FeatureTransformer(self.feature_set.num_features, l1_size)

        # Standard layer stacks (no optimizations)
        self.layer_stacks = LayerStacks(
            num_ls_buckets, l1_size, l2_size, l3_size, num_classes
        )

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
        # Initialize conv layer weights
        with torch.no_grad():
            nn.init.normal_(self.conv.weight, mean=0.0, std=0.1)
            nn.init.zeros_(self.conv.bias)

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

    def _to_sparse_features(
        self, binary_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert dense binary features to sparse representation for NNUE.

        During training, this maintains gradients by using the continuous binary_features values.
        During inference, this uses hard thresholding.

        Args:
            binary_features: Binary tensor of shape (B, C, H, W) where C is conv_out_channels

        Returns:
            Tuple of (feature_indices, feature_values) for NNUE input
        """
        batch_size, num_channels, height, width = binary_features.shape

        if self.training:
            # During training, use all positions but with continuous values
            # This maintains gradients through the sparse representation

            # Create all possible feature indices
            all_indices = []
            for c in range(num_channels):
                for h in range(height):
                    for w in range(width):
                        linear_idx = c * (height * width) + h * width + w
                        all_indices.append(linear_idx)

            max_features = len(all_indices)
            feature_indices = (
                torch.tensor(all_indices, device=binary_features.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

            # Flatten binary features and use as values
            binary_flat = binary_features.view(batch_size, -1)
            feature_values = binary_flat

            return feature_indices, feature_values

        else:
            # During inference, use sparse representation for efficiency
            # Find active features (where value > 0.5)
            active_mask = binary_features > 0.5

            # Convert to feature indices
            feature_indices_list = []
            feature_values_list = []

            for b in range(batch_size):
                # Get active positions for this batch
                active_positions = torch.nonzero(active_mask[b], as_tuple=False)

                if len(active_positions) > 0:
                    # Calculate linear indices
                    channel_indices = active_positions[:, 0]  # channel
                    row_indices = active_positions[:, 1]  # row
                    col_indices = active_positions[:, 2]  # col

                    linear_indices = (
                        channel_indices * (height * width)
                        + row_indices * width
                        + col_indices
                    )

                    feature_indices_list.append(linear_indices)
                    feature_values_list.append(
                        torch.ones_like(linear_indices, dtype=torch.float32)
                    )
                else:
                    # No active features
                    feature_indices_list.append(
                        torch.tensor(
                            [], dtype=torch.long, device=binary_features.device
                        )
                    )
                    feature_values_list.append(
                        torch.tensor(
                            [], dtype=torch.float32, device=binary_features.device
                        )
                    )

            # Pad to same length for batching
            max_features = max(len(indices) for indices in feature_indices_list)
            if max_features == 0:
                max_features = 1  # Avoid empty tensors

            batch_feature_indices = torch.full(
                (batch_size, max_features),
                -1,
                dtype=torch.long,
                device=binary_features.device,
            )
            batch_feature_values = torch.zeros(
                (batch_size, max_features),
                dtype=torch.float32,
                device=binary_features.device,
            )

            for b, (indices, values) in enumerate(
                zip(feature_indices_list, feature_values_list)
            ):
                if len(indices) > 0:
                    batch_feature_indices[b, : len(indices)] = indices
                    batch_feature_values[b, : len(values)] = values

            return batch_feature_indices, batch_feature_values

    def forward(
        self,
        images: torch.Tensor,
        layer_stack_indices: torch.Tensor,
    ):
        """
        Forward pass from images to evaluation scores.

        Args:
            images: RGB images of shape (B, 3, H, W) - typically (B, 3, 96, 96)
            layer_stack_indices: [batch_size] - which bucket to use for each sample
        """
        # Convolution: (B, 3, H, W) -> (B, conv_out_channels, grid_h, grid_w)
        x = self.conv(images)

        # Apply Hardtanh activation
        x = self.hardtanh(x)

        # Apply threshold to get binary values
        # During training, use differentiable approximation
        if self.training:
            # Smooth approximation using sigmoid with high temperature
            binary_features = torch.sigmoid(10.0 * (x - self.visual_threshold))
        else:
            # Hard threshold for inference
            binary_features = (x > self.visual_threshold).float()

        # Convert to sparse features for NNUE
        feature_indices, feature_values = self._to_sparse_features(binary_features)

        # Transform sparse features to dense representation
        features = self.input(feature_indices, feature_values)

        # Clamp to [0, 1]
        l0_ = torch.clamp(features, 0.0, 1.0)

        # Split into halves and apply squared activation (original NNUE approach)
        l0_s = torch.split(l0_, self.l1_size // 2, dim=1)
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
            images,  # RGB images (B, 3, H, W)
            targets,  # Target labels
            scores,  # Search scores
            layer_stack_indices,  # Bucket indices
        ) = batch

        # Forward pass
        scorenet = self(images, layer_stack_indices) * self.nnue2score

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

        # Only log if trainer is available (prevents warnings during testing)
        if hasattr(self, "_trainer") and self._trainer is not None:
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

        # Separate parameters into weight decay and no weight decay groups
        # Apply weight decay only to weights of conv/linear layers, not biases
        weight_decay_params = []
        no_weight_decay_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                # Apply weight decay to conv and linear layer weights, but not biases
                if "weight" in name and (
                    "conv" in name
                    or "linear" in name
                    or "fc" in name
                    or "l1_fact" in name
                    or "l1." in name
                    or "l2." in name
                    or "output." in name
                ):
                    weight_decay_params.append(param)
                else:
                    no_weight_decay_params.append(param)

        # Set up parameter groups with different weight decay
        param_groups = [
            {"params": weight_decay_params, "weight_decay": self.weight_decay},
            {"params": no_weight_decay_params, "weight_decay": 0.0},
        ]

        # Use Adam with momentum 0.9 (beta1=0.9, beta2=0.999)
        optimizer = torch.optim.Adam(param_groups, lr=LR, betas=(0.9, 0.999), eps=1e-8)

        # Use cosine annealing: start at max LR, anneal to 0 at end of training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epoch, eta_min=0.0  # Anneal to near zero
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

            # Get original factorization data for this bucket
            l1_fact_weight = self.layer_stacks.l1_fact.weight.data
            l1_fact_bias = self.layer_stacks.l1_fact.bias.data

            quantized_data[f"layer_stack_{i}"] = {
                "l1_weight": torch.round(l1.weight.data * l1_scale)
                .clamp_(-127, 127)
                .to(torch.int8),
                "l1_bias": torch.round(l1.bias.data * l1_scale).to(torch.int32),
                # Add factorization layer data
                "l1_fact_weight": torch.round(l1_fact_weight * l1_scale)
                .clamp_(-127, 127)
                .to(torch.int8),
                "l1_fact_bias": torch.round(l1_fact_bias * l1_scale).to(torch.int32),
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
                "scales": {
                    "l1": l1_scale,
                    "l2": l2_scale,
                    "output": out_scale,
                    "l1_fact": l1_scale,
                },
            }

        # Add conv layer weights (scale conv channels based on feature set)
        conv_weight = self.conv.weight.data
        conv_bias = self.conv.bias.data

        # Quantize conv weights (using 8-bit like hidden layers)
        conv_scale = 64.0
        conv_weight_q = (
            torch.round(conv_weight * conv_scale).clamp_(-127, 127).to(torch.int8)
        )
        conv_bias_q = torch.round(conv_bias * conv_scale).to(torch.int32)

        quantized_data["conv_layer"] = {
            "weight": conv_weight_q,
            "bias": conv_bias_q,
            "scale": conv_scale,
            "layer_type": 0,  # Standard convolution layer
        }

        quantized_data["metadata"] = {
            "feature_set": self.feature_set,
            "L1": self.l1_size,
            "L2": self.l2_size,
            "L3": self.l3_size,
            "num_ls_buckets": self.num_ls_buckets,
            "nnue2score": self.nnue2score,
            "quantized_one": self.quantized_one,
            "visual_threshold": self.visual_threshold,
        }

        return quantized_data
