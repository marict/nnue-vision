"""PyTorch Models

PyTorch implementations of NNUE and EtinyNet models for computer vision tasks.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Loss parameters for NNUE
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


# Default layer sizes for NNUE (matching original implementation)
DEFAULT_L1 = 1024
DEFAULT_L2 = 15
DEFAULT_L3 = 32


@dataclass
class GridFeatureSet:
    """Grid-based feature set for NNUE."""

    grid_size: int = 10
    num_features_per_square: int = 8

    @property
    def num_features(self) -> int:
        return self.grid_size * self.grid_size * self.num_features_per_square


class LinearDepthwiseBlock(nn.Module):
    """Linear Depthwise Block (LB) from EtinyNet paper."""

    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()

        # 1x1 pointwise expansion
        self.pw_expand = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 3x3 depthwise convolution
        self.dw_conv = nn.Conv2d(
            mid_channels,
            mid_channels,
            3,
            stride=stride,
            padding=1,
            groups=mid_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 1x1 pointwise projection
        self.pw_project = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Residual connection
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x

        # Pointwise expansion
        out = F.relu6(self.bn1(self.pw_expand(x)))

        # Depthwise convolution
        out = F.relu6(self.bn2(self.dw_conv(out)))

        # Pointwise projection
        out = self.bn3(self.pw_project(out))

        # Add residual connection
        if self.use_residual:
            out = out + identity

        return out


class DenseLinearDepthwiseBlock(nn.Module):
    """Dense Linear Depthwise Block (DLB) from EtinyNet paper."""

    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()

        # Same structure as LB but with dense connections
        self.lb = LinearDepthwiseBlock(in_channels, mid_channels, out_channels, stride)

        # Dense connection combines input and output
        self.use_dense = stride == 1 and in_channels == out_channels

        # If using dense connection, add a projection layer to handle concatenated channels
        if self.use_dense:
            # After concatenation: in_channels + out_channels -> out_channels
            self.dense_proj = nn.Conv2d(
                in_channels + out_channels, out_channels, 1, bias=False
            )
            self.dense_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.lb(x)

        # Dense connection: concatenate input and output if same spatial size
        if self.use_dense:
            concatenated = torch.cat([x, out], dim=1)
            # Project back to expected output channels
            out = self.dense_bn(self.dense_proj(concatenated))

        return out


class EtinyNet(nn.Module):
    """
    EtinyNet: Extremely Tiny Network for TinyML

    Efficient convolutional neural network for resource-constrained environments.
    """

    def __init__(
        self,
        variant="1.0",
        num_classes=1000,
        input_size=112,
        weight_decay=1e-4,
        use_asq=False,
        asq_bits=4,
    ):
        super().__init__()

        self.variant = variant
        self.num_classes = num_classes
        self.input_size = input_size
        self.weight_decay = weight_decay
        self.use_asq = use_asq
        self.asq_bits = asq_bits

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
                ],  # DLB: [192,256,256] × 1, [256,256,256] × 1
                "final_channels": 1280,
            }
        elif variant == "0.75":
            self.configs = {
                "conv_channels": 24,
                "stage1": [(24, 24, 24), 3],  # LB: [24,24,24] × 3
                "stage2": [
                    (24, 96, 96),
                    1,
                    (96, 96, 96),
                    2,
                ],  # LB: [24,96,96] × 1, [96,96,96] × 2
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
                ],  # DLB: [144,192,192] × 1, [192,192,192] × 1
                "final_channels": 960,
            }
        elif variant == "0.98M":
            # Custom variant for fair NNUE comparison (980K parameters)
            self.configs = {
                "conv_channels": 28,
                "stage1": [(28, 28, 28), 3],
                "stage2": [
                    (28, 112, 112),
                    1,
                    (112, 112, 112),
                    2,
                ],
                "stage3": [
                    (112, 168, 168),
                    1,
                    (168, 168, 168),
                    2,
                ],
                "stage4": [
                    (168, 224, 224),
                    1,
                    (224, 224, 224),
                    1,
                ],
                "final_channels": 1120,
            }
        elif variant == "micro":
            # Ultra-minimal variant for local testing (~50K parameters)
            self.configs = {
                "conv_channels": 8,
                "stage1": [(8, 8, 8), 1],  # LB: [8,8,8] × 1
                "stage2": [
                    (8, 16, 16),
                    1,
                    (16, 16, 16),
                    1,
                ],  # LB: [8,16,16] × 1, [16,16,16] × 1
                "stage3": [
                    (16, 24, 24),
                    1,
                    (24, 24, 24),
                    1,
                ],  # DLB: [16,24,24] × 1, [24,24,24] × 1
                "stage4": [
                    (24, 32, 32),
                    1,
                    (32, 32, 32),
                    1,
                ],  # DLB: [24,32,32] × 1, [32,32,32] × 1
                "final_channels": 128,
            }
        else:
            raise ValueError(f"Unknown EtinyNet variant: {variant}")

        # Build network
        self._build_network()

        # Set final_channels for serialization compatibility
        self.final_channels = self.configs["final_channels"]

    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _build_network(self):
        """Build the EtinyNet architecture."""
        configs = self.configs

        # Initial convolution (3x3, stride 2)
        self.conv_initial = nn.Conv2d(
            3, configs["conv_channels"], 3, stride=2, padding=1, bias=False
        )
        self.bn_initial = nn.BatchNorm2d(configs["conv_channels"])

        # Stage 1: Linear Depthwise Blocks
        stage1_config = configs["stage1"]
        in_ch, mid_ch, out_ch = stage1_config[0]
        num_blocks = stage1_config[1]

        stage1_layers = []
        for i in range(num_blocks):
            input_ch = configs["conv_channels"] if i == 0 else out_ch
            stride = 2 if i == 0 else 1  # First block has stride 2
            stage1_layers.append(LinearDepthwiseBlock(input_ch, mid_ch, out_ch, stride))

        self.stage1 = nn.Sequential(*stage1_layers)

        # Stage 2: Linear Depthwise Blocks
        stage2_config = configs["stage2"]
        stage2_layers = []

        # First LB block
        _, mid_ch, out_ch = stage2_config[0]
        num_blocks = stage2_config[1]
        prev_out_ch = configs["stage1"][0][2]  # Output from stage1

        for i in range(num_blocks):
            input_ch = prev_out_ch if i == 0 else out_ch
            stride = 2 if i == 0 else 1
            stage2_layers.append(LinearDepthwiseBlock(input_ch, mid_ch, out_ch, stride))

        # Second LB blocks
        _, mid_ch, out_ch = stage2_config[2]
        num_blocks = stage2_config[3]
        prev_out_ch = stage2_config[0][2]

        for i in range(num_blocks):
            input_ch = prev_out_ch if i == 0 else out_ch
            stage2_layers.append(LinearDepthwiseBlock(input_ch, mid_ch, out_ch, 1))

        self.stage2 = nn.Sequential(*stage2_layers)

        # Stage 3: Dense Linear Depthwise Blocks
        stage3_config = configs["stage3"]
        stage3_layers = []

        # First DLB block
        _, mid_ch, out_ch = stage3_config[0]
        num_blocks = stage3_config[1]
        prev_out_ch = stage2_config[2][2]  # Output from stage2

        for i in range(num_blocks):
            input_ch = prev_out_ch if i == 0 else out_ch
            stride = 2 if i == 0 else 1
            stage3_layers.append(
                DenseLinearDepthwiseBlock(input_ch, mid_ch, out_ch, stride)
            )

        # Second DLB blocks
        _, mid_ch, out_ch = stage3_config[2]
        num_blocks = stage3_config[3]
        prev_out_ch = stage3_config[0][2]

        for i in range(num_blocks):
            input_ch = prev_out_ch if i == 0 else out_ch
            stage3_layers.append(DenseLinearDepthwiseBlock(input_ch, mid_ch, out_ch, 1))

        self.stage3 = nn.Sequential(*stage3_layers)

        # Stage 4: Dense Linear Depthwise Blocks
        stage4_config = configs["stage4"]
        stage4_layers = []

        # First DLB block
        _, mid_ch, out_ch = stage4_config[0]
        num_blocks = stage4_config[1]
        prev_out_ch = stage3_config[2][2]  # Output from stage3

        for i in range(num_blocks):
            input_ch = prev_out_ch if i == 0 else out_ch
            stride = 2 if i == 0 else 1
            stage4_layers.append(
                DenseLinearDepthwiseBlock(input_ch, mid_ch, out_ch, stride)
            )

        # Second DLB blocks
        in_ch, mid_ch, out_ch = stage4_config[2]
        num_blocks = stage4_config[3]
        prev_out_ch = stage4_config[0][2]

        for i in range(num_blocks):
            input_ch = prev_out_ch if i == 0 else out_ch
            stage4_layers.append(DenseLinearDepthwiseBlock(input_ch, mid_ch, out_ch, 1))

        self.stage4 = nn.Sequential(*stage4_layers)

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Final conv layer to reach target channels
        final_in_ch = stage4_config[2][2]
        self.conv_final = nn.Conv2d(
            final_in_ch, configs["final_channels"], 1, bias=False
        )
        self.bn_final = nn.BatchNorm2d(configs["final_channels"])

        # Classification head
        self.classifier = nn.Linear(configs["final_channels"], self.num_classes)

    def forward(self, x):
        """Forward pass through EtinyNet."""
        # Initial convolution
        x = F.relu6(self.bn_initial(self.conv_initial(x)))

        # Forward through stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Final conv and global pooling
        x = F.relu6(self.bn_final(self.conv_final(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.classifier(x)

        return x


class NNUE(nn.Module):
    """
    NNUE model for computer vision

    Neural Network for Universal Evaluation adapted for computer vision tasks.
    """

    def __init__(
        self,
        feature_set: Optional[GridFeatureSet] = None,
        l1_size: int = DEFAULT_L1,
        l2_size: int = DEFAULT_L2,
        l3_size: int = DEFAULT_L3,
        loss_params=LossParams(),
        visual_threshold=0.0,
        num_classes=1,
        weight_decay=5e-4,
    ):
        super().__init__()

        # Architecture configuration
        if feature_set is None:
            feature_set = GridFeatureSet(grid_size=10, num_features_per_square=8)

        self.feature_set = feature_set
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.l3_size = l3_size
        self.visual_threshold = visual_threshold
        self.num_classes = num_classes
        self.loss_params = loss_params
        self.weight_decay = weight_decay

        # Calculate conv parameters
        conv_out_channels = feature_set.num_features_per_square
        input_size = 96
        target_grid_size = feature_set.grid_size
        conv_stride = max(1, (input_size + 2 * 1 - 3) // (target_grid_size - 1))

        # Convolutional frontend
        self.conv = nn.Conv2d(
            3,
            conv_out_channels,
            kernel_size=3,
            stride=conv_stride,
            padding=1,
            bias=False,
        )

        # Hardtanh activation for feature extraction
        self.hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)

        # Input layer (feature transformer)
        max_features = feature_set.num_features
        self.input = FeatureTransformer(max_features, l1_size)

        # Simple classifier (replaces chess-specific layer stacks)
        self.classifier = SimpleClassifier(l1_size, l2_size, l3_size, num_classes)

        # NNUE-to-score scaling factor
        self.nnue2score = nn.Parameter(torch.tensor(600.0))

    def _clip_weights(self):
        """Clip weights to expected ranges for quantization."""
        # Clip input layer weights to [-1, 1] range
        if hasattr(self.input, "weight"):
            with torch.no_grad():
                self.input.weight.clamp_(-1.0, 1.0)

        # Clip classifier weights to [-1, 1] range
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    module.weight.clamp_(-1.0, 1.0)

    def get_quantized_model_data(self):
        """Extract and quantize all NNUE model data for serialization."""
        self.eval()
        self._clip_weights()

        # Model metadata
        metadata = {
            "feature_set": self.feature_set,
            "L1": self.l1_size,
            "L2": self.l2_size,
            "L3": self.l3_size,
            "num_classes": self.num_classes,
            "visual_threshold": self.visual_threshold,
            "nnue2score": self.nnue2score.item(),
            "quantized_one": 127.0,
        }

        quantized_data = {"metadata": metadata}

        # Quantize conv layer
        quantized_data["conv_layer"] = {
            "weight": self.conv.weight.detach().cpu().numpy(),
            "bias": None,  # Conv layer has no bias
            "scale": 64.0,  # Default scale for quantization
        }

        # Quantize feature transformer
        quantized_data["feature_transformer"] = {
            "weight": self.input.weight.detach().cpu().numpy(),
            "bias": (
                self.input.bias.detach().cpu().numpy()
                if self.input.bias is not None
                else None
            ),
            "scale": 64.0,  # Default scale for quantization
        }

        # Quantize simple classifier (new architecture without buckets)
        linear_layers = [
            layer
            for layer in self.classifier.classifier
            if isinstance(layer, nn.Linear)
        ]

        classifier_data = {"layers": []}

        for i, layer in enumerate(linear_layers):
            layer_data = {
                "weight": layer.weight.detach().cpu().numpy(),
                "bias": (
                    layer.bias.detach().cpu().numpy()
                    if layer.bias is not None
                    else np.zeros(layer.weight.shape[0])
                ),
                "scale": 64.0,  # Default scale for quantization
            }
            classifier_data["layers"].append(layer_data)

        quantized_data["classifier"] = classifier_data

        return quantized_data

    def _to_sparse_features(self, binary_features: torch.Tensor):
        """Convert binary feature maps to sparse feature representation."""
        batch_size = binary_features.shape[0]

        # Flatten spatial dimensions
        binary_features_flat = binary_features.view(batch_size, -1)

        # Find non-zero features
        feature_indices_list = []
        feature_values_list = []

        for b in range(batch_size):
            nonzero_indices = torch.nonzero(binary_features_flat[b] > 0.5).squeeze(-1)
            nonzero_values = binary_features_flat[b][nonzero_indices]

            feature_indices_list.append(nonzero_indices)
            feature_values_list.append(nonzero_values)

        # Pad to same length
        max_features = (
            max(len(indices) for indices in feature_indices_list)
            if feature_indices_list
            else 1
        )
        max_features = max(max_features, 1)  # Ensure at least 1

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

    def forward(self, images: torch.Tensor):
        """Forward pass from images to evaluation scores."""
        # Convolution: (B, 3, H, W) -> (B, conv_out_channels, grid_h, grid_w)
        x = self.conv(images)

        # Apply Hardtanh activation
        x = self.hardtanh(x)

        # Apply threshold to get binary values
        if self.training:
            # Smooth approximation using sigmoid
            binary_features = torch.sigmoid(10.0 * (x - self.visual_threshold))
        else:
            # Hard threshold for inference
            binary_features = (x > self.visual_threshold).float()

        # Convert to sparse features for NNUE
        feature_indices, feature_values = self._to_sparse_features(binary_features)

        # Transform sparse features to dense representation
        features = self.input(feature_indices, feature_values)

        # Clamp to [0, 1] (keep this - it's part of quantization preparation)
        l0_ = torch.clamp(features, 0.0, 1.0)

        # Apply pairwise multiplication (core NNUE technique for feature interactions)
        # Split features in half and multiply element-wise to create quadratic interactions
        l0_s = torch.split(l0_, self.l1_size // 2, dim=1)
        l0_s1 = (
            l0_s[0] * l0_s[1]
        )  # Element-wise multiplication creates feature interactions

        # Concatenate multiplied features with original half (standard NNUE approach)
        l0_ = torch.cat([l0_s1, l0_s[0]], dim=1) * (127 / 128)

        # Pass through simple classifier
        x = self.classifier(l0_)

        return x


class FeatureTransformer(nn.Module):
    """Feature transformer for NNUE (input layer)."""

    def __init__(self, num_features: int, output_size: int):
        super().__init__()
        self.num_features = num_features
        self.output_size = output_size

        # Embedding layer for sparse features
        self.weight = nn.Parameter(torch.randn(num_features, output_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, feature_indices: torch.Tensor, feature_values: torch.Tensor):
        """Transform sparse features to dense representation."""
        batch_size = feature_indices.shape[0]

        # Initialize output
        output = self.bias.unsqueeze(0).expand(batch_size, -1).clone()

        # Add contributions from active features
        for b in range(batch_size):
            valid_mask = feature_indices[b] >= 0
            valid_indices = feature_indices[b][valid_mask]
            valid_values = feature_values[b][valid_mask]

            if len(valid_indices) > 0:
                # Clamp indices to valid range
                valid_indices = torch.clamp(valid_indices, 0, self.num_features - 1)
                feature_weights = self.weight[
                    valid_indices
                ]  # [num_active, output_size]
                contributions = (feature_weights * valid_values.unsqueeze(-1)).sum(
                    dim=0
                )
                output[b] += contributions

        return output


class SimpleClassifier(nn.Module):
    """Simple classifier for NNUE computer vision (replaces chess-specific layer stacks)."""

    def __init__(
        self,
        l1_size: int,
        l2_size: int,
        l3_size: int,
        num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Single classifier network (no buckets needed for vision)
        # Input size is l1_size because of pairwise multiplication: l1_size/2 * l1_size/2 + l1_size/2 = l1_size
        self.classifier = nn.Sequential(
            nn.Linear(l1_size, l2_size),
            nn.ReLU(),
            nn.Linear(l2_size, l3_size),
            nn.ReLU(),
            nn.Linear(l3_size, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through the classifier."""
        return self.classifier(x)
