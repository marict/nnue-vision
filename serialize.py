#!/usr/bin/env python3
"""
Serialize NNUE PyTorch models to .nnue format for C++ deployment.
Serialize EtinyNet PyTorch models to .etiny format for C++ deployment.

This script converts trained NNUE and EtinyNet models to the binary formats expected by
C++ implementations, with proper quantization and optimization.
"""

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from model import NNUE, EtinyNet, GridFeatureSet


def write_nnue_header(f, metadata: Dict[str, Any]) -> None:
    """Write the NNUE file header with model metadata."""
    # Magic number for NNUE files (4 bytes)
    f.write(b"NNUE")

    # Version (4 bytes) - increment to 2 for conv layer support
    f.write(struct.pack("<I", 2))

    # Architecture metadata
    feature_set = metadata["feature_set"]
    f.write(struct.pack("<I", feature_set.num_features))  # Input features
    f.write(struct.pack("<I", metadata["L1"]))  # L1 size
    f.write(struct.pack("<I", metadata["L2"]))  # L2 size
    f.write(struct.pack("<I", metadata["L3"]))  # L3 size
    f.write(struct.pack("<I", metadata["num_ls_buckets"]))  # Number of layer stacks

    # Quantization parameters
    f.write(struct.pack("<f", metadata["nnue2score"]))
    f.write(struct.pack("<f", metadata["quantized_one"]))

    # Visual processing parameters (new in version 2)
    f.write(struct.pack("<f", metadata["visual_threshold"]))


def write_etinynet_header(f, metadata: Dict[str, Any]) -> None:
    """Write the EtinyNet file header with model metadata."""
    # Magic number for EtinyNet files (4 bytes)
    f.write(b"ETNY")

    # Version (4 bytes)
    f.write(struct.pack("<I", 1))

    # Architecture metadata
    variant_str = metadata["variant"].encode("utf-8")
    f.write(struct.pack("<I", len(variant_str)))
    f.write(variant_str)  # Variant string ("1.0" or "0.75")

    f.write(struct.pack("<I", metadata["num_classes"]))  # Number of classes
    f.write(struct.pack("<I", metadata["input_size"]))  # Input image size
    f.write(struct.pack("<I", metadata["conv_channels"]))  # Initial conv channels
    f.write(struct.pack("<I", metadata["final_channels"]))  # Final feature channels

    # Quantization parameters (if ASQ is used)
    f.write(struct.pack("<?", metadata["use_asq"]))  # Boolean for ASQ usage
    if metadata["use_asq"]:
        f.write(struct.pack("<I", metadata["asq_bits"]))
        f.write(struct.pack("<f", metadata["lambda_param"]))


def write_conv_layer(f, conv_data: Dict[str, Any]) -> None:
    """Write quantized convolutional layer weights and biases."""
    weight = conv_data[
        "weight"
    ]  # int8, shape (out_channels, in_channels, kernel_h, kernel_w)
    bias = conv_data["bias"]  # int32, shape (out_channels,)
    scale = conv_data["scale"]

    # Write inner layer type identifier for ConvLayer as expected by C++ loader
    f.write(struct.pack("<I", 0))  # 0 = STANDARD_CONV

    # Write scale
    f.write(struct.pack("<f", scale))

    # Write weight dimensions (out_channels, in_channels, kernel_h, kernel_w)
    f.write(struct.pack("<I", weight.shape[0]))  # out_channels
    f.write(struct.pack("<I", weight.shape[1]))  # in_channels (should be 3)
    f.write(struct.pack("<I", weight.shape[2]))  # kernel_height (should be 3)
    f.write(struct.pack("<I", weight.shape[3]))  # kernel_width (should be 3)

    # Write weights (int8, little endian)
    weight_bytes = weight.cpu().numpy().astype("i1").tobytes()
    f.write(weight_bytes)

    # Write bias dimensions and data
    f.write(struct.pack("<I", bias.shape[0]))  # out_channels
    bias_bytes = bias.cpu().numpy().astype("<i4").tobytes()
    f.write(bias_bytes)


def write_depthwise_separable_layer(f, layer_data: Dict[str, Any]) -> None:
    """Write quantized LinearDepthwiseBlock layer."""

    # Write scales for all 4 components (required by C++ loader)
    f.write(struct.pack("<f", layer_data["depthwise_scale"]))  # dconv1_scale
    f.write(struct.pack("<f", layer_data["pointwise_scale"]))  # pconv_scale
    f.write(struct.pack("<f", layer_data["depthwise2_scale"]))  # dconv2_scale
    f.write(struct.pack("<f", layer_data["pointwise_out_scale"]))  # pconv_out_scale

    # Write dimensions (required by C++ loader)
    pw_weight = layer_data["pointwise_weight"]
    pw_out_weight = layer_data["pointwise_out_weight"]

    in_channels = pw_weight.shape[1]  # Input channels (from pconv)
    mid_channels = pw_weight.shape[0]  # Mid channels (from pconv output)
    out_channels = pw_out_weight.shape[0]  # Output channels (from pconv_out)
    stride = layer_data["stride"]

    f.write(struct.pack("<I", in_channels))
    f.write(struct.pack("<I", mid_channels))
    f.write(struct.pack("<I", out_channels))
    f.write(struct.pack("<I", stride))

    # Write dconv1 weights (first depthwise conv)
    dw1_weight = layer_data["depthwise_weight"]
    dw1_weight_count = dw1_weight.shape[0] * 9  # 3x3 kernel
    f.write(dw1_weight.cpu().numpy().astype("i1").tobytes())

    # Write pconv weights (first pointwise conv)
    pw_weight_count = mid_channels * in_channels
    f.write(pw_weight.cpu().numpy().astype("i1").tobytes())

    # Write pconv biases
    pw_bias = layer_data["pointwise_bias"]
    f.write(struct.pack("<I", pw_bias.shape[0]))
    f.write(pw_bias.cpu().numpy().astype("<i4").tobytes())

    # Write dconv2 weights (second depthwise conv)
    dw2_weight = layer_data["depthwise2_weight"]
    dw2_weight_count = mid_channels * 9  # 3x3 kernel
    f.write(dw2_weight.cpu().numpy().astype("i1").tobytes())

    # Write pconv_out weights (final pointwise conv)
    pw_out_weight = layer_data["pointwise_out_weight"]
    pw_out_weight_count = out_channels * mid_channels
    f.write(pw_out_weight.cpu().numpy().astype("i1").tobytes())

    # Write pconv_out biases
    pw_out_bias = layer_data["pointwise_out_bias"]
    f.write(struct.pack("<I", pw_out_bias.shape[0]))
    f.write(pw_out_bias.cpu().numpy().astype("<i4").tobytes())


def write_linear_layer(f, layer_data: Dict[str, Any]) -> None:
    """Write quantized linear layer (for classifier)."""
    weight = layer_data["weight"]  # int8
    bias = layer_data["bias"]  # int32
    scale = layer_data["scale"]

    # Write scale
    f.write(struct.pack("<f", scale))

    # Write dimensions (in_features, out_features)
    f.write(struct.pack("<I", weight.shape[1]))  # in_features
    f.write(struct.pack("<I", weight.shape[0]))  # out_features

    # Write weights (int8, little endian)
    f.write(weight.cpu().numpy().astype("i1").tobytes())

    # Write bias dimensions and data
    f.write(struct.pack("<I", bias.shape[0]))  # Should equal out_features
    f.write(bias.cpu().numpy().astype("<i4").tobytes())


def quantize_conv_layer(conv_layer, scale=64.0):
    """Quantize a convolutional layer."""
    weight = conv_layer.weight.data
    bias = (
        conv_layer.bias.data
        if conv_layer.bias is not None
        else torch.zeros(conv_layer.out_channels)
    )

    # Quantize weights to int8
    weight_q = torch.round(weight * scale).clamp(-127, 127).to(torch.int8)
    bias_q = torch.round(bias * scale).to(torch.int32)

    return {"weight": weight_q, "bias": bias_q, "scale": scale}


def quantize_linear_layer(linear_layer, scale=64.0):
    """Quantize a linear layer."""
    weight = linear_layer.weight.data
    bias = (
        linear_layer.bias.data
        if linear_layer.bias is not None
        else torch.zeros(linear_layer.out_features)
    )

    # Quantize weights to int8
    weight_q = torch.round(weight * scale).clamp(-127, 127).to(torch.int8)
    bias_q = torch.round(bias * scale).to(torch.int32)

    return {"weight": weight_q, "bias": bias_q, "scale": scale}


def quantize_linear_depthwise_block(block, scale=64.0):
    """Quantize a Linear Depthwise Block."""
    data = {}

    # Determine block type
    if hasattr(block, "linear_block"):  # DenseLinearDepthwiseBlock
        data["layer_type"] = 2
        actual_block = block.linear_block
        data["use_skip"] = block.use_skip
    else:  # LinearDepthwiseBlock
        data["layer_type"] = 1
        actual_block = block
        data["use_skip"] = False

    # Get stride from first depthwise conv
    data["stride"] = actual_block.dconv1.stride[0]

    # Quantize depthwise conv1 (no bias)
    dw1_weight = actual_block.dconv1.weight.data
    dw1_weight_q = torch.round(dw1_weight * scale).clamp(-127, 127).to(torch.int8)

    data["depthwise_weight"] = dw1_weight_q
    data["depthwise_scale"] = scale

    # Quantize pointwise conv (with bias)
    pw_weight = actual_block.pconv.weight.data
    pw_bias = actual_block.pconv.bias.data
    pw_weight_q = torch.round(pw_weight * scale).clamp(-127, 127).to(torch.int8)
    pw_bias_q = torch.round(pw_bias * scale).to(torch.int32)

    data["pointwise_weight"] = pw_weight_q
    data["pointwise_bias"] = pw_bias_q
    data["pointwise_scale"] = scale

    # Quantize depthwise conv2 (no bias)
    dw2_weight = actual_block.dconv2.weight.data
    dw2_weight_q = torch.round(dw2_weight * scale).clamp(-127, 127).to(torch.int8)

    data["depthwise2_weight"] = dw2_weight_q
    data["depthwise2_scale"] = scale

    # Quantize final pointwise conv (with bias)
    pout_weight = actual_block.pconv_out.weight.data
    pout_bias = actual_block.pconv_out.bias.data
    pout_weight_q = torch.round(pout_weight * scale).clamp(-127, 127).to(torch.int8)
    pout_bias_q = torch.round(pout_bias * scale).to(torch.int32)

    data["pointwise_out_weight"] = pout_weight_q
    data["pointwise_out_bias"] = pout_bias_q
    data["pointwise_out_scale"] = scale

    return data


def get_etinynet_quantized_data(model: EtinyNet):
    """Extract and quantize all EtinyNet layer data."""
    model.eval()

    quantized_data = {}

    # Model metadata
    quantized_data["metadata"] = {
        "variant": model.variant,
        "num_classes": model.num_classes,
        "input_size": model.input_size,
        "conv_channels": model.configs["conv_channels"],
        "final_channels": model.final_channels,
        "use_asq": model.use_asq,
        "asq_bits": getattr(model, "asq", {}).get("bits", 4) if model.use_asq else 4,
        "lambda_param": model.asq.lambda_param.item() if model.use_asq else 2.0,
    }

    # Extract and quantize layers from each stage
    layers = []

    # Stage 1: Extract initial conv layer and LinearDepthwiseBlocks
    # stage1 structure: Sequential(Conv2d, ReLU, MaxPool2d, LinearDepthwiseBlock, LinearDepthwiseBlock, ...)
    stage1_list = list(model.stage1.children())

    # Initial conv layer (first module in stage1)
    initial_conv = stage1_list[0]  # Conv2d
    initial_conv_data = quantize_conv_layer(initial_conv)
    initial_conv_data["layer_type"] = 0  # Standard conv
    layers.append(initial_conv_data)

    # Extract LinearDepthwiseBlocks from stage1 (skip Conv2d, ReLU, MaxPool2d)
    for module in stage1_list[3:]:  # Skip Conv2d, ReLU, MaxPool2d
        if hasattr(module, "dconv1"):  # LinearDepthwiseBlock
            layer_data = quantize_linear_depthwise_block(module)
            layers.append(layer_data)

    # Extract blocks from stage2, stage3, stage4
    for stage in [model.stage2, model.stage3, model.stage4]:
        for module in stage.children():  # Use .children() not .modules()
            if hasattr(module, "dconv1"):  # LinearDepthwiseBlock
                layer_data = quantize_linear_depthwise_block(module)
                layers.append(layer_data)
            elif hasattr(module, "linear_block"):  # DenseLinearDepthwiseBlock
                layer_data = quantize_linear_depthwise_block(module)
                layers.append(layer_data)

    quantized_data["layers"] = layers

    # Quantize classifier (Linear layer, not conv)
    classifier_data = quantize_linear_layer(model.classifier)
    classifier_data["layer_type"] = 3  # Linear layer
    quantized_data["classifier"] = classifier_data

    return quantized_data


def serialize_etinynet_model(model: EtinyNet, output_path: Path) -> None:
    """
    Serialize an EtinyNet model to .etiny binary format.

    Args:
        model: Trained EtinyNet model
        output_path: Path to write .etiny file
    """
    model.eval()

    # Get quantized model data
    quantized_data = get_etinynet_quantized_data(model)

    with open(output_path, "wb") as f:
        # Write header
        write_etinynet_header(f, quantized_data["metadata"])

        # Write number of layers (including classifier)
        total_layers = len(quantized_data["layers"]) + 1  # +1 for classifier
        f.write(struct.pack("<I", total_layers))

        # Write each layer
        for layer_data in quantized_data["layers"]:
            # Write the layer type identifier first
            f.write(struct.pack("<I", layer_data["layer_type"]))

            if layer_data["layer_type"] == 0:  # Standard conv
                write_conv_layer(f, layer_data)
            elif layer_data["layer_type"] in [1, 2]:  # LB or DLB
                write_depthwise_separable_layer(f, layer_data)

        # Write classifier as final layer (preceded by its type id)
        f.write(struct.pack("<I", quantized_data["classifier"]["layer_type"]))
        write_linear_layer(f, quantized_data["classifier"])

    print(f"Successfully serialized EtinyNet to {output_path}")


def write_feature_transformer(f, ft_data: Dict[str, Any]) -> None:
    """Write quantized feature transformer weights and biases."""
    weight = ft_data["weight"]  # int16
    bias = ft_data["bias"]  # int32
    scale = ft_data["scale"]

    # Write scale
    f.write(struct.pack("<f", scale))

    # Write weight dimensions
    f.write(struct.pack("<I", weight.shape[0]))  # num_features
    f.write(struct.pack("<I", weight.shape[1]))  # output_size (L1)

    # Write weights (int16, little endian)
    weight_bytes = weight.cpu().numpy().astype("<i2").tobytes()
    f.write(weight_bytes)

    # Write bias dimensions and data
    f.write(struct.pack("<I", bias.shape[0]))  # output_size (L1)
    bias_bytes = bias.cpu().numpy().astype("<i4").tobytes()
    f.write(bias_bytes)


def write_layer_stack(f, ls_data: Dict[str, Any]) -> None:
    """Write quantized layer stack weights and biases."""
    scales = ls_data["scales"]

    # Write scales (including factorization scale)
    f.write(struct.pack("<f", scales["l1"]))
    f.write(struct.pack("<f", scales["l2"]))
    f.write(struct.pack("<f", scales["output"]))
    f.write(struct.pack("<f", scales["l1_fact"]))

    # L1 layer
    l1_weight = ls_data["l1_weight"]  # int8
    l1_bias = ls_data["l1_bias"]  # int32

    f.write(struct.pack("<I", l1_weight.shape[0]))  # output_size (L2)
    f.write(struct.pack("<I", l1_weight.shape[1]))  # input_size (L1)
    f.write(l1_weight.cpu().numpy().astype("i1").tobytes())

    f.write(struct.pack("<I", l1_bias.shape[0]))
    f.write(l1_bias.cpu().numpy().astype("<i4").tobytes())

    # L1 factorization layer
    l1_fact_weight = ls_data["l1_fact_weight"]  # int8
    l1_fact_bias = ls_data["l1_fact_bias"]  # int32

    f.write(struct.pack("<I", l1_fact_weight.shape[0]))  # output_size (L2+1)
    f.write(struct.pack("<I", l1_fact_weight.shape[1]))  # input_size (L1)
    f.write(l1_fact_weight.cpu().numpy().astype("i1").tobytes())

    f.write(struct.pack("<I", l1_fact_bias.shape[0]))
    f.write(l1_fact_bias.cpu().numpy().astype("<i4").tobytes())

    # L2 layer
    l2_weight = ls_data["l2_weight"]  # int8
    l2_bias = ls_data["l2_bias"]  # int32

    f.write(struct.pack("<I", l2_weight.shape[0]))  # output_size (L3)
    f.write(struct.pack("<I", l2_weight.shape[1]))  # input_size (L2 * 2)
    f.write(l2_weight.cpu().numpy().astype("i1").tobytes())

    f.write(struct.pack("<I", l2_bias.shape[0]))
    f.write(l2_bias.cpu().numpy().astype("<i4").tobytes())

    # Output layer
    output_weight = ls_data["output_weight"]  # int8
    output_bias = ls_data["output_bias"]  # int32

    f.write(struct.pack("<I", output_weight.shape[0]))  # output_size (1)
    f.write(struct.pack("<I", output_weight.shape[1]))  # input_size (L3)
    f.write(output_weight.cpu().numpy().astype("i1").tobytes())

    f.write(struct.pack("<I", output_bias.shape[0]))
    f.write(output_bias.cpu().numpy().astype("<i4").tobytes())


def serialize_model(model: NNUE, output_path: Path) -> None:
    """
    Serialize an NNUE model to .nnue binary format.

    Args:
        model: Trained NNUE model with configurable architecture
        output_path: Path to write .nnue file
    """
    # Ensure model is in eval mode and weights are clipped
    model.eval()
    model._clip_weights()

    # Get quantized model data
    quantized_data = model.get_quantized_model_data()

    with open(output_path, "wb") as f:
        # Write header
        write_nnue_header(f, quantized_data["metadata"])

        # Write convolutional layer (new in version 2)
        write_conv_layer(f, quantized_data["conv_layer"])

        # Write feature transformer
        write_feature_transformer(f, quantized_data["feature_transformer"])

        # Write layer stacks
        num_ls_buckets = quantized_data["metadata"]["num_ls_buckets"]
        for i in range(num_ls_buckets):
            write_layer_stack(f, quantized_data[f"layer_stack_{i}"])

    print(f"Successfully serialized model to {output_path}")


def load_model_from_checkpoint(checkpoint_path: Path) -> NNUE:
    """Load NNUE model from PyTorch checkpoint or state dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle both full checkpoints and state dicts
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Try to extract hyperparameters from checkpoint
        # If they're not available, we'll infer them from the state dict
        saved_num_ls_buckets = checkpoint.get("num_ls_buckets")
        saved_feature_set = checkpoint.get("feature_set")
        saved_l1_size = checkpoint.get("l1_size")
        saved_l2_size = checkpoint.get("l2_size")
        saved_l3_size = checkpoint.get("l3_size")

        # If we have all saved parameters, use them
        if all(
            param is not None
            for param in [
                saved_num_ls_buckets,
                saved_feature_set,
                saved_l1_size,
                saved_l2_size,
                saved_l3_size,
            ]
        ):
            feature_set = saved_feature_set
            l1_size = saved_l1_size
            l2_size = saved_l2_size
            l3_size = saved_l3_size
            num_ls_buckets = saved_num_ls_buckets
        else:
            # Infer architecture from state dict
            feature_set, l1_size, l2_size, l3_size, num_ls_buckets = (
                infer_architecture_from_state_dict(state_dict)
            )
    else:
        state_dict = checkpoint
        # Infer model architecture from state dict shapes
        feature_set, l1_size, l2_size, l3_size, num_ls_buckets = (
            infer_architecture_from_state_dict(state_dict)
        )

    # Create model with inferred parameters
    model = NNUE(
        feature_set=feature_set,
        l1_size=l1_size,
        l2_size=l2_size,
        l3_size=l3_size,
        num_ls_buckets=num_ls_buckets,
    )

    # Handle Lightning checkpoints by removing the 'nnue.' prefix
    if any(key.startswith("nnue.") for key in state_dict.keys()):
        state_dict = {
            k.replace("nnue.", ""): v
            for k, v in state_dict.items()
            if k.startswith("nnue.")
        }

    model.load_state_dict(state_dict)

    return model


def detect_model_type(checkpoint_path: Path) -> str:
    """Detect whether checkpoint contains NNUE or EtinyNet model."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Get state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Look for EtinyNet-specific layer names
    etinynet_indicators = [
        "stage1",
        "stage2",
        "stage3",
        "stage4",
        "global_pool",
        "classifier",
        "dconv1",
        "dconv2",
        "pconv",
        "pconv_out",
    ]

    # Look for NNUE-specific layer names
    nnue_indicators = [
        "input.weight",
        "input.bias",
        "layer_stacks",
        "conv.weight",
        "nnue.input.weight",
        "nnue.layer_stacks",
    ]

    # Check for EtinyNet patterns
    for key in state_dict.keys():
        for indicator in etinynet_indicators:
            if indicator in key:
                return "etinynet"

    # Check for NNUE patterns
    for key in state_dict.keys():
        for indicator in nnue_indicators:
            if indicator in key:
                return "nnue"

    # Default to NNUE if unclear
    return "nnue"


def load_etinynet_from_checkpoint(checkpoint_path: Path) -> EtinyNet:
    """Load EtinyNet model from PyTorch checkpoint or state dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Get state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Try to extract hyperparameters
        variant = checkpoint.get("variant", "1.0")
        num_classes = checkpoint.get("num_classes", 1000)
        input_size = checkpoint.get("input_size", 112)
        use_asq = checkpoint.get("use_asq", False)
        asq_bits = checkpoint.get("asq_bits", 4)
    else:
        state_dict = checkpoint
        # Use defaults - try to infer from state dict
        variant = infer_etinynet_variant_from_state_dict(state_dict)
        num_classes = infer_num_classes_from_state_dict(state_dict)
        input_size = 112  # Default
        use_asq = False  # Default
        asq_bits = 4

    # Create model
    model = EtinyNet(
        variant=variant,
        num_classes=num_classes,
        input_size=input_size,
        use_asq=use_asq,
        asq_bits=asq_bits,
    )

    model.load_state_dict(state_dict)
    return model


def infer_etinynet_variant_from_state_dict(state_dict) -> str:
    """Infer EtinyNet variant (1.0 or 0.75) from state dict."""
    # Look at initial conv layer output channels
    for key in state_dict.keys():
        if "stage1.0.weight" in key:  # Initial conv layer
            conv_weight = state_dict[key]
            out_channels = conv_weight.shape[0]
            if out_channels == 32:
                return "1.0"
            elif out_channels == 24:
                return "0.75"

    # Default to 1.0
    return "1.0"


def infer_num_classes_from_state_dict(state_dict) -> int:
    """Infer number of output classes from classifier layer."""
    for key in state_dict.keys():
        if "classifier.weight" in key:
            classifier_weight = state_dict[key]
            return classifier_weight.shape[0]

    # Default to 1000 (ImageNet)
    return 1000


def load_model_from_checkpoint_auto(checkpoint_path: Path):
    """Automatically detect and load NNUE or EtinyNet model."""
    model_type = detect_model_type(checkpoint_path)

    if model_type == "etinynet":
        return load_etinynet_from_checkpoint(checkpoint_path)
    else:
        return load_model_from_checkpoint(checkpoint_path)


def infer_architecture_from_state_dict(
    state_dict,
) -> tuple[GridFeatureSet, int, int, int, int]:
    """Infer model architecture from state dict tensor shapes."""

    # Handle Lightning checkpoints with 'nnue.' prefix
    prefix = ""
    if "nnue.input.weight" in state_dict:
        prefix = "nnue."
    elif "input.weight" not in state_dict:
        raise ValueError(
            "Cannot find NNUE model weights in state dict. Available keys: "
            + str(list(state_dict.keys())[:10])
        )

    # Get input layer weight shape to determine feature set and L1
    input_weight_shape = state_dict[f"{prefix}input.weight"].shape
    num_features = input_weight_shape[0]  # [num_features, L1]
    l1_size = input_weight_shape[1]

    # Infer number of layer stack buckets from output layer shape FIRST
    output_weight_shape = state_dict[f"{prefix}layer_stacks.output.weight"].shape
    num_ls_buckets = output_weight_shape[0]  # [num_buckets, L3]

    # Infer grid size and features per square from conv layer
    conv_weight_shape = state_dict[f"{prefix}conv.weight"].shape
    conv_out_channels = conv_weight_shape[0]  # This should match features_per_square

    # Calculate grid size from total features and conv output channels
    grid_size = int((num_features / conv_out_channels) ** 0.5)

    if grid_size * grid_size * conv_out_channels != num_features:
        # Fallback: try common configurations
        common_configs = [
            (4, 6),  # 4x4 grid, 6 features per square = 96 total
            (8, 12),  # 8x8 grid, 12 features per square = 768 total
            (16, 8),  # 16x16 grid, 8 features per square = 2048 total
            (32, 64),  # 32x32 grid, 64 features per square = 65536 total
        ]

        for grid_size, features_per_square in common_configs:
            if grid_size * grid_size * features_per_square == num_features:
                break
        else:
            # Last resort: assume square grid with conv_out_channels features per square
            grid_size = int((num_features / conv_out_channels) ** 0.5)
            features_per_square = conv_out_channels
    else:
        features_per_square = conv_out_channels

    feature_set = GridFeatureSet(grid_size, features_per_square)

    # Infer L2 and L3 sizes from layer stack shapes
    # L2 size from l1_fact layer: l1_fact has shape [L2+1, L1]
    l1_fact_weight_shape = state_dict[f"{prefix}layer_stacks.l1_fact.weight"].shape
    l2_size = l1_fact_weight_shape[0] - 1  # Remove the +1 from factorization

    # L3 size from output layer: output has shape [num_buckets, L3]
    l3_size = output_weight_shape[1]  # L3 size

    # Verify with L2 layer shape (should be [L3 * num_buckets, L2 * 2])
    l2_weight_shape = state_dict[f"{prefix}layer_stacks.l2.weight"].shape
    expected_l2_input_size = l2_size * 2  # Due to squared concatenation
    expected_l2_output_size = l3_size * num_ls_buckets

    if (
        l2_weight_shape[0] != expected_l2_output_size
        or l2_weight_shape[1] != expected_l2_input_size
    ):
        # Fall back to inferring from L2 layer directly
        l3_size = l2_weight_shape[0] // num_ls_buckets
        l2_size = l2_weight_shape[1] // 2  # Account for squared concatenation

    return feature_set, l1_size, l2_size, l3_size, num_ls_buckets


def run_etinynet_cpp(model_path: Path, image: torch.Tensor) -> np.ndarray:  # type: ignore
    """Run the C++ EtinyNet engine on a single image and return logits.

    This helper builds the `etinynet_inference` executable if necessary and
    invokes it to get the engine output for comparison with PyTorch.

    Args:
        model_path: Path to .etiny model file.
        image: 3×H×W image tensor in 0-1 range (float32, on CPU).

    Returns:
        NumPy array of logits from the C++ engine.
    """
    # Ensure image is on CPU and contiguous
    image_cpu = image.detach().to(torch.float32).cpu().contiguous()
    h, w = image_cpu.shape[1:]

    # Write image to temporary binary file (float32 little-endian)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f_img:
        img_path = Path(f_img.name)
        f_img.write(image_cpu.numpy().tobytes())

    # Build the executable if it does not exist
    exec_path = Path("engine/build/etinynet_inference")
    if not exec_path.exists():
        # Trigger CMake build (assumes build directory already configured)
        build_cmd = ["cmake", "-S", "engine", "-B", "engine/build"]
        subprocess.run(build_cmd, check=True)
        build_cmd = [
            "cmake",
            "--build",
            "engine/build",
            "--target",
            "etinynet_inference",
        ]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to build etinynet_inference executable: {result.stderr}"
            )

    # Run the executable
    run_cmd = [
        str(exec_path),
        str(model_path),
        str(img_path),
        str(h),
        str(w),
    ]
    result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"EtinyNet engine failed: {result.stderr}")

    # Parse results
    logits = []
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_"):
            parts = line.split(":")
            if len(parts) == 2:
                value = float(parts[1].strip())
                logits.append(value)
    if not logits:
        raise RuntimeError("No RESULT_ lines found in engine output")

    # Clean up temp file
    img_path.unlink(missing_ok=True)

    return np.array(logits, dtype=np.float32)


def main():
    """Main serialization entry point."""
    parser = argparse.ArgumentParser(
        description="Serialize NNUE or EtinyNet model to binary format"
    )
    parser.add_argument("input", type=Path, help="Input model file (.pt or .ckpt)")
    parser.add_argument(
        "output", type=Path, help="Output binary file path (.nnue or .etiny)"
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Feature set specification (will be auto-detected if not provided)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["auto", "nnue", "etinynet"],
        default="auto",
        help="Force specific model type (auto-detect by default)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Detect or use specified model type
    if args.model_type == "auto":
        model_type = detect_model_type(args.input)
        print(f"Auto-detected model type: {model_type}")
    else:
        model_type = args.model_type
        print(f"Using specified model type: {model_type}")

    # Load model based on type
    print(f"Loading model from {args.input}")
    if model_type == "etinynet":
        model = load_etinynet_from_checkpoint(args.input)

        print(f"Detected EtinyNet architecture:")
        print(f"  Variant: EtinyNet-{model.variant}")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Estimated FLOPs: {model.count_flops():,}")
        print(f"  Input size: {model.input_size}x{model.input_size}")
        print(f"  Classes: {model.num_classes}")
        print(f"  ASQ enabled: {model.use_asq}")

        # Determine output format
        if args.output.suffix not in [".etiny", ".bin"]:
            # Auto-determine extension
            output_path = args.output.with_suffix(".etiny")
        else:
            output_path = args.output

        # Serialize to .etiny format
        print(f"Serializing EtinyNet to {output_path}")
        serialize_etinynet_model(model, output_path)

    else:  # NNUE model
        model = load_model_from_checkpoint(args.input)

        print(f"Detected NNUE architecture:")
        print(
            f"  Feature set: {model.feature_set.name} ({model.feature_set.num_features} features)"
        )
        print(
            f"  Layer sizes: {model.l1_size} -> {model.l2_size} -> {model.l3_size} -> 1"
        )
        print(f"  Layer stack buckets: {model.num_ls_buckets}")

        # Determine output format
        if args.output.suffix not in [".nnue", ".bin"]:
            # Auto-determine extension
            output_path = args.output.with_suffix(".nnue")
        else:
            output_path = args.output

        # Serialize to .nnue format
        print(f"Serializing NNUE to {output_path}")
        serialize_model(model, output_path)

    print("Serialization complete!")


if __name__ == "__main__":
    main()
