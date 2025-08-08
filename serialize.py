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

from nnue import NNUE, EtinyNet, GridFeatureSet


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
    f.write(struct.pack("<I", 1))  # num_ls_buckets (default to 1 for new architecture)

    # Quantization parameters
    f.write(struct.pack("<f", metadata["nnue2score"]))
    f.write(struct.pack("<f", metadata["quantized_one"]))
    f.write(struct.pack("<f", 0.1))  # visual_threshold (default value)


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
    if hasattr(weight, "cpu"):
        weight_bytes = weight.cpu().numpy().astype("i1").tobytes()
    else:
        weight_bytes = weight.astype("i1").tobytes()
    f.write(weight_bytes)

    # Write bias dimensions and data
    f.write(struct.pack("<I", bias.shape[0]))  # out_channels
    if hasattr(bias, "cpu"):
        bias_bytes = bias.cpu().numpy().astype("<i4").tobytes()
    else:
        bias_bytes = bias.astype("<i4").tobytes()
    f.write(bias_bytes)


def write_depthwise_separable_layer(f, layer_data: Dict[str, Any]) -> None:
    """Write quantized LinearDepthwiseBlock layer (corrected to match Python architecture)."""

    # Write scales for the 3 components (pw_expand, dw_conv, pw_project) + unused scale
    f.write(struct.pack("<f", layer_data["pointwise_scale"]))  # pw_expand_scale
    f.write(struct.pack("<f", layer_data["depthwise2_scale"]))  # dw_conv_scale
    f.write(struct.pack("<f", layer_data["pointwise_out_scale"]))  # pw_project_scale

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

    # Write pw_expand weights (pointwise expansion: in_channels -> mid_channels)
    pw_expand_weight = layer_data[
        "pointwise_weight"
    ]  # This is actually pw_expand in our mapping
    f.write(pw_expand_weight.cpu().numpy().astype("i1").tobytes())

    # Write pw_expand biases
    pw_expand_bias = layer_data["pointwise_bias"]
    f.write(struct.pack("<I", pw_expand_bias.shape[0]))
    f.write(pw_expand_bias.cpu().numpy().astype("<i4").tobytes())

    # Write dw_conv weights (depthwise conv: mid_channels -> mid_channels)
    dw_conv_weight = layer_data[
        "depthwise2_weight"
    ]  # This is actually dw_conv in our mapping
    f.write(dw_conv_weight.cpu().numpy().astype("i1").tobytes())

    # Write pw_project weights (pointwise projection: mid_channels -> out_channels)
    pw_project_weight = layer_data["pointwise_out_weight"]
    f.write(pw_project_weight.cpu().numpy().astype("i1").tobytes())

    # Write bias count and data (pw_project has no bias, so write zeros)
    out_channels = layer_data["pointwise_out_weight"].shape[0]
    f.write(struct.pack("<I", out_channels))
    bias_data = np.zeros(out_channels, dtype=np.int32)
    f.write(bias_data.tobytes())


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

    # Get stride from depthwise conv
    data["stride"] = actual_block.dw_conv.stride[0]

    # Our LinearDepthwiseBlock structure: pw_expand -> dw_conv -> pw_project
    # Map correctly to serialization format:

    # Quantize pointwise expansion (pw_expand): in_channels -> mid_channels
    pw_expand_weight = actual_block.pw_expand.weight.data
    pw_expand_weight_q = (
        torch.round(pw_expand_weight * scale).clamp(-127, 127).to(torch.int8)
    )
    # pw_expand uses BatchNorm bias, but C++ engine expects explicit bias
    mid_channels = pw_expand_weight.shape[0]
    pw_expand_bias_q = torch.zeros(mid_channels, dtype=torch.int32)

    data["pointwise_weight"] = pw_expand_weight_q  # pw_expand weights
    data["pointwise_bias"] = pw_expand_bias_q  # pw_expand biases
    data["pointwise_scale"] = scale

    # Quantize depthwise conv (dw_conv): mid_channels -> mid_channels (groups=mid_channels)
    dw_weight = actual_block.dw_conv.weight.data
    dw_weight_q = torch.round(dw_weight * scale).clamp(-127, 127).to(torch.int8)

    data["depthwise2_weight"] = dw_weight_q  # dw_conv weights
    data["depthwise2_scale"] = scale

    # Quantize pointwise projection (pw_project): mid_channels -> out_channels
    pw_project_weight = actual_block.pw_project.weight.data
    pw_project_weight_q = (
        torch.round(pw_project_weight * scale).clamp(-127, 127).to(torch.int8)
    )
    # pw_project has no bias in PyTorch (uses BatchNorm)
    out_channels = pw_project_weight.shape[0]

    data["pointwise_out_weight"] = pw_project_weight_q  # pw_project weights
    data["pointwise_out_scale"] = scale

    # No unused fields - clean implementation

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
        "asq_bits": model.asq.bits if model.use_asq else 4,
        "lambda_param": model.asq.lambda_param.item() if model.use_asq else 2.0,
    }

    # Extract and quantize layers from each stage
    layers = []

    # Extract initial conv layer (separate from stages)
    initial_conv = model.conv_initial  # Conv2d
    initial_conv_data = quantize_conv_layer(initial_conv)
    initial_conv_data["layer_type"] = 0  # Standard conv
    layers.append(initial_conv_data)

    # Extract LinearDepthwiseBlocks from stage1
    stage1_list = list(model.stage1.children())

    # Extract LinearDepthwiseBlocks from stage1 (all modules are LinearDepthwiseBlocks)
    for module in stage1_list:
        if hasattr(module, "pw_expand"):  # LinearDepthwiseBlock
            layer_data = quantize_linear_depthwise_block(module)
            layers.append(layer_data)

    # Extract blocks from stage2, stage3, stage4
    for stage in [model.stage2, model.stage3, model.stage4]:
        for module in stage.children():  # Use .children() not .modules()
            if hasattr(module, "pw_expand"):  # LinearDepthwiseBlock
                layer_data = quantize_linear_depthwise_block(module)
                layers.append(layer_data)
            elif hasattr(module, "lb"):  # DenseLinearDepthwiseBlock
                layer_data = quantize_linear_depthwise_block(
                    module.lb
                )  # Get the inner LinearDepthwiseBlock
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
    if hasattr(weight, "cpu"):
        weight_bytes = weight.cpu().numpy().astype("<i2").tobytes()
    else:
        weight_bytes = weight.astype("<i2").tobytes()
    f.write(weight_bytes)

    # Write bias dimensions and data
    f.write(struct.pack("<I", bias.shape[0]))  # output_size (L1)
    if hasattr(bias, "cpu"):
        bias_bytes = bias.cpu().numpy().astype("<i4").tobytes()
    else:
        bias_bytes = bias.astype("<i4").tobytes()
    f.write(bias_bytes)


def write_layer_stack(f, classifier_data: Dict[str, Any]) -> None:
    """Write quantized layer stack in the format expected by C++ engine."""
    layers = classifier_data["layers"]

    # Extract layer data
    l1_layer = layers[0]  # L1: l1_size -> l2_size
    l2_layer = layers[1]  # L2: l2_size -> l3_size
    l3_layer = layers[2]  # L3: l3_size -> num_classes

    # Write scales (including factorization scale)
    f.write(struct.pack("<f", l1_layer["scale"]))  # l1_scale
    f.write(struct.pack("<f", l2_layer["scale"]))  # l2_scale
    f.write(struct.pack("<f", l3_layer["scale"]))  # output_scale
    f.write(struct.pack("<f", l1_layer["scale"]))  # l1_fact_scale (use same as l1)

    # Write L1 layer: l1_size -> (l2_size + 1)
    l1_weight = l1_layer["weight"]
    l1_bias = l1_layer["bias"]
    l2_size = l1_weight.shape[0]
    l1_size = l1_weight.shape[1]

    # Create extended L1 layer: l1_size -> (l2_size + 1)
    l1_extended_weight = torch.zeros(l2_size + 1, l1_size, dtype=torch.int8)
    l1_extended_weight[:l2_size, :] = l1_weight
    l1_extended_bias = torch.zeros(l2_size + 1, dtype=torch.int32)
    l1_extended_bias[:l2_size] = l1_bias

    f.write(struct.pack("<I", l2_size + 1))  # l1_out_size
    f.write(struct.pack("<I", l1_size))  # l1_in_size
    f.write(l1_extended_weight.cpu().numpy().astype("i1").tobytes())
    f.write(struct.pack("<I", l2_size + 1))  # l1_bias_count
    f.write(l1_extended_bias.cpu().numpy().astype("<i4").tobytes())

    # Write L1 factorization layer: l1_size -> l1_size (identity)
    f.write(struct.pack("<I", l1_size))  # l1_fact_out_size
    f.write(struct.pack("<I", l1_size))  # l1_fact_in_size
    # Identity matrix weights
    identity_weights = torch.eye(l1_size, dtype=torch.int8) * 127  # Quantized identity
    f.write(identity_weights.cpu().numpy().astype("i1").tobytes())
    f.write(struct.pack("<I", l1_size))  # l1_fact_bias_count
    zero_bias = torch.zeros(l1_size, dtype=torch.int32)
    f.write(zero_bias.cpu().numpy().astype("<i4").tobytes())

    # Write L2 layer: (l2_size * 2) -> l3_size
    l2_weight = l2_layer["weight"]
    l2_bias = l2_layer["bias"]
    l3_size = l2_weight.shape[0]

    # Create extended L2 layer: (l2_size * 2) -> l3_size
    l2_extended_weight = torch.zeros(l3_size, l2_size * 2, dtype=torch.int8)
    l2_extended_weight[:, :l2_size] = l2_weight
    l2_extended_bias = l2_bias

    f.write(struct.pack("<I", l3_size))  # l2_out_size
    f.write(struct.pack("<I", l2_size * 2))  # l2_in_size
    f.write(l2_extended_weight.cpu().numpy().astype("i1").tobytes())
    f.write(struct.pack("<I", l3_size))  # l2_bias_count
    f.write(l2_extended_bias.cpu().numpy().astype("<i4").tobytes())

    # Write output layer: l3_size -> num_classes (full multiclass)
    l3_weight = l3_layer["weight"]
    l3_bias = l3_layer["bias"]
    num_classes = l3_weight.shape[0]

    f.write(struct.pack("<I", num_classes))  # out_out_size (number of classes)
    f.write(struct.pack("<I", l3_size))  # out_in_size
    f.write(l3_weight.cpu().numpy().astype("i1").tobytes())
    f.write(struct.pack("<I", num_classes))  # out_bias_count
    f.write(l3_bias.cpu().numpy().astype("<i4").tobytes())


def write_classifier(f, classifier_data: Dict[str, Any]) -> None:
    """Write quantized classifier weights and biases."""
    # Use layer stack format for compatibility with C++ engine
    write_layer_stack(f, classifier_data)


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

        # Write classifier
        write_classifier(f, quantized_data["classifier"])

    print(f"Successfully serialized model to {output_path}")


def load_model_from_checkpoint(checkpoint_path: Path) -> NNUE:
    """Load NNUE model from PyTorch checkpoint or state dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle both full checkpoints and state dicts
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        # Try to extract hyperparameters from checkpoint
        # If they're not available, we'll infer them from the state dict
        saved_num_classes = checkpoint.get("num_classes")
        saved_feature_set = checkpoint.get("feature_set")
        saved_l1_size = checkpoint.get("l1_size")
        saved_l2_size = checkpoint.get("l2_size")
        saved_l3_size = checkpoint.get("l3_size")

        # If we have all saved parameters, use them
        if all(
            param is not None
            for param in [
                saved_num_classes,
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
            num_classes = saved_num_classes
        else:
            # Infer architecture from state dict
            feature_set, l1_size, l2_size, l3_size, num_classes = (
                infer_architecture_from_state_dict(state_dict)
            )
    else:
        state_dict = checkpoint
        # Infer model architecture from state dict shapes
        feature_set, l1_size, l2_size, l3_size, num_classes = (
            infer_architecture_from_state_dict(state_dict)
        )

    # Create model with inferred parameters
    model = NNUE(
        feature_set=feature_set,
        l1_size=l1_size,
        l2_size=l2_size,
        l3_size=l3_size,
        num_classes=num_classes,
    )

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

    raise ValueError(
        f"Could not determine model type from checkpoint: {checkpoint_path}"
    )


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
    """Infer EtinyNet variant from state dict based on initial conv channels."""
    # Look at initial conv layer output channels
    for key in state_dict.keys():
        if "stage1.0.weight" in key:  # Initial conv layer
            conv_weight = state_dict[key]
            out_channels = conv_weight.shape[0]
            if out_channels == 32:
                return "1.0"
            elif out_channels == 28:
                return "0.98M"
            elif out_channels == 24:
                return "0.75"
            elif out_channels == 8:
                return "micro"

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

    # Check for required NNUE model weights
    if "input.weight" not in state_dict:
        raise ValueError(
            "Cannot find NNUE model weights in state dict. Available keys: "
            + str(list(state_dict.keys())[:10])
        )

    # Get input layer weight shape to determine feature set and L1
    input_weight_shape = state_dict["input.weight"].shape
    num_features = input_weight_shape[0]  # [num_features, L1]
    l1_size = input_weight_shape[1]

    # Infer grid size and features per square from conv layer
    conv_weight_shape = state_dict["conv.weight"].shape
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

    # Infer architecture from SimpleClassifier layers
    # Find the first and last layers of the classifier
    classifier_layers = [
        key for key in state_dict.keys() if key.startswith("classifier.classifier")
    ]

    # Get L2 and L3 from classifier layers
    # First linear layer: classifier.classifier.0.weight has shape [L2, L1]
    first_layer_key = "classifier.classifier.0.weight"
    if first_layer_key in state_dict:
        l2_size = state_dict[first_layer_key].shape[0]
    else:
        l2_size = 16  # Default fallback

    # Second linear layer: classifier.classifier.2.weight has shape [L3, L2]
    second_layer_key = "classifier.classifier.2.weight"
    if second_layer_key in state_dict:
        l3_size = state_dict[second_layer_key].shape[0]
    else:
        l3_size = 32  # Default fallback

    # Final layer: classifier.classifier.4.weight has shape [num_classes, L3]
    final_layer_key = "classifier.classifier.4.weight"
    if final_layer_key in state_dict:
        num_classes = state_dict[final_layer_key].shape[0]
    else:
        num_classes = 10  # Default fallback for CIFAR-10

    return feature_set, l1_size, l2_size, l3_size, num_classes


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
        print(f"  Number of classes: {model.num_classes}")

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
