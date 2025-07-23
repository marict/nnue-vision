#!/usr/bin/env python3
"""
Serialize NNUE PyTorch models to .nnue format for C++ deployment.

This script converts trained NNUE models to the binary format expected by
C++ NNUE implementations, with proper quantization and optimization.
"""

import argparse
import struct
from pathlib import Path
from typing import Any, Dict

import torch

from model import NNUE, GridFeatureSet


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


def write_conv_layer(f, conv_data: Dict[str, Any]) -> None:
    """Write quantized convolutional layer weights and biases."""
    weight = conv_data[
        "weight"
    ]  # int8, shape (out_channels, in_channels, kernel_h, kernel_w)
    bias = conv_data["bias"]  # int32, shape (out_channels,)
    scale = conv_data["scale"]

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

    # Write scales
    f.write(struct.pack("<f", scales["l1"]))
    f.write(struct.pack("<f", scales["l2"]))
    f.write(struct.pack("<f", scales["output"]))

    # L1 layer
    l1_weight = ls_data["l1_weight"]  # int8
    l1_bias = ls_data["l1_bias"]  # int32

    f.write(struct.pack("<I", l1_weight.shape[0]))  # output_size (L2)
    f.write(struct.pack("<I", l1_weight.shape[1]))  # input_size (L1)
    f.write(l1_weight.cpu().numpy().astype("i1").tobytes())

    f.write(struct.pack("<I", l1_bias.shape[0]))
    f.write(l1_bias.cpu().numpy().astype("<i4").tobytes())

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
    model.load_state_dict(state_dict)

    return model


def infer_architecture_from_state_dict(
    state_dict,
) -> tuple[GridFeatureSet, int, int, int, int]:
    """Infer model architecture from state dict tensor shapes."""
    # Get input layer weight shape to determine feature set and L1
    input_weight_shape = state_dict["input.weight"].shape
    num_features = input_weight_shape[0]  # [num_features, L1]
    l1_size = input_weight_shape[1]

    # Infer number of layer stack buckets from output layer shape FIRST
    output_weight_shape = state_dict["layer_stacks.output.weight"].shape
    num_ls_buckets = output_weight_shape[0]  # [num_buckets, L3]

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

    # Infer L2 and L3 sizes from layer stack shapes
    # L2 size from l1_fact layer: l1_fact has shape [L2+1, L1]
    l1_fact_weight_shape = state_dict["layer_stacks.l1_fact.weight"].shape
    l2_size = l1_fact_weight_shape[0] - 1  # Remove the +1 from factorization

    # L3 size from output layer: output has shape [num_buckets, L3]
    l3_size = output_weight_shape[1]  # L3 size

    # Verify with L2 layer shape (should be [L3 * num_buckets, L2 * 2])
    l2_weight_shape = state_dict["layer_stacks.l2.weight"].shape
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


def main():
    """Main serialization entry point."""
    parser = argparse.ArgumentParser(description="Serialize NNUE model to .nnue format")
    parser.add_argument("input", type=Path, help="Input model file (.pt or .ckpt)")
    parser.add_argument("output", type=Path, help="Output .nnue file path")
    parser.add_argument(
        "--features",
        type=str,
        help="Feature set specification (will be auto-detected if not provided)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Load model (architecture will be auto-detected)
    print(f"Loading model from {args.input}")
    model = load_model_from_checkpoint(args.input)

    print(f"Detected architecture:")
    print(
        f"  Feature set: {model.feature_set.name} ({model.feature_set.num_features} features)"
    )
    print(f"  Layer sizes: {model.l1_size} -> {model.l2_size} -> {model.l3_size} -> 1")
    print(f"  Layer stack buckets: {model.num_ls_buckets}")

    # Serialize to .nnue format
    print(f"Serializing to {args.output}")
    serialize_model(model, args.output)

    print("Serialization complete!")


if __name__ == "__main__":
    main()
