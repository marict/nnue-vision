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
    f.write(struct.pack("<I", weight.shape[0]))  # out_channels (12)
    f.write(struct.pack("<I", weight.shape[1]))  # in_channels (3)
    f.write(struct.pack("<I", weight.shape[2]))  # kernel_height (3)
    f.write(struct.pack("<I", weight.shape[3]))  # kernel_width (3)

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
    f.write(struct.pack("<I", weight.shape[1]))  # output_size

    # Write weights (int16, little endian)
    weight_bytes = weight.cpu().numpy().astype("<i2").tobytes()
    f.write(weight_bytes)

    # Write bias dimensions and data
    f.write(struct.pack("<I", bias.shape[0]))  # output_size
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

    f.write(struct.pack("<I", l1_weight.shape[0]))  # output_size
    f.write(struct.pack("<I", l1_weight.shape[1]))  # input_size
    f.write(l1_weight.cpu().numpy().astype("i1").tobytes())

    f.write(struct.pack("<I", l1_bias.shape[0]))
    f.write(l1_bias.cpu().numpy().astype("<i4").tobytes())

    # L2 layer
    l2_weight = ls_data["l2_weight"]  # int8
    l2_bias = ls_data["l2_bias"]  # int32

    f.write(struct.pack("<I", l2_weight.shape[0]))
    f.write(struct.pack("<I", l2_weight.shape[1]))
    f.write(l2_weight.cpu().numpy().astype("i1").tobytes())

    f.write(struct.pack("<I", l2_bias.shape[0]))
    f.write(l2_bias.cpu().numpy().astype("<i4").tobytes())

    # Output layer
    output_weight = ls_data["output_weight"]  # int8
    output_bias = ls_data["output_bias"]  # int32

    f.write(struct.pack("<I", output_weight.shape[0]))
    f.write(struct.pack("<I", output_weight.shape[1]))
    f.write(output_weight.cpu().numpy().astype("i1").tobytes())

    f.write(struct.pack("<I", output_bias.shape[0]))
    f.write(output_bias.cpu().numpy().astype("<i4").tobytes())


def serialize_model(model: NNUE, output_path: Path) -> None:
    """
    Serialize an NNUE model to .nnue binary format.

    Args:
        model: Trained NNUE model
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
        num_ls_buckets = checkpoint.get("num_ls_buckets", 8)
    else:
        state_dict = checkpoint
        # Infer model architecture from state dict shapes
        _, num_ls_buckets = infer_architecture_from_state_dict(state_dict)

    # Create model with inferred parameters
    # Note: feature_set is fixed in NNUE model (8x8x12 for visual wake words)
    model = NNUE(num_ls_buckets=num_ls_buckets)
    model.load_state_dict(state_dict)

    return model


def infer_architecture_from_state_dict(state_dict) -> tuple[GridFeatureSet, int]:
    """Infer model architecture from state dict tensor shapes."""
    # Get input layer weight shape to determine feature set
    input_weight_shape = state_dict["input.weight"].shape
    num_features = input_weight_shape[0]  # [num_features, L1]

    # Infer grid size and features per square
    # This is a simplified heuristic - in practice, this info should be saved
    common_configs = [
        (4, 6),  # 4x4 grid, 6 features per square = 96 total
        (8, 12),  # 8x8 grid, 12 features per square = 768 total
        (16, 8),  # 16x16 grid, 8 features per square = 2048 total
    ]

    for grid_size, features_per_square in common_configs:
        if grid_size * grid_size * features_per_square == num_features:
            feature_set = GridFeatureSet(grid_size, features_per_square)
            break
    else:
        # Fallback: assume square grid with 12 features per square
        grid_size = int((num_features / 12) ** 0.5)
        feature_set = GridFeatureSet(grid_size, 12)

    # Infer number of layer stack buckets from output layer shape
    output_weight_shape = state_dict["layer_stacks.output.weight"].shape
    num_ls_buckets = output_weight_shape[0]  # [num_buckets, L3]

    return feature_set, num_ls_buckets


def main():
    """Main serialization entry point."""
    parser = argparse.ArgumentParser(description="Serialize NNUE model to .nnue format")
    parser.add_argument("input", type=Path, help="Input model file (.pt or .ckpt)")
    parser.add_argument("output", type=Path, help="Output .nnue file path")
    parser.add_argument(
        "--features", type=str, default="Grid8x8_12", help="Feature set specification"
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Load model
    print(f"Loading model from {args.input}")
    model = load_model_from_checkpoint(args.input)

    # Serialize to .nnue format
    print(f"Serializing to {args.output}")
    serialize_model(model, args.output)

    print("Serialization complete!")


if __name__ == "__main__":
    main()
