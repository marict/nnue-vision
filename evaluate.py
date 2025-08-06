#!/usr/bin/env python3
"""Model Evaluation Module

Provides evaluation functions for NNUE and EtinyNet models, including both
PyTorch and compiled C++ engine evaluation.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from serialize import serialize_model


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics for model outputs."""
    # Convert to numpy for sklearn metrics
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Ensure we have the right shapes
    if outputs_np.ndim == 1:
        outputs_np = outputs_np.reshape(-1, 1)
    if targets_np.ndim == 1:
        targets_np = targets_np.reshape(-1)

    # For binary classification or regression, use the first output
    if outputs_np.shape[1] == 1:
        predictions = (outputs_np[:, 0] > 0.5).astype(int)
        targets_binary = (targets_np > 0.5).astype(int)
    else:
        # Multi-class classification
        predictions = outputs_np.argmax(axis=1)
        targets_binary = targets_np.astype(int)

    # Compute metrics
    accuracy = accuracy_score(targets_binary, predictions)
    f1 = f1_score(targets_binary, predictions, average="weighted", zero_division=0)
    precision = precision_score(
        targets_binary, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        targets_binary, predictions, average="weighted", zero_division=0
    )

    return {
        "acc": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn,
    device: Optional[torch.device] = None,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model using PyTorch for standard metrics."""
    total_loss = 0
    all_outputs = []
    all_targets = []

    for batch in loader:
        # Move batch to device
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        batch = (images, labels)

        loss = loss_fn(model, batch)
        total_loss += loss.item()
        outputs = model(batch[0])
        all_outputs.append(outputs.cpu())
        all_targets.append(batch[1].cpu())

    outputs = torch.cat(all_outputs)
    targets = torch.cat(all_targets)
    metrics = compute_metrics(outputs, targets)
    return total_loss / len(loader), metrics


def evaluate_compiled_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    model_type: str,
) -> Dict[str, float]:
    """Evaluate model using compiled C++ engine for real-world performance metrics."""
    # Check if C++ engine is available
    if model_type == "nnue":
        cpp_executable = Path("engine/build/regression_test")
        if not cpp_executable.exists():
            raise RuntimeError(
                f"C++ NNUE engine not found: {cpp_executable}. "
                f"Run 'cd engine && mkdir -p build && cd build && cmake .. && make' to build it."
            )
    elif model_type == "etinynet":
        cpp_executable = Path("engine/build/etinynet_inference")
        if not cpp_executable.exists():
            raise RuntimeError(
                f"C++ EtinyNet engine not found: {cpp_executable}. "
                f"Run 'cd engine && mkdir -p build && cd build && cmake .. && make' to build it."
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Serialize model to temporary file
    with tempfile.NamedTemporaryFile(suffix=f".{model_type}", delete=False) as f:
        model_path = Path(f.name)

    try:
        serialize_model(model, model_path)

        # Evaluate a subset of the dataset for speed
        all_outputs = []
        all_targets = []
        sample_count = 0
        max_samples = 100  # Limit for speed

        for batch in loader:
            if sample_count >= max_samples:
                break

            images, labels = batch
            batch_size = images.shape[0]
            processed_samples = min(batch_size, max_samples - sample_count)

            # For NNUE, we need to extract features
            if model_type == "nnue":
                # Use a simple feature extraction (first few pixels as features)
                for i in range(processed_samples):
                    img = images[i].cpu().numpy()
                    # Extract simple features (first 50 pixels as feature indices)
                    feature_indices = list(range(min(50, img.size)))

                    # Run C++ inference
                    cpp_args = [str(cpp_executable), str(model_path)] + [
                        str(f) for f in feature_indices
                    ]
                    result = subprocess.run(
                        cpp_args, capture_output=True, text=True, timeout=10
                    )

                    if result.returncode == 0:
                        # Parse C++ output
                        for line in result.stdout.split("\n"):
                            if line.startswith("RESULT_INCREMENTAL_0:"):
                                try:
                                    cpp_output = float(line.split(": ")[1])
                                    all_outputs.append(torch.tensor([cpp_output]))
                                    break
                                except (ValueError, IndexError) as e:
                                    raise RuntimeError(
                                        f"Failed to parse C++ output: {e}. Output: {line}"
                                    )
                    else:
                        raise RuntimeError(
                            f"C++ NNUE engine failed with return code {result.returncode}. Stderr: {result.stderr}"
                        )

            elif model_type == "etinynet":
                # For EtinyNet, save image to binary file and run inference
                for i in range(processed_samples):
                    img = images[i].cpu().numpy()

                    # Save image to temporary binary file
                    with tempfile.NamedTemporaryFile(
                        suffix=".bin", delete=False
                    ) as img_f:
                        img_path = Path(img_f.name)
                        img.tofile(img_f.name)

                    try:
                        # Run C++ inference
                        cpp_args = [
                            str(cpp_executable),
                            str(model_path),
                            str(img_path),
                            str(img.shape[1]),
                            str(img.shape[2]),
                        ]
                        result = subprocess.run(
                            cpp_args, capture_output=True, text=True, timeout=10
                        )

                        if result.returncode == 0:
                            # Parse C++ output (first line should be logits)
                            lines = result.stdout.strip().split("\n")
                            if lines:
                                try:
                                    cpp_output = float(lines[0])
                                    all_outputs.append(torch.tensor([cpp_output]))
                                except (ValueError, IndexError) as e:
                                    raise RuntimeError(
                                        f"Failed to parse C++ EtinyNet output: {e}. Output: {lines[0] if lines else 'empty'}"
                                    )
                            else:
                                raise RuntimeError(
                                    "C++ EtinyNet engine returned empty output"
                                )
                        else:
                            raise RuntimeError(
                                f"C++ EtinyNet engine failed with return code {result.returncode}. Stderr: {result.stderr}"
                            )
                    finally:
                        if img_path.exists():
                            img_path.unlink()

            # Add targets for the samples we processed
            actual_labels = labels[:batch_size]
            for i in range(processed_samples):
                target = actual_labels[i]
                # Ensure target is a tensor, not a scalar
                if target.dim() == 0:
                    target = target.unsqueeze(0)
                all_targets.append(target)
            sample_count += processed_samples

        if all_outputs and len(all_outputs) > 0:
            try:
                # Ensure all tensors have the same shape before concatenating
                if len(all_outputs) == 1:
                    outputs = all_outputs[0]
                else:
                    outputs = torch.cat(all_outputs)

                if len(all_targets) == 1:
                    targets = all_targets[0]
                else:
                    targets = torch.cat(all_targets)

                metrics = compute_metrics(outputs, targets)

                return metrics
            except Exception as e:
                print(f"⚠️ Error computing metrics: {e}")
                print(f"  Outputs shape: {[o.shape for o in all_outputs]}")
                print(f"  Targets shape: {[t.shape for t in all_targets]}")
                raise
        else:
            raise RuntimeError("No outputs generated during compiled model evaluation")

    finally:
        if model_path.exists():
            model_path.unlink()


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn,
    model_type: str,
    device: Optional[torch.device] = None,
    include_compiled: bool = True,
) -> Dict[str, any]:
    """Comprehensive model evaluation including both PyTorch and compiled metrics."""
    results = {}

    # Standard PyTorch evaluation
    try:
        loss, metrics = evaluate_model(model, loader, loss_fn, device)
        results["pytorch"] = {"loss": loss, "metrics": metrics}
    except:
        print(f"❌ PyTorch evaluation failed!")
        raise

    # Compiled evaluation (if requested and available)
    if include_compiled:
        try:
            compiled_metrics = evaluate_compiled_model(model, loader, model_type)
            results["compiled"] = {"metrics": compiled_metrics}
        except:
            print(f"❌ Compiled evaluation failed!")
            raise

    return results
