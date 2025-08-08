#!/usr/bin/env python3
"""Model Evaluation Module

Provides evaluation functions for NNUE and EtinyNet models, including both
PyTorch and compiled C++ engine evaluation.
"""

import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from serialize import serialize_etinynet_model, serialize_model


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
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels.long())
        total_loss += loss.item()
        all_outputs.append(outputs.cpu())
        all_targets.append(labels.cpu())

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
        # Allow override via environment for sanitizer smoke tests
        override = os.environ.get("NNUE_ENGINE_EXEC")
        cpp_executable = (
            Path(override) if override else Path("engine/build/nnue_inference")
        )
        if not cpp_executable.exists():
            raise RuntimeError(
                f"C++ NNUE engine not found: {cpp_executable}. "
                f"Run 'cd engine && mkdir -p build && cd build && cmake .. && make' to build it."
            )
    elif model_type == "etinynet":
        override = os.environ.get("ETINY_ENGINE_EXEC")
        cpp_executable = (
            Path(override) if override else Path("engine/build/etinynet_inference")
        )
        if not cpp_executable.exists():
            raise RuntimeError(
                f"C++ EtinyNet engine not found: {cpp_executable}. "
                f"Run 'cd engine && mkdir -p build && cd build && cmake .. && make' to build it."
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Serialize model to temporary file with unique name
    model_path = Path(tempfile.mktemp(suffix=f".{model_type}"))

    try:
        # Use appropriate serialization function based on model type
        if model_type == "nnue":
            serialize_model(model, model_path)
        elif model_type == "etinynet":
            serialize_etinynet_model(model, model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Evaluate a subset of the dataset for speed
        all_outputs = []
        all_targets = []
        densities: list[float] = []
        # Evaluate the entire loader (to match PyTorch eval set exactly)

        # Timing measurements
        total_inference_time = 0.0
        total_samples_processed = 0

        for batch in loader:
            # No early break; consume entire loader

            images, labels = batch
            batch_size = images.shape[0]
            processed_samples = batch_size

            # For NNUE, use C++ engine for inference
            if model_type == "nnue":
                # Process each sample individually with C++ engine
                for i in range(processed_samples):
                    img = images[i].cpu().numpy()

                    # Save image to temporary binary file with unique name
                    img_path = Path(tempfile.mktemp(suffix=".bin"))
                    img.tofile(str(img_path))

                    try:
                        # Run C++ inference with timing
                        cpp_args = [
                            str(cpp_executable),
                            str(model_path),
                            str(img_path),
                            str(img.shape[1]),
                            str(img.shape[2]),
                        ]

                        start_time = time.time()
                        result = subprocess.run(
                            cpp_args, capture_output=True, text=True, timeout=10
                        )
                        inference_time = time.time() - start_time

                        total_inference_time += inference_time
                        total_samples_processed += 1

                        if result.returncode == 0:
                            # Parse C++ output
                            # Expected new format: comma-separated values with last element = density, preceding = logits
                            lines = result.stdout.strip().split("\n")
                            if lines:
                                parts = lines[0].split(",")
                                if len(parts) >= 2:
                                    try:
                                        density = float(parts[-1])
                                        logit_vals = [float(x) for x in parts[:-1]]
                                        logits_tensor = torch.tensor(
                                            logit_vals, dtype=torch.float32
                                        )
                                        # Shape to [1, C]
                                        all_outputs.append(logits_tensor.unsqueeze(0))
                                        densities.append(density)
                                    except ValueError as e:
                                        raise RuntimeError(
                                            f"Failed to parse NNUE CSV logits: {e}. Output: {lines[0]}"
                                        )
                                else:
                                    raise RuntimeError(
                                        f"Malformed NNUE output: expected CSV logits and density. Raw: {lines[0]}"
                                    )
                            else:
                                raise RuntimeError(
                                    "C++ NNUE engine returned empty output"
                                )
                        else:
                            # Enhanced error reporting for debugging
                            error_details = f"C++ NNUE engine failed with return code {result.returncode}"
                            if result.stderr:
                                error_details += f"\nStderr: {result.stderr}"
                            if result.stdout:
                                error_details += f"\nStdout: {result.stdout}"

                            # Add command details for debugging
                            error_details += f"\nCommand: {' '.join(cpp_args)}"
                            error_details += f"\nModel path: {model_path}"
                            error_details += f"\nImage path: {img_path}"
                            error_details += (
                                f"\nModel path exists: {model_path.exists()}"
                            )
                            error_details += f"\nImage path exists: {img_path.exists()}"

                            # Check file sizes
                            if model_path.exists():
                                error_details += f"\nModel file size: {model_path.stat().st_size} bytes"
                            if img_path.exists():
                                error_details += f"\nImage file size: {img_path.stat().st_size} bytes"

                            # Persist artifacts for exact repro
                            failure_root = Path("logs/compiled_eval_failures")
                            timestamp_dir = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            failure_dir = failure_root / timestamp_dir
                            try:
                                failure_dir.mkdir(parents=True, exist_ok=True)
                                saved_model = failure_dir / model_path.name
                                saved_image = failure_dir / img_path.name
                                # Copy bytes atomically
                                saved_model.write_bytes(model_path.read_bytes())
                                saved_image.write_bytes(img_path.read_bytes())
                                # Write repro script
                                repro = failure_dir / "repro.sh"
                                repro.write_text(
                                    "#!/usr/bin/env bash\n"
                                    "set -euo pipefail\n"
                                    f'cd "{Path.cwd()}"\n'
                                    f"{' '.join([str(cpp_executable), str(saved_model), str(saved_image), str(img.shape[1]), str(img.shape[2])])} | cat\n"
                                )
                                os.chmod(repro, 0o755)
                                error_details += f"\nSaved repro to: {failure_dir}"
                                error_details += f"\n  Model: {saved_model}"
                                error_details += f"\n  Image: {saved_image}"
                                error_details += f"\n  Script: {repro}"
                            except Exception as persist_err:
                                error_details += f"\n⚠️ Failed to persist repro artifacts: {persist_err}"

                            raise RuntimeError(error_details)
                    finally:
                        if img_path.exists():
                            img_path.unlink()

                # Add targets for the samples we processed
                for i in range(processed_samples):
                    target = labels[i]
                    # Ensure target is a tensor, not a scalar
                    if target.dim() == 0:
                        target = target.unsqueeze(0)
                    all_targets.append(target)

                # No sample cap; process full loader

            elif model_type == "etinynet":
                # For EtinyNet, save image to binary file and run inference
                for i in range(processed_samples):
                    img = images[i].cpu().numpy()

                    # Save image to temporary binary file with unique name
                    img_path = Path(tempfile.mktemp(suffix=".bin"))
                    img.tofile(str(img_path))

                    try:
                        # Run C++ inference with timing
                        cpp_args = [
                            str(cpp_executable),
                            str(model_path),
                            str(img_path),
                            str(img.shape[1]),
                            str(img.shape[2]),
                        ]

                        start_time = time.time()
                        result = subprocess.run(
                            cpp_args, capture_output=True, text=True, timeout=10
                        )
                        inference_time = time.time() - start_time

                        total_inference_time += inference_time
                        total_samples_processed += 1

                        if result.returncode == 0:
                            # Parse C++ output: expect lines like "RESULT_i: <value>"
                            lines = result.stdout.strip().split("\n")
                            vals = []
                            for line in lines:
                                if line.startswith("RESULT_") and ":" in line:
                                    try:
                                        _, rhs = line.split(":", 1)
                                        vals.append(float(rhs.strip()))
                                    except ValueError as e:
                                        raise RuntimeError(
                                            f"Failed to parse EtinyNet RESULT line '{line}': {e}"
                                        )
                            if not vals:
                                raise RuntimeError(
                                    f"C++ EtinyNet engine produced no RESULT_ lines. Raw stdout: {result.stdout[:200]}"
                                )
                            all_outputs.append(
                                torch.tensor(vals, dtype=torch.float32).unsqueeze(0)
                            )
                        else:
                            raise RuntimeError(
                                f"C++ EtinyNet engine failed with return code {result.returncode}. Stderr: {result.stderr}"
                            )
                    finally:
                        if img_path.exists():
                            img_path.unlink()

                    # Add targets for the samples we processed
                    for i in range(processed_samples):
                        target = labels[i]
                        if target.dim() == 0:
                            target = target.unsqueeze(0)
                        all_targets.append(target)

            # Add targets for the samples we processed (do this once per batch)
            # Note: This was causing duplicate target collection for NNUE models
            # Targets are now collected within each model type block

        if all_outputs and len(all_outputs) > 0:
            try:
                # Ensure consistent [N, C] shape before concatenating
                outputs = (
                    torch.cat(all_outputs, dim=0)
                    if len(all_outputs) > 1
                    else all_outputs[0]
                )

                targets = (
                    torch.cat(all_targets, dim=0)
                    if len(all_targets) > 1
                    else all_targets[0]
                )

                # Guard: if multiclass labels but outputs are [N,1], warn/raise
                inferred_num_classes = (
                    int(targets.max().item()) + 1 if targets.numel() > 0 else 1
                )
                if (
                    inferred_num_classes > 2
                    and outputs.ndim == 2
                    and outputs.shape[1] == 1
                ):
                    raise RuntimeError(
                        f"Compiled NNUE produced shape {tuple(outputs.shape)} for {inferred_num_classes}-class labels."
                        " This would compute misleading binary metrics. Ensure engine emits C logits."
                    )

                metrics = compute_metrics(outputs, targets)

                # Calculate speed metrics
                if total_samples_processed > 0:
                    avg_ms_per_sample = (
                        total_inference_time / total_samples_processed
                    ) * 1000
                    metrics["ms_per_sample"] = avg_ms_per_sample
                else:
                    metrics["ms_per_sample"] = 0.0

                # Calculate density metric (for NNUE models)
                if model_type == "nnue" and densities:
                    metrics["latent_density"] = sum(densities) / len(densities)
                else:
                    metrics["latent_density"] = 0.0

                return metrics
            except Exception as e:
                print(f"⚠️ Error computing metrics: {e}")
                print(f"  Outputs shape: {[o.shape for o in all_outputs]}")
                print(f"  Targets shape: {[t.shape for t in all_targets]}")
                raise
        else:
            raise RuntimeError("No outputs generated during compiled model evaluation")

    finally:
        # Only remove model file if we didn't already persist it
        if model_path.exists():
            try:
                model_path.unlink()
            except Exception:
                pass


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn,
    model_type: str,
    device: Optional[torch.device] = None,
    include_compiled: bool = True,
) -> Dict[str, Any]:
    """Comprehensive model evaluation including both PyTorch and compiled metrics."""
    results = {}

    # Standard PyTorch evaluation
    try:
        loss, metrics = evaluate_model(model, loader, loss_fn, device)
        results["pytorch"] = {"loss": loss, "metrics": metrics}
    except Exception as e:
        print(f"❌ PyTorch evaluation failed: {e}")
        raise

    # Compiled evaluation (if requested and available)
    if include_compiled:
        try:
            compiled_metrics = evaluate_compiled_model(model, loader, model_type)
            results["compiled"] = {"metrics": compiled_metrics}
        except Exception as e:
            print(f"❌ Compiled evaluation failed: {e}")
            raise

    return results


## REMB: removed unused compiled_parity_check helper; parity is covered by tests
