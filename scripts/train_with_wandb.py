#!/usr/bin/env python3
"""
Example script for training the Visual Wake Words model with comprehensive wandb logging.

This script demonstrates how to use the unified train.py with various configuration options
for wandb experiment tracking.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_training_experiment(
    experiment_name: str,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 20,
    image_size: int = 96,
    note: str = None,
    wandb_api_key: str = None,
    gpus: str = None,
):
    """
    Run a training experiment with specified parameters.

    Args:
        experiment_name: Name for the wandb project
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        image_size: Input image size (square)
        note: Optional note to add to run name
        wandb_api_key: Wandb API key (if not set as env var)
        gpus: GPU specification (e.g., "0,1" or "auto")
    """

    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "nnue",
        "--project_name",
        experiment_name,
        "--learning_rate",
        str(learning_rate),
        "--batch_size",
        str(batch_size),
        "--max_epochs",
        str(max_epochs),
        "--image_size",
        str(image_size),
        "--patience",
        "10",
        "--save_top_k",
        "3",
    ]

    if note:
        cmd.extend(["--note", note])

    if wandb_api_key:
        cmd.extend(["--wandb_api_key", wandb_api_key])

    if gpus:
        cmd.extend(["--gpus", gpus])

    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)

    # Run the command
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
        print(f"✅ Experiment '{experiment_name}' completed successfully!")
    except subprocess.CalledProcessError as e:
        print(
            f"❌ Experiment '{experiment_name}' failed with return code {e.returncode}"
        )
        return False

    return True


def main():
    """Run a series of training experiments."""

    # Check if wandb API key is available
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("⚠️  WANDB_API_KEY not found in environment variables.")
        print("Set it with: export WANDB_API_KEY=your_api_key_here")
        print("Or pass it directly to the experiment function.")
        return

    print("🚀 Starting Visual Wake Words training experiments with wandb logging")
    print("=" * 80)

    # Experiment 1: Baseline with default parameters
    success = run_training_experiment(
        experiment_name="visual_wake_words_baseline",
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=20,
        note="baseline_experiment",
    )

    if not success:
        print("❌ Baseline experiment failed, stopping.")
        return

    # Experiment 2: Higher learning rate
    run_training_experiment(
        experiment_name="visual_wake_words_lr_experiments",
        learning_rate=5e-3,
        batch_size=32,
        max_epochs=20,
        note="higher_lr_5e-3",
    )

    # Experiment 3: Different batch size
    run_training_experiment(
        experiment_name="visual_wake_words_batch_experiments",
        learning_rate=1e-3,
        batch_size=64,
        max_epochs=20,
        note="larger_batch_64",
    )

    # Experiment 4: Smaller image size for faster training
    run_training_experiment(
        experiment_name="visual_wake_words_size_experiments",
        learning_rate=1e-3,
        batch_size=32,
        max_epochs=20,
        image_size=64,
        note="smaller_image_64x64",
    )

    print("=" * 80)
    print("🎉 All experiments completed!")
    print("Check your wandb dashboard to view the results.")


if __name__ == "__main__":
    main()
