#!/usr/bin/env python3
"""Unified Training Script

Train NNUE or EtinyNet models with a unified interface.

Usage Examples:
    # Train NNUE with default config
    python train.py nnue

    # Train EtinyNet with default config
    python train.py etinynet

    # Train with custom config
    python train.py nnue --config config/train_nnue_custom.py

    # Train EtinyNet with specific variant and dataset
    python train.py etinynet --variant 1.0 --dataset cifar100

    # Override config parameters
    python train.py nnue --batch_size 64 --max_epochs 100 --learning_rate 0.001
"""

import sys
from typing import Dict

from etinynet_adapter import EtinyNetAdapter
from nnue_adapter import NNUEAdapter
from training_framework import BaseTrainer, ModelAdapter


def get_available_adapters() -> Dict[str, ModelAdapter]:
    """Return a dictionary of available model adapters."""
    return {
        "nnue": NNUEAdapter(),
        "etinynet": EtinyNetAdapter(),
    }


def main():
    """Main entry point for unified training."""
    if len(sys.argv) < 2:
        print("Usage: python train.py <model_type> [options]")
        print("Available model types:")
        for model_type in get_available_adapters().keys():
            print(f"  {model_type}")
        sys.exit(1)

    model_type = sys.argv[1].lower()
    adapters = get_available_adapters()

    if model_type not in adapters:
        print(f"Error: Unknown model type '{model_type}'")
        print("Available model types:")
        for available_type in adapters.keys():
            print(f"  {available_type}")
        sys.exit(1)

    # Remove model type from sys.argv so argparse doesn't see it
    sys.argv.pop(1)

    # Create trainer with the appropriate adapter
    adapter = adapters[model_type]
    trainer = BaseTrainer(adapter)

    # Run training
    exit_code = trainer.run_training()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
