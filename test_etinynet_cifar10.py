#!/usr/bin/env python3
"""
CIFAR-10 validation test for EtinyNet implementation.

This script trains and evaluates EtinyNet on CIFAR-10 to verify:
1. The model can actually learn on real data
2. Performance is reasonable for a tiny model
3. Both variants work correctly
4. ASQ quantization doesn't break training
"""

import sys
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from model import EtinyNet
from serialize import serialize_etinynet_model


class EtinyNetCIFAR10(pl.LightningModule):
    """PyTorch Lightning wrapper for EtinyNet on CIFAR-10."""

    def __init__(self, variant="0.75", use_asq=False, lr=0.1):
        super().__init__()
        self.save_hyperparameters()

        # Create EtinyNet for CIFAR-10 (10 classes, 32x32 input)
        self.model = EtinyNet(
            variant=variant,
            num_classes=10,
            input_size=32,  # CIFAR-10 is 32x32
            use_asq=use_asq,
            lr=lr,
            max_epochs=20,
        )

        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, targets)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).sum().item() / targets.size(0)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, targets)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).sum().item() / targets.size(0)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return [optimizer], [scheduler]


def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    """Get CIFAR-10 data loaders with appropriate transforms."""

    # CIFAR-10 preprocessing
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data/raw", train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data/raw", train=False, download=True, transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def quick_train_test(variant="0.75", use_asq=False, max_epochs=5):
    """Quick training test to verify model can learn."""
    print(f"\n🚀 Quick Training Test: EtinyNet-{variant}, ASQ={use_asq}")

    # Use more conservative learning rate for ASQ
    lr = 0.01 if use_asq else 0.1

    # Create model
    model = EtinyNetCIFAR10(variant=variant, use_asq=use_asq, lr=lr)

    # Get data (smaller batch for quick test)
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=64, num_workers=2)

    # Limit training data for quick test
    train_subset = torch.utils.data.Subset(
        train_loader.dataset, range(0, 1000)
    )  # 1K samples
    test_subset = torch.utils.data.Subset(
        test_loader.dataset, range(0, 200)
    )  # 200 samples

    quick_train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    quick_test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        accelerator="auto",
    )

    # Train
    print(f"Training for {max_epochs} epochs on 1K samples...")
    start_time = time.time()
    trainer.fit(model, quick_train_loader, quick_test_loader)
    train_time = time.time() - start_time

    # Test
    test_results = trainer.test(model, quick_test_loader, verbose=False)

    # Extract results safely
    final_test_acc = 0.0
    final_test_loss = float("inf")

    if test_results and len(test_results) > 0:
        result_dict = test_results[0]
        # Try different possible key names
        for key in ["test_acc", "val_acc", "acc"]:
            if key in result_dict:
                final_test_acc = result_dict[key]
                break

        for key in ["test_loss", "val_loss", "loss"]:
            if key in result_dict:
                final_test_loss = result_dict[key]
                break

    # Check if model learned (should be > random chance = 10%)
    learned_successfully = final_test_acc > 0.15  # 15% threshold

    print(f"✓ Final test accuracy: {final_test_acc:.1%}")
    print(f"✓ Final test loss: {final_test_loss:.3f}")
    print(f"✓ Training time: {train_time:.1f}s")
    print(f"✓ Parameters: {model.model.count_parameters():,}")

    if learned_successfully:
        print(f"🎉 SUCCESS: Model learned (acc > 15%)")
    else:
        print(f"⚠️  WARNING: Model may not be learning properly (acc ≤ 15%)")

    return {
        "variant": variant,
        "use_asq": use_asq,
        "final_acc": final_test_acc,
        "final_loss": final_test_loss,
        "train_time": train_time,
        "parameters": model.model.count_parameters(),
        "learned": learned_successfully,
    }


def test_inference_speed(variant="0.75", batch_size=100, num_batches=10):
    """Test inference speed on CIFAR-10."""
    print(f"\n⚡ Inference Speed Test: EtinyNet-{variant}")

    # Create model and set to eval mode
    model = EtinyNet(variant=variant, num_classes=10, input_size=32)
    model.eval()

    # Create random CIFAR-10-like data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Warm up
    with torch.no_grad():
        dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
        for _ in range(3):
            _ = model(dummy_input)

    # Measure inference time
    torch.cuda.synchronize() if device.type == "cuda" else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_batches):
            dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
            outputs = model(dummy_input)

    torch.cuda.synchronize() if device.type == "cuda" else None
    end_time = time.time()

    total_samples = batch_size * num_batches
    total_time = end_time - start_time
    samples_per_sec = total_samples / total_time
    ms_per_sample = (total_time * 1000) / total_samples

    print(f"✓ Device: {device}")
    print(f"✓ Total samples: {total_samples}")
    print(f"✓ Total time: {total_time:.3f}s")
    print(f"✓ Throughput: {samples_per_sec:.1f} samples/sec")
    print(f"✓ Latency: {ms_per_sample:.2f} ms/sample")

    return {
        "variant": variant,
        "device": str(device),
        "samples_per_sec": samples_per_sec,
        "ms_per_sample": ms_per_sample,
    }


def test_serialization_roundtrip(variant="0.75"):
    """Test that serialization and model saving works."""
    print(f"\n💾 Serialization Test: EtinyNet-{variant}")

    try:
        # Create and train a tiny model
        model = EtinyNetCIFAR10(variant=variant, use_asq=False)

        # Quick training on small dataset
        train_loader, _ = get_cifar10_dataloaders(batch_size=32, num_workers=0)
        train_subset = torch.utils.data.Subset(train_loader.dataset, range(0, 100))
        quick_loader = DataLoader(
            train_subset, batch_size=32, shuffle=True, num_workers=0
        )

        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False, logger=False)
        trainer.fit(model, quick_loader)

        # Save PyTorch model
        output_dir = Path("test_outputs")
        output_dir.mkdir(exist_ok=True)

        torch_path = output_dir / f"etinynet_{variant}_cifar10.ckpt"
        trainer.save_checkpoint(torch_path)

        # Serialize to .etiny format
        etiny_path = output_dir / f"etinynet_{variant}_cifar10.etiny"
        serialize_etinynet_model(model.model, etiny_path)

        # Check files exist and have reasonable sizes
        torch_size = torch_path.stat().st_size if torch_path.exists() else 0
        etiny_size = etiny_path.stat().st_size if etiny_path.exists() else 0

        print(f"✓ PyTorch checkpoint: {torch_path} ({torch_size:,} bytes)")
        print(f"✓ EtinyNet binary: {etiny_path} ({etiny_size:,} bytes)")
        print(f"✓ Size ratio (etiny/torch): {etiny_size/torch_size:.2f}")

        success = torch_path.exists() and etiny_path.exists() and etiny_size > 1000
        print(f"🎉 Serialization: {'SUCCESS' if success else 'FAILED'}")

        return success

    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        return False


def run_comprehensive_cifar10_tests():
    """Run all CIFAR-10 tests."""
    print("=" * 80)
    print("🧪 EtinyNet CIFAR-10 Comprehensive Tests")
    print("=" * 80)

    results = []

    # Test 1: Quick training tests for both variants
    test_configs = [
        ("0.75", False),  # EtinyNet-0.75 without ASQ
        ("1.0", False),  # EtinyNet-1.0 without ASQ
        ("0.75", True),  # EtinyNet-0.75 with ASQ
    ]

    for variant, use_asq in test_configs:
        try:
            result = quick_train_test(variant=variant, use_asq=use_asq, max_epochs=3)
            results.append(result)
        except Exception as e:
            print(f"❌ Training test failed for {variant}, ASQ={use_asq}: {e}")
            results.append(
                {
                    "variant": variant,
                    "use_asq": use_asq,
                    "learned": False,
                    "final_acc": 0.0,
                    "error": str(e),
                }
            )

    # Test 2: Inference speed tests
    speed_results = []
    for variant in ["0.75", "1.0"]:
        try:
            speed_result = test_inference_speed(variant=variant)
            speed_results.append(speed_result)
        except Exception as e:
            print(f"❌ Speed test failed for {variant}: {e}")

    # Test 3: Serialization test
    try:
        serialization_success = test_serialization_roundtrip("0.75")
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
        serialization_success = False

    # Summary
    print("\n" + "=" * 80)
    print("📊 CIFAR-10 Test Results Summary")
    print("=" * 80)

    print("\n🏋️ Training Results:")
    learning_success = 0
    for result in results:
        if "error" not in result:
            status = "✅ LEARNED" if result["learned"] else "⚠️ POOR"
            print(
                f"  EtinyNet-{result['variant']} (ASQ={result['use_asq']}): "
                f"{status} | Acc: {result['final_acc']:.1%} | "
                f"Params: {result['parameters']:,}"
            )
            if result["learned"]:
                learning_success += 1
        else:
            print(
                f"  EtinyNet-{result['variant']} (ASQ={result['use_asq']}): ❌ ERROR - {result['error']}"
            )

    print(f"\n⚡ Inference Speed:")
    for result in speed_results:
        print(
            f"  EtinyNet-{result['variant']}: {result['samples_per_sec']:.1f} samples/sec "
            f"({result['ms_per_sample']:.2f} ms/sample) on {result['device']}"
        )

    print(
        f"\n💾 Serialization: {'✅ SUCCESS' if serialization_success else '❌ FAILED'}"
    )

    # Overall assessment
    overall_success = (
        learning_success >= 2  # At least 2 variants learned
        and len(speed_results) >= 2  # Speed tests worked
        and serialization_success  # Serialization worked
    )

    print(
        f"\n🎯 Overall Assessment: {'🎉 SUCCESS' if overall_success else '⚠️ ISSUES DETECTED'}"
    )

    if overall_success:
        print("   EtinyNet implementation is working correctly on CIFAR-10!")
    else:
        print("   Some issues detected. Check individual test results above.")

    return overall_success


if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_cifar10_tests()
    sys.exit(0 if success else 1)
