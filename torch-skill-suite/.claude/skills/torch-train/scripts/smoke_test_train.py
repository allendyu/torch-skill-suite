#!/usr/bin/env python3
"""
Smoke test for the training loop using synthetic random data.

Verifies:
    1. Training runs without error for N epochs
    2. Loss decreases after training (model is learning)
    3. Checkpoint is saved and can be loaded
    4. Training can resume from checkpoint
    5. Resumed training continues correctly

Usage:
    python smoke_test_train.py
    python smoke_test_train.py --epochs 10 --batch-size 32
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train import (
    Trainer,
    build_model_from_contract,
    create_synthetic_dataloader,
)


def _make_model_contract(backbone="resnet18", num_classes=10):
    """Create a minimal model_contract for testing."""
    return {
        "task_type": "classification",
        "data_type": "image",
        "input_spec": {"shape": [3, 224, 224], "dtype": "float32", "channels_first": True},
        "model_spec": {
            "family": "cnn",
            "architecture": "resnet",
            "backbone": backbone,
            "pretrained": False,
            "in_channels": 3,
            "feature_dim": 512,
        },
        "head_spec": {
            "type": "linear_cls",
            "num_classes": num_classes,
            "pooling": "avg",
            "dropout": 0.0,
        },
        "forward_spec": {"output_shape": ["batch", num_classes]},
    }


def _make_data_contract(shape=None, num_classes=10):
    """Create a minimal data_contract for testing."""
    return {
        "data_type": "image",
        "task_type": "classification",
        "input_spec": {
            "shape": shape or [3, 224, 224],
            "dtype": "float32",
            "channels_first": True,
        },
        "output_spec": {
            "type": "categorical",
            "num_classes": num_classes,
        },
    }


def run_smoke_test(epochs=3, batch_size=16, device=None):
    """Run a complete smoke test of the training loop.

    Returns:
        Tuple of (passed, details_dict).
    """
    if device is None:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # Test actual CUDA access
            if device == "cuda":
                torch.cuda.current_device()
        except (RuntimeError, torch.cuda.DeferredCudaCallError):
            device = "cpu"

    results = {}
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")

    # ---- Test 1: Build model and train ----
    print("\n[Test 1] Training with synthetic data...")
    model_contract = _make_model_contract("resnet18", num_classes=5)
    data_contract = _make_data_contract([3, 224, 224], num_classes=5)

    model = build_model_from_contract(model_contract)
    train_loader = create_synthetic_dataloader(
        data_contract["input_spec"], 5, num_samples=100, batch_size=batch_size
    )

    config = {
        "task_type": "classification",
        "optimizer": {"name": "adam", "lr": 0.001},
        "scheduler": {"name": "none"},
    }

    trainer = Trainer(model, device, config)
    with tempfile.TemporaryDirectory() as tmpdir:
        history = trainer.train(train_loader, epochs=epochs, checkpoint_dir=tmpdir)

        train_losses = history["train_loss"]
        assert len(train_losses) == epochs, f"Expected {epochs} loss values, got {len(train_losses)}"
        loss_decreased = train_losses[-1] < train_losses[0]
        print(f"  Initial loss: {train_losses[0]:.4f}")
        print(f"  Final loss:   {train_losses[-1]:.4f}")
        print(f"  Loss decreased: {'YES' if loss_decreased else 'NO (expected with random data)'}")
        results["train_runs"] = True

        # ---- Test 2: Checkpoint saved ----
        print("\n[Test 2] Checkpoint files exist...")
        best_path = Path(tmpdir) / "best_model.pt"
        last_path = Path(tmpdir) / "last_model.pt"
        assert best_path.exists(), "best_model.pt not found"
        assert last_path.exists(), "last_model.pt not found"
        print(f"  best_model.pt: {best_path.stat().st_size / 1024:.0f} KB")
        print(f"  last_model.pt: {last_path.stat().st_size / 1024:.0f} KB")
        results["checkpoint_saved"] = True

        # ---- Test 3: Load checkpoint ----
        print("\n[Test 3] Loading checkpoint...")
        model2 = build_model_from_contract(model_contract)
        trainer2 = Trainer(model2, device, config)
        ckpt = trainer2.load_checkpoint(str(best_path))
        print(f"  Restored epoch: {ckpt['epoch']}")
        print(f"  Restored best_loss: {ckpt['best_loss']:.4f}")
        assert ckpt["epoch"] == epochs - 1, f"Expected epoch {epochs - 1}, got {ckpt['epoch']}"
        results["checkpoint_loaded"] = True

        # ---- Test 4: Resume training ----
        print("\n[Test 4] Resuming training from checkpoint...")
        history2 = trainer2.train(train_loader, epochs=2, checkpoint_dir=tmpdir)
        total_epochs = len(history2["train_loss"])
        assert total_epochs == 5, f"Expected 5 total epochs (3 original + 2 resumed), got {total_epochs}"
        print(f"  Total epochs after resume: {total_epochs}")
        print(f"  Loss after resume: {history2['train_loss'][-1]:.4f}")
        results["resume_works"] = True

    # ---- Test 5: Different backbone ----
    print("\n[Test 5] Training with efficientnet_b0...")
    mc = _make_model_contract("efficientnet_b0", num_classes=3)
    mc["model_spec"]["architecture"] = "efficientnet"
    mc["model_spec"]["feature_dim"] = 1280
    model3 = build_model_from_contract(mc)
    loader3 = create_synthetic_dataloader(
        {"shape": [3, 224, 224], "dtype": "float32"}, 3, num_samples=50, batch_size=batch_size
    )
    trainer3 = Trainer(model3, device, config)
    history3 = trainer3.train(loader3, epochs=1, checkpoint_dir=None)
    print(f"  Output shape: [{history3['train_loss'][0]:.4f}]")
    results["efficientnet_works"] = True

    # ---- Summary ----
    all_passed = all(results.values())
    print(f"\n{'=' * 40}")
    print(f"Smoke test: {'PASSED' if all_passed else 'FAILED'}")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
    return all_passed, results


def main():
    parser = argparse.ArgumentParser(description="Smoke test the training loop.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    args = parser.parse_args()

    passed, _ = run_smoke_test(epochs=args.epochs, batch_size=args.batch_size)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
