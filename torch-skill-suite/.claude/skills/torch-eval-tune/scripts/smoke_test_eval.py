#!/usr/bin/env python3
"""
Smoke test for torch-eval-tune.

Verifies:
    1. Evaluate a trained checkpoint and compute metrics
    2. Generate tuning plan from training history
    3. End-to-end: train -> evaluate -> tune

Usage:
    python smoke_test_eval.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

TRAIN_SCRIPTS_DIR = SCRIPT_DIR / ".." / ".." / "torch-train" / "scripts"
sys.path.insert(0, str(TRAIN_SCRIPTS_DIR.resolve()))

from train import (
    Trainer,
    build_model_from_contract,
    create_synthetic_dataloader,
    _add_template_path,
)
_add_template_path()

from evaluate import evaluate
from tune import generate_tuning_plan


def _make_model_contract(backbone="resnet18", num_classes=5):
    return {
        "task_type": "classification",
        "data_type": "image",
        "input_spec": {"shape": [3, 224, 224], "dtype": "float32", "channels_first": True},
        "model_spec": {
            "family": "cnn", "architecture": "resnet", "backbone": backbone,
            "pretrained": False, "in_channels": 3, "feature_dim": 512,
        },
        "head_spec": {"type": "linear_cls", "num_classes": num_classes, "pooling": "avg", "dropout": 0.0},
        "forward_spec": {"output_shape": ["batch", num_classes]},
        "metadata": {"route": "image_classification", "priority": "P0"},
    }


def run_smoke_test():
    print("=" * 50)
    print("torch-eval-tune Smoke Test")
    print("=" * 50)

    device = "cpu"
    mc = _make_model_contract("resnet18", num_classes=5)
    input_spec = mc["input_spec"]
    num_classes = mc["head_spec"]["num_classes"]

    # ---- Train a model ----
    print("\n[Step 1] Training model for evaluation...")
    model = build_model_from_contract(mc)
    train_loader = create_synthetic_dataloader(
        input_spec, num_classes, num_samples=100, batch_size=8
    )
    config = {"task_type": "classification", "optimizer": {"name": "adam", "lr": 0.01}}
    trainer = Trainer(model, device, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "best_model.pt")
        history = trainer.train(train_loader, epochs=3, checkpoint_dir=tmpdir)
        print(f"  Final loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final acc:  {history['train_acc'][-1]:.1f}%")

        # ---- Test 1: Evaluate checkpoint ----
        print("\n[Test 1] Evaluating checkpoint...")
        report = evaluate(
            mc, ckpt_path, input_spec=input_spec, num_classes=num_classes,
            synthetic=True, batch_size=8, device=device,
        )
        assert "metrics" in report
        assert "accuracy" in report["metrics"]
        assert report["metrics"]["accuracy"] > 0
        assert "confusion_matrix" in report["metrics"]
        print(f"  Accuracy: {report['metrics']['accuracy']:.1f}%")
        print(f"  Macro F1: {report['metrics']['macro_f1']:.4f}")
        print(f"  Loss:     {report['loss']:.4f}")
        print("  [PASS] Evaluation produced valid metrics")

        # ---- Test 2: Generate tuning plan ----
        print("\n[Test 2] Generating tuning plan...")
        plan = generate_tuning_plan(history, report, target_accuracy=80.0)
        assert "analysis" in plan
        assert "suggestions" in plan
        assert len(plan["suggestions"]) >= 1
        print(f"  Suggestions: {len(plan['suggestions'])}")
        for s in plan["suggestions"]:
            print(f"    [P{s['priority']}] [{s['category']}] {s['action']}")
        print("  [PASS] Tuning plan generated")

        # ---- Test 3: Overfitting detection ----
        print("\n[Test 3] Overfitting detection...")
        overfit_history = {
            "train_loss": [2.0, 1.0, 0.3, 0.1, 0.05],
            "val_loss": [2.1, 1.5, 1.8, 2.0, 2.2],
            "train_acc": [30, 50, 70, 85, 95],
            "val_acc": [28, 45, 42, 40, 38],
        }
        plan2 = generate_tuning_plan(overfit_history)
        categories = [s["category"] for s in plan2["suggestions"]]
        assert "regularization" in categories, f"Expected regularization suggestion, got: {categories}"
        print("  [PASS] Overfitting correctly detected")

        # ---- Test 4: Plateau detection ----
        print("\n[Test 4] Plateau detection...")
        plateau_history = {
            "train_loss": [0.8, 0.6, 0.55, 0.548, 0.547, 0.546],
            "val_loss": [0.9, 0.75, 0.72, 0.72, 0.719, 0.718],
            "train_acc": [60, 65, 68, 68, 68, 68],
            "val_acc": [55, 58, 60, 60, 60, 60],
        }
        plan3 = generate_tuning_plan(plateau_history)
        categories3 = [s["category"] for s in plan3["suggestions"]]
        assert "learning_rate" in categories3, f"Expected learning_rate suggestion, got: {categories3}"
        print("  [PASS] Plateau correctly detected")

    print(f"\n{'=' * 50}")
    print("Smoke test: PASSED")
    print(f"{'=' * 50}")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
