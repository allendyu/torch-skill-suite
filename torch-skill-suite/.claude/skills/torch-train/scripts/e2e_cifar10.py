#!/usr/bin/env python3
"""
End-to-end integration test: synthetic ImageFolder → train → verify.

Pipeline:
    1. Generate synthetic ImageFolder dataset on disk (random PNG images)
    2. Generate data_contract.yaml
    3. Generate model_contract.yaml via torch-model resolver
    4. Train with torch-train
    5. Verify checkpoint and metrics

No network required — all data is generated locally.

Usage:
    python e2e_cifar10.py
    python e2e_cifar10.py --epochs 5 --batch-size 32 --backbone resnet18
    python e2e_cifar10.py --output-dir ./e2e_output
"""

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add shared package to path
_SHARED_PYTHON = Path(__file__).resolve().parent.parent.parent.parent.parent / "shared" / "python"
if str(_SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SHARED_PYTHON))

from torch_skill_shared.yaml_utils import emit_yaml


# ---------------------------------------------------------------------------
# Path setup — add sibling skill scripts to sys.path
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_SCRIPTS_DIR = SCRIPT_DIR / ".." / ".." / "torch-model" / "scripts"
TRAIN_SCRIPTS_DIR = SCRIPT_DIR

for _p in [str(MODEL_SCRIPTS_DIR.resolve()), str(TRAIN_SCRIPTS_DIR.resolve())]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from resolve_model import resolve as resolve_model
from train import Trainer, build_model_from_contract, _add_template_path

_add_template_path()

SYNTHETIC_CLASSES = ["cat", "dog", "bird", "fish", "rabbit"]
NUM_CLASSES = len(SYNTHETIC_CLASSES)


# ---------------------------------------------------------------------------
# Step 1: Generate synthetic ImageFolder on disk
# ---------------------------------------------------------------------------

def generate_synthetic_imagefolder(data_dir, num_classes=5, samples_per_class=50,
                                   img_size=(224, 224), val_ratio=0.2, seed=42):
    """Generate random PNG images organized in ImageFolder structure.

    Args:
        data_dir: Root directory for organized data.
        num_classes: Number of class folders.
        samples_per_class: Images per class (train split).
        img_size: (H, W) of generated images.
        val_ratio: Fraction of samples to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dir, val_dir, num_classes).
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Check if already generated
    if os.path.isdir(train_dir) and any(
        os.path.isdir(os.path.join(train_dir, d)) for d in os.listdir(train_dir)
    ):
        print(f"ImageFolder structure already exists at {data_dir}, skipping generation.")
        return train_dir, val_dir, num_classes

    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating synthetic ImageFolder dataset ({num_classes} classes, "
          f"{samples_per_class} images/class)...")

    for class_idx, class_name in enumerate(SYNTHETIC_CLASSES[:num_classes]):
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        n_val = int(samples_per_class * val_ratio)
        n_train = samples_per_class - n_val

        # Generate train images
        for i in range(n_train):
            img = _make_random_image(img_size, class_idx)
            img.save(os.path.join(train_class_dir, f"img_{class_idx}_{i:04d}.png"))

        # Generate val images
        for i in range(n_val):
            img = _make_random_image(img_size, class_idx)
            img.save(os.path.join(val_class_dir, f"img_{class_idx}_{i:04d}.png"))

    train_count = sum(1 for _ in Path(train_dir).rglob("*.png"))
    val_count = sum(1 for _ in Path(val_dir).rglob("*.png"))
    print(f"  Train: {train_count} images")
    print(f"  Val:   {val_count} images")

    return train_dir, val_dir, num_classes


def _make_random_image(size, class_idx):
    """Generate a random RGB image with a class-specific bias.

    Each class has a different color channel bias so the model can learn
    to distinguish them (otherwise pure noise is unlearnable).
    """
    # Base random pixels
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8).astype(np.float32)

    # Class-specific color bias (makes classes distinguishable)
    biases = [
        [1.3, 0.7, 0.7],   # cat: red-ish
        [0.7, 0.7, 1.3],   # dog: blue-ish
        [0.7, 1.3, 0.7],   # bird: green-ish
        [1.3, 1.3, 0.7],   # fish: yellow-ish
        [1.3, 0.7, 1.3],   # rabbit: purple-ish
    ]
    bias = biases[class_idx % len(biases)]

    arr[:, :, 0] *= bias[0]
    arr[:, :, 1] *= bias[1]
    arr[:, :, 2] *= bias[2]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Step 2: Generate data_contract.yaml
# ---------------------------------------------------------------------------

def generate_data_contract(train_dir, val_dir, output_path):
    """Generate a data_contract.yaml for the synthetic ImageFolder dataset."""
    contract = {
        "data_type": "image",
        "task_type": "classification",
        "input_spec": {
            "shape": [3, 224, 224],
            "dtype": "float32",
            "channels_first": True,
        },
        "output_spec": {
            "type": "categorical",
            "num_classes": NUM_CLASSES,
            "label_map": {i: name for i, name in enumerate(SYNTHETIC_CLASSES)},
        },
        "splits": {
            "train": train_dir,
            "val": val_dir,
        },
        "preprocessing": [
            {"name": "resize", "params": {"size": [224, 224]}},
            {"name": "normalize", "params": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }},
        ],
        "data_format_option": "user_provided",
        "user_format_spec": {
            "format_type": "ImageFolder",
            "details": {
                "structure": "class folders inside split folders",
                "extensions": [".png"],
            },
        },
        "metadata": {
            "source": "synthetic",
            "description": "Synthetic ImageFolder dataset for E2E testing",
        },
    }

    yaml_text = emit_yaml(contract)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(f"# data_contract.yaml — synthetic E2E\n{yaml_text}\n")
    print(f"data_contract → {output_path}")
    return contract


# ---------------------------------------------------------------------------
# Step 3: Generate model_contract.yaml via torch-model resolver
# ---------------------------------------------------------------------------

def generate_model_contract(data_contract, project_spec, output_path):
    """Use torch-model's resolve_model to generate model_contract."""
    model_contract = resolve_model(data_contract, project_spec)
    yaml_text = emit_yaml(model_contract)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(f"# model_contract.yaml — generated by torch-model resolver\n{yaml_text}\n")
    backbone = model_contract["model_spec"]["backbone"]
    print(f"model_contract → {output_path}  (backbone: {backbone})")
    return model_contract


# ---------------------------------------------------------------------------
# Step 4: Train
# ---------------------------------------------------------------------------

def run_training(data_contract, model_contract, train_dir, val_dir, args):
    """Run torch-train with ImageFolder data."""
    input_spec = data_contract.get("input_spec", {})
    preprocessing = data_contract.get("preprocessing", [])
    num_classes = model_contract["head_spec"]["num_classes"]

    print(f"\nBuilding model: {model_contract['model_spec']['backbone']}")
    model = build_model_from_contract(model_contract)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    from train import create_imagefolder_dataloader

    print(f"\nLoading data from {train_dir}")
    train_loader = create_imagefolder_dataloader(
        train_dir, input_spec, preprocessing=preprocessing,
        batch_size=args.batch_size, is_train=True,
    )
    print(f"  Train batches: {len(train_loader)}")

    val_loader = None
    if val_dir and os.path.isdir(val_dir):
        val_loader = create_imagefolder_dataloader(
            val_dir, input_spec, preprocessing=preprocessing,
            batch_size=args.batch_size, is_train=False,
        )
        print(f"  Val batches:   {len(val_loader)}")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\nDevice: {device}")

    trainer_config = {
        "task_type": "classification",
        "optimizer": {"name": "adam", "lr": args.lr, "weight_decay": 1e-4},
        "scheduler": {"name": "cosine", "t_max": args.epochs},
    }

    trainer = Trainer(model, device, trainer_config)

    print(f"\nTraining {args.epochs} epoch(s)...")
    print("-" * 60)
    history = trainer.train(
        train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
    )
    print("-" * 60)

    return history


# ---------------------------------------------------------------------------
# Step 5: Verify
# ---------------------------------------------------------------------------

def verify_results(history, checkpoint_dir, min_epochs=2):
    """Verify training produced valid results."""
    errors = []

    train_losses = history.get("train_loss", [])
    if len(train_losses) < min_epochs:
        errors.append(f"Not enough epochs: {len(train_losses)} < {min_epochs}")
    elif train_losses[-1] >= train_losses[0]:
        errors.append(f"Loss did not decrease: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    else:
        print(f"  [PASS] Loss decreased: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")

    train_accs = history.get("train_acc", [])
    if train_accs:
        final_acc = train_accs[-1]
        # With class-specific color bias, model should learn well above random
        if final_acc > 30.0:
            print(f"  [PASS] Train accuracy {final_acc:.1f}% > 30% (random=20% for 5 classes)")
        else:
            print(f"  [WARN] Train accuracy {final_acc:.1f}% <= 30%, may need more epochs")

    val_accs = history.get("val_acc", [])
    if val_accs:
        print(f"  [INFO] Val accuracy: {val_accs[-1]:.1f}%")

    for fname in ["best_model.pt", "last_model.pt"]:
        fpath = os.path.join(checkpoint_dir, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  [PASS] Checkpoint {fname} ({size_kb:.0f} KB)")
        else:
            errors.append(f"Checkpoint {fname} not found")

    return errors




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic ImageFolder E2E: torch-data -> torch-model -> torch-train"
    )
    parser.add_argument("--output-dir", default="./e2e_output",
                        help="Output directory for data, contracts, and checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--backbone", default="resnet18",
                        help="Backbone (resnet18, resnet34, resnet50, efficientnet_b0)")
    parser.add_argument("--device", default="auto",
                        help="Device (auto/cuda/cpu)")
    parser.add_argument("--samples-per-class", type=int, default=50,
                        help="Synthetic images per class")
    args = parser.parse_args()

    output_dir = args.output_dir
    data_dir = os.path.join(output_dir, "data")
    contract_dir = os.path.join(output_dir, "contracts")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(contract_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("Synthetic ImageFolder E2E Pipeline")
    print("  torch-data -> torch-model -> torch-train")
    print("=" * 60)

    # ---- Step 1: Generate synthetic data ----
    print("\n[Step 1/4] Generating synthetic ImageFolder dataset...")
    train_dir, val_dir, num_classes = generate_synthetic_imagefolder(
        data_dir, num_classes=NUM_CLASSES,
        samples_per_class=args.samples_per_class,
    )

    # ---- Step 2: Generate data_contract ----
    print("\n[Step 2/4] Generating data_contract.yaml...")
    dc_path = os.path.join(contract_dir, "data_contract.yaml")
    data_contract = generate_data_contract(train_dir, val_dir, dc_path)

    # ---- Step 3: Generate model_contract ----
    print("\n[Step 3/4] Generating model_contract.yaml...")
    project_spec = {
        "task_type": "image_classification",
        "framework": "pytorch",
        "input_modality": "image",
        "constraints": {},
    }
    if args.backbone:
        if "efficientnet" in args.backbone:
            project_spec["constraints"]["model_size_mb"] = 25
        elif args.backbone == "resnet50":
            project_spec["constraints"]["model_size_mb"] = 250
        elif args.backbone == "resnet18":
            project_spec["constraints"]["model_size_mb"] = 50

    mc_path = os.path.join(contract_dir, "model_contract.yaml")
    model_contract = generate_model_contract(data_contract, project_spec, mc_path)

    # Override backbone if explicitly specified
    if args.backbone and model_contract["model_spec"]["backbone"] != args.backbone:
        print(f"  Overriding backbone: {model_contract['model_spec']['backbone']} -> {args.backbone}")
        model_contract["model_spec"]["backbone"] = args.backbone
        if args.backbone in ("resnet18", "resnet34"):
            model_contract["model_spec"]["architecture"] = "resnet"
            model_contract["model_spec"]["feature_dim"] = 512
        elif args.backbone == "resnet50":
            model_contract["model_spec"]["architecture"] = "resnet"
            model_contract["model_spec"]["feature_dim"] = 2048
        elif "efficientnet" in args.backbone:
            model_contract["model_spec"]["architecture"] = "efficientnet"
            model_contract["model_spec"]["feature_dim"] = 1280

    # ---- Step 4: Train ----
    print("\n[Step 4/4] Training...")
    args.checkpoint_dir = checkpoint_dir
    history = run_training(data_contract, model_contract, train_dir, val_dir, args)

    # ---- Verify ----
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    errors = verify_results(history, checkpoint_dir)

    if errors:
        print(f"\nE2E test FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"\nE2E test PASSED")
        print(f"  Output: {output_dir}")
        print(f"  Contracts: {contract_dir}/")
        print(f"  Checkpoints: {checkpoint_dir}/")
        sys.exit(0)


if __name__ == "__main__":
    main()
