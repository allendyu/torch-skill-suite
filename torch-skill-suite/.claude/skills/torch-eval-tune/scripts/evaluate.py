#!/usr/bin/env python3
"""
Model evaluation for classification tasks (P0).

Consumes model_contract.yaml + checkpoint + data_contract.yaml,
runs inference on validation set, computes metrics, and outputs
a structured evaluation report.

Usage:
    python evaluate.py --model-contract mc.yaml --checkpoint best_model.pt \\
                       --data-contract dc.yaml --data-dir path/to/val \\
                       --output eval_report.yaml
    python evaluate.py --model-contract mc.yaml --checkpoint best_model.pt --synthetic
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent

# Add shared package to path
_SHARED_PYTHON = SCRIPT_DIR.parent.parent.parent.parent.parent / "shared" / "python"
if str(_SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SHARED_PYTHON))

from torch_skill_shared.yaml_utils import load_yaml, emit_yaml
from torch_skill_shared.model_builder import (
    build_model_from_contract,
    create_synthetic_dataloader,
    create_synthetic_text_dataloader,
    create_synthetic_segmentation_dataloader,
    create_synthetic_regression_dataloader,
    create_imagefolder_dataloader,
    _create_synthetic_dataloader_for_contract,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(all_labels, all_preds, num_classes):
    """Compute classification metrics from accumulated predictions.

    Args:
        all_labels: (N,) int64 array of true labels.
        all_preds: (N,) int64 array of predicted labels.
        num_classes: Number of classes.

    Returns:
        Dict with accuracy, per_class_precision, per_class_recall, per_class_f1,
        confusion_matrix.
    """
    accuracy = float((all_labels == all_preds).mean()) * 100.0

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_precision[int(c)] = round(float(precision), 4)
        per_class_recall[int(c)] = round(float(recall), 4)
        per_class_f1[int(c)] = round(float(f1), 4)

    macro_f1 = float(np.mean(list(per_class_f1.values())))

    return {
        "accuracy": round(accuracy, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
    }


def compute_regression_metrics(all_labels, all_preds):
    """Compute regression metrics from accumulated predictions.

    Args:
        all_labels: (N,) float array of true values.
        all_preds: (N,) float array of predicted values.

    Returns:
        Dict with mse, mae, rmse, r2.
    """
    mse = float(np.mean((all_labels - all_preds) ** 2))
    mae = float(np.mean(np.abs(all_labels - all_preds)))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((all_labels - all_preds) ** 2)
    ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def compute_loss(model, dataloader, criterion, device):
    """Compute average loss over a dataloader."""
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0], batch[1]
            labels = labels.to(device)
            if isinstance(inputs, dict):
                outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            else:
                outputs = model(inputs.to(device))
            if isinstance(outputs, dict):
                outputs = outputs.get("out", outputs)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
    return total_loss / total if total > 0 else float("inf")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(model_contract, checkpoint_path, data_contract=None,
             data_dir=None, input_spec=None, num_classes=None,
             synthetic=False, batch_size=16, device=None):
    """Run full evaluation.

    Args:
        model_contract: Parsed model_contract dict.
        checkpoint_path: Path to .pt checkpoint.
        data_contract: Optional parsed data_contract dict.
        data_dir: Path to ImageFolder data.
        input_spec: Dict with shape/dtype (used if no data_contract).
        num_classes: Number of classes (used if no data_contract).
        synthetic: Use synthetic random data.
        batch_size: Batch size.
        device: torch device string.

    Returns:
        Dict with loss, metrics, metadata.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve num_classes and input_spec
    if data_contract:
        if input_spec is None:
            input_spec = data_contract.get("input_spec", {})
        if num_classes is None:
            num_classes = (data_contract.get("output_spec", {}).get("num_classes")
                           or model_contract.get("head_spec", {}).get("num_classes", 10))
    else:
        if input_spec is None:
            input_spec = model_contract.get("input_spec", {})
        if num_classes is None:
            num_classes = model_contract.get("head_spec", {}).get("num_classes", 10)

    print(f"Device: {device}")
    print(f"Backbone: {model_contract['model_spec']['backbone']}")
    print(f"Classes: {num_classes}")

    # Build model and load checkpoint
    print("\nBuilding model...")
    model = build_model_from_contract(model_contract)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        print("Warning: This checkpoint requires full unpickling (may contain Python objects).")
        print("  Only load checkpoints from trusted sources.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Build dataloader
    preprocessing = data_contract.get("preprocessing", []) if data_contract else []
    output_dim = model_contract.get("head_spec", {}).get("output_dim", 1)
    if synthetic or (data_dir is None and not data_dir):
        print(f"Using synthetic validation data")
        val_loader = _create_eval_dataloader(
            model_contract, input_spec, num_classes, output_dim,
            num_samples=200, batch_size=batch_size
        )
    else:
        print(f"Using ImageFolder data from: {data_dir}")
        val_loader = create_imagefolder_dataloader(
            data_dir, input_spec, preprocessing=preprocessing,
            batch_size=batch_size, is_train=False
        )

    # Run inference
    task_type = model_contract.get("task_type", "classification")
    criterion = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()
    all_labels = []
    all_preds = []

    print(f"\nRunning evaluation on {len(val_loader)} batches...")
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0], batch[1]
            if isinstance(inputs, dict):
                outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            else:
                outputs = model(inputs.to(device))
            if isinstance(outputs, dict):
                outputs = outputs.get("out", outputs)

            if task_type == "regression":
                all_labels.append(labels.numpy())
                all_preds.append(outputs.cpu().numpy())
            else:
                _, predicted = outputs.max(1)
                all_labels.append(labels.numpy())
                all_preds.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Compute loss
    val_loss = compute_loss(model, val_loader, criterion, device)

    # Compute metrics
    if task_type == "regression":
        metrics = compute_regression_metrics(all_labels, all_preds)
    else:
        metrics = compute_metrics(all_labels, all_preds, num_classes)

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "num_samples": int(len(all_labels)),
        "loss": round(val_loss, 4),
        "metrics": metrics,
        "metadata": {
            "route": model_contract.get("metadata", {}).get("route", "unknown"),
            "backbone": model_contract["model_spec"]["backbone"],
            "num_classes": num_classes,
        },
    }

    return report


def _create_eval_dataloader(model_contract, input_spec, num_classes, output_dim=1,
                             num_samples=200, batch_size=16):
    """Create the appropriate synthetic DataLoader for evaluation.

    Dispatches to route-specific dataloaders based on model architecture.
    """
    architecture = model_contract.get("model_spec", {}).get("architecture", "")
    task_type = model_contract.get("task_type", "classification")

    if architecture == "bert":
        return create_synthetic_text_dataloader(
            input_spec, num_classes, num_samples=num_samples, batch_size=batch_size
        )
    elif architecture in ("deeplabv3", "unet"):
        return create_synthetic_segmentation_dataloader(
            input_spec, num_classes, num_samples=num_samples, batch_size=batch_size
        )
    elif architecture == "mlp" and task_type == "regression":
        return create_synthetic_regression_dataloader(
            input_spec, output_dim=output_dim, num_samples=num_samples, batch_size=batch_size
        )
    else:
        return create_synthetic_dataloader(
            input_spec, num_classes, num_samples=num_samples, batch_size=batch_size
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained classification model.")
    parser.add_argument("--model-contract", required=True, help="Path to model_contract.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--data-contract", help="Path to data_contract.yaml")
    parser.add_argument("--data-dir", help="Path to validation data directory (ImageFolder)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output", "-o", help="Output path for eval report (default: stdout)")
    args = parser.parse_args()

    # Load contracts
    model_contract = load_yaml(args.model_contract)
    data_contract = None
    if args.data_contract:
        data_contract = load_yaml(args.data_contract)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = evaluate(
        model_contract, args.checkpoint,
        data_contract=data_contract, data_dir=args.data_dir,
        synthetic=args.synthetic, batch_size=args.batch_size, device=device,
    )

    # Output
    yaml_output = emit_yaml(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(f"# Evaluation report\n{yaml_output}\n")
        print(f"\nEvaluation report → {args.output}")
    else:
        print(f"\n# Evaluation report\n{yaml_output}")

    # Summary
    if "accuracy" in report["metrics"]:
        print(f"\nAccuracy: {report['metrics']['accuracy']:.1f}%")
        print(f"Macro F1: {report['metrics']['macro_f1']:.4f}")
    else:
        print(f"\nMSE: {report['metrics']['mse']:.4f}")
        print(f"R²:  {report['metrics']['r2']:.4f}")
    print(f"Loss:     {report['loss']:.4f}")


if __name__ == "__main__":
    main()
