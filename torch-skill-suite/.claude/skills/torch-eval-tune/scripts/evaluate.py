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
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPTS_DIR = SCRIPT_DIR / ".." / ".." / "torch-train" / "scripts"
MODEL_SCRIPTS_DIR = SCRIPT_DIR / ".." / ".." / "torch-model" / "scripts"

for _p in [str(TRAIN_SCRIPTS_DIR.resolve()), str(MODEL_SCRIPTS_DIR.resolve())]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import (
    build_model_from_contract,
    create_synthetic_dataloader,
    create_imagefolder_dataloader,
    _load_yaml,
    _add_template_path,
)
_add_template_path()


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Build dataloader
    preprocessing = data_contract.get("preprocessing", []) if data_contract else []
    if synthetic or (data_dir is None and not data_dir):
        print(f"Using synthetic validation data")
        val_loader = create_synthetic_dataloader(
            input_spec, num_classes, num_samples=200, batch_size=batch_size
        )
    else:
        print(f"Using ImageFolder data from: {data_dir}")
        val_loader = create_imagefolder_dataloader(
            data_dir, input_spec, preprocessing=preprocessing,
            batch_size=batch_size, is_train=False
        )

    # Run inference
    criterion = nn.CrossEntropyLoss()
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

            _, predicted = outputs.max(1)
            all_labels.append(labels.numpy())
            all_preds.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # Compute loss
    val_loss = compute_loss(model, val_loader, criterion, device)

    # Compute metrics
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


# ---------------------------------------------------------------------------
# YAML output
# ---------------------------------------------------------------------------

def _emit_yaml(data, indent=0):
    lines = []
    prefix = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_emit_yaml(value, indent + 1))
            elif isinstance(value, bool):
                lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
            elif isinstance(value, str):
                lines.append(f"{prefix}{key}: {value}")
            elif value is None:
                lines.append(f"{prefix}{key}: null")
            elif isinstance(value, float):
                lines.append(f"{prefix}{key}: {value}")
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                inner = _emit_yaml(item, indent + 1).lstrip()
                lines.append(f"{prefix}- {inner}")
            elif isinstance(item, bool):
                lines.append(f"{prefix}- {'true' if item else 'false'}")
            else:
                lines.append(f"{prefix}- {item}")
    return "\n".join(lines)


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
    model_contract = _load_yaml(args.model_contract)
    data_contract = None
    if args.data_contract:
        data_contract = _load_yaml(args.data_contract)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = evaluate(
        model_contract, args.checkpoint,
        data_contract=data_contract, data_dir=args.data_dir,
        synthetic=args.synthetic, batch_size=args.batch_size, device=device,
    )

    # Output
    yaml_output = _emit_yaml(report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(f"# Evaluation report\n{yaml_output}\n")
        print(f"\nEvaluation report → {args.output}")
    else:
        print(f"\n# Evaluation report\n{yaml_output}")

    # Summary
    print(f"\nAccuracy: {report['metrics']['accuracy']:.1f}%")
    print(f"Macro F1: {report['metrics']['macro_f1']:.4f}")
    print(f"Loss:     {report['loss']:.4f}")


if __name__ == "__main__":
    main()
