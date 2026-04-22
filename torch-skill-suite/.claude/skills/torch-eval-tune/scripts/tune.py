#!/usr/bin/env python3
"""
Tuning plan generator.

Analyzes training history and evaluation results to produce
prioritized, actionable tuning suggestions.

Usage:
    python tune.py --history train_history.json --eval eval_report.yaml
    python tune.py --history train_history.json --output tuning_plan.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _detect_overfitting(train_losses, val_losses):
    """Detect overfitting: train loss decreasing, val loss increasing."""
    if len(train_losses) < 3 or len(val_losses) < 3:
        return False, 0.0
    train_trend = train_losses[-1] - train_losses[0]
    val_trend = val_losses[-1] - val_losses[min(2, len(val_losses) - 1)]
    gap = val_losses[-1] - train_losses[-1]
    return val_trend > 0 and train_trend < 0, gap


def _detect_underfitting(train_losses, val_losses, target_loss=None):
    """Detect underfitting: both losses high and barely improving."""
    if len(train_losses) < 3:
        return False
    if target_loss is None:
        target_loss = 1.5
    recent_improvement = abs(train_losses[-1] - train_losses[max(0, len(train_losses) - 3)])
    return train_losses[-1] > target_loss and recent_improvement < 0.05


def _detect_plateau(train_losses, window=3):
    """Detect plateau: loss flattened in recent epochs."""
    if len(train_losses) < window * 2:
        return False
    recent = train_losses[-window:]
    return max(recent) - min(recent) < 0.01


def _detect_divergence(train_losses):
    """Detect divergence: loss increasing rapidly."""
    if len(train_losses) < 3:
        return False
    return train_losses[-1] > train_losses[0] * 1.5


# ---------------------------------------------------------------------------
# Suggestion generation
# ---------------------------------------------------------------------------

def generate_tuning_plan(history, eval_report=None, target_accuracy=None):
    """Generate a prioritized tuning plan from training history and eval results.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists.
        eval_report: Optional dict with evaluation metrics.
        target_accuracy: Optional target accuracy threshold.

    Returns:
        Dict with analysis summary and prioritized suggestions.
    """
    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    train_accs = history.get("train_acc", [])
    val_accs = history.get("val_acc", [])

    analysis = {}
    suggestions = []

    # ---- Overfitting ----
    is_overfitting, gap = _detect_overfitting(train_losses, val_losses)
    analysis["overfitting"] = {"detected": is_overfitting, "train_val_gap": round(gap, 4)}

    if is_overfitting:
        suggestions.append({
            "priority": 1,
            "category": "regularization",
            "action": "Increase regularization to reduce overfitting",
            "details": [
                "Increase dropout rate (try +0.1 to +0.2)",
                "Add or increase weight decay (try 1e-4 to 5e-4)",
                "Add data augmentation (RandomHorizontalFlip, RandomCrop, ColorJitter)",
                "Consider early stopping based on val_loss",
            ],
        })

    # ---- Underfitting ----
    is_underfitting = _detect_underfitting(train_losses, val_losses)
    analysis["underfitting"] = {"detected": is_underfitting}

    if is_underfitting:
        suggestions.append({
            "priority": 1,
            "category": "capacity",
            "action": "Increase model capacity or extend training",
            "details": [
                "Train for more epochs",
                "Try a larger backbone (resnet34 -> resnet50)",
                "Reduce regularization if currently applied",
                "Check if learning rate is too low",
            ],
        })

    # ---- Plateau ----
    is_plateau = _detect_plateau(train_losses)
    analysis["plateau"] = {"detected": is_plateau}

    if is_plateau:
        suggestions.append({
            "priority": 2,
            "category": "learning_rate",
            "action": "Loss plateau detected — adjust learning rate schedule",
            "details": [
                "Reduce learning rate by 0.5x to 0.1x",
                "Switch to cosine annealing scheduler",
                "Try learning rate warmup for first few epochs",
            ],
        })

    # ---- Divergence ----
    is_diverging = _detect_divergence(train_losses)
    analysis["divergence"] = {"detected": is_diverging}

    if is_diverging:
        suggestions.append({
            "priority": 1,
            "category": "learning_rate",
            "action": "Training diverging — reduce learning rate immediately",
            "details": [
                "Reduce learning rate by 10x",
                "Check for NaN/Inf in gradients (use gradient clipping)",
                "Verify data normalization (mean/std values)",
            ],
        })

    # ---- Accuracy-based suggestions ----
    if eval_report and target_accuracy:
        acc = eval_report.get("metrics", {}).get("accuracy", 0)
        if acc < target_accuracy:
            suggestions.append({
                "priority": 1,
                "category": "accuracy",
                "action": f"Accuracy {acc:.1f}% below target {target_accuracy:.1f}%",
                "details": [
                    "Check per-class metrics for class imbalance issues",
                    "Review misclassified examples for patterns",
                    "Consider collecting more data for underperforming classes",
                ],
            })

    # ---- Class imbalance check ----
    if eval_report:
        per_class_f1 = eval_report.get("metrics", {}).get("per_class_f1", {})
        if per_class_f1:
            f1_values = list(per_class_f1.values())
            if len(f1_values) >= 2 and max(f1_values) - min(f1_values) > 0.2:
                worst = min(per_class_f1, key=per_class_f1.get)
                best = max(per_class_f1, key=per_class_f1.get)
                suggestions.append({
                    "priority": 2,
                    "category": "class_balance",
                    "action": "Significant class performance gap detected",
                    "details": [
                        f"Best class {best}: F1={per_class_f1[best]:.4f}",
                        f"Worst class {worst}: F1={per_class_f1[worst]:.4f}",
                        "Consider class-weighted loss or oversampling for weak classes",
                    ],
                })

    # Sort by priority
    suggestions.sort(key=lambda s: s["priority"])

    # ---- If everything looks good ----
    if not suggestions:
        suggestions.append({
            "priority": 3,
            "category": "fine_tuning",
            "action": "Training looks healthy — consider fine-tuning",
            "details": [
                "Try longer training with cosine annealing",
                "Experiment with different optimizers (SGD with momentum)",
                "Fine-tune with a lower learning rate",
            ],
        })

    return {
        "analysis": analysis,
        "suggestions": suggestions,
        "summary": {
            "num_epochs": len(train_losses),
            "final_train_loss": round(train_losses[-1], 4) if train_losses else None,
            "final_val_loss": round(val_losses[-1], 4) if val_losses else None,
            "final_train_acc": round(train_accs[-1], 1) if train_accs else None,
            "final_val_acc": round(val_accs[-1], 1) if val_accs else None,
        },
    }


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
            elif isinstance(item, str):
                lines.append(f"{prefix}- {item}")
            else:
                lines.append(f"{prefix}- {item}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate a tuning plan from training history.")
    parser.add_argument("--history", required=True, help="Path to training history JSON file")
    parser.add_argument("--eval", help="Path to eval report YAML/JSON")
    parser.add_argument("--target-accuracy", type=float, help="Target accuracy threshold")
    parser.add_argument("--output", "-o", help="Output path for tuning_plan.yaml")
    args = parser.parse_args()

    # Load history
    with open(args.history, "r", encoding="utf-8") as fh:
        history = json.load(fh)

    # Load eval report if provided
    eval_report = None
    if args.eval:
        eval_path = Path(args.eval)
        if eval_path.suffix in (".json",):
            with open(args.eval, "r", encoding="utf-8") as fh:
                eval_report = json.load(fh)
        else:
            # Simple YAML parse for eval report
            sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / ".." / "torch-train" / "scripts"))
            from train import _load_yaml
            eval_report = _load_yaml(args.eval)

    plan = generate_tuning_plan(history, eval_report, args.target_accuracy)

    yaml_output = _emit_yaml(plan)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(f"# Tuning plan\n{yaml_output}\n")
        print(f"Tuning plan → {args.output}")
    else:
        print(f"# Tuning plan\n{yaml_output}")

    # Print suggestions
    print(f"\nSuggestions ({len(plan['suggestions'])}):")
    for s in plan["suggestions"]:
        print(f"  [P{s['priority']}] [{s['category']}] {s['action']}")


if __name__ == "__main__":
    main()
