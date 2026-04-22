"""Unit tests for torch-eval-tune detection and metrics functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate import compute_metrics
from tune import (
    _detect_overfitting,
    _detect_underfitting,
    _detect_plateau,
    _detect_divergence,
    generate_tuning_plan,
)


class TestComputeMetrics:
    def test_perfect_accuracy(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        preds = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(labels, preds, num_classes=3)
        assert metrics["accuracy"] == 100.0
        assert metrics["macro_f1"] == 1.0
        for c in range(3):
            assert metrics["per_class_f1"][c] == 1.0

    def test_zero_accuracy(self):
        labels = np.array([0, 0, 0, 0])
        preds = np.array([1, 1, 1, 1])
        metrics = compute_metrics(labels, preds, num_classes=2)
        assert metrics["accuracy"] == 0.0
        assert metrics["macro_f1"] == 0.0

    def test_confusion_matrix_shape(self):
        labels = np.random.randint(0, 5, size=100)
        preds = np.random.randint(0, 5, size=100)
        metrics = compute_metrics(labels, preds, num_classes=5)
        cm = np.array(metrics["confusion_matrix"])
        assert cm.shape == (5, 5)
        assert cm.sum() == 100

    def test_per_class_metrics_keys(self):
        labels = np.array([0, 1, 2, 0, 1])
        preds = np.array([0, 1, 1, 0, 2])
        metrics = compute_metrics(labels, preds, num_classes=3)
        for key in ["per_class_precision", "per_class_recall", "per_class_f1"]:
            assert set(metrics[key].keys()) == {0, 1, 2}

    def test_macro_f1_range(self):
        labels = np.random.randint(0, 5, size=200)
        preds = np.random.randint(0, 5, size=200)
        metrics = compute_metrics(labels, preds, num_classes=5)
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 100.0


class TestOverfittingDetection:
    def test_detects_overfitting(self):
        is_over, gap = _detect_overfitting(
            [2.0, 1.0, 0.3, 0.1, 0.05],
            [2.1, 1.5, 1.8, 2.0, 2.2],
        )
        assert is_over is True
        assert gap > 0

    def test_no_overfitting_when_both_decreasing(self):
        is_over, _ = _detect_overfitting(
            [2.0, 1.5, 1.0, 0.8, 0.5],
            [2.1, 1.6, 1.2, 0.9, 0.6],
        )
        assert is_over is False

    def test_short_history_no_overfitting(self):
        is_over, _ = _detect_overfitting([1.0, 0.8], [1.1, 0.9])
        assert is_over is False


class TestUnderfittingDetection:
    def test_detects_underfitting(self):
        is_under = _detect_underfitting(
            [2.0, 1.9, 1.85, 1.83, 1.82],
            [2.1, 2.0, 1.95, 1.93, 1.92],
        )
        assert is_under is True

    def test_no_underfitting_when_improving(self):
        is_under = _detect_underfitting(
            [3.0, 2.0, 1.0, 0.5, 0.3],
            [3.1, 2.1, 1.2, 0.6, 0.4],
        )
        assert is_under is False

    def test_no_underfitting_below_target(self):
        is_under = _detect_underfitting(
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.6, 0.5, 0.4, 0.3, 0.2],
        )
        assert is_under is False

    def test_short_history_no_underfitting(self):
        is_under = _detect_underfitting([2.0, 1.9], [2.1, 2.0])
        assert is_under is False

    def test_custom_target_loss(self):
        is_under = _detect_underfitting(
            [1.2, 1.18, 1.17],
            [1.3, 1.28, 1.27],
            target_loss=0.5,
        )
        assert is_under is True


class TestPlateauDetection:
    def test_detects_plateau(self):
        is_plat = _detect_plateau(
            [0.8, 0.6, 0.55, 0.548, 0.547, 0.546]
        )
        assert is_plat is True

    def test_no_plateau_when_improving(self):
        is_plat = _detect_plateau(
            [2.0, 1.5, 1.0, 0.8, 0.6, 0.4]
        )
        assert is_plat is False

    def test_short_history_no_plateau(self):
        is_plat = _detect_plateau([1.0, 0.9, 0.8])
        assert is_plat is False


class TestDivergenceDetection:
    def test_detects_divergence(self):
        is_div = _detect_divergence([0.5, 0.6, 0.8, 1.2])
        assert is_div is True

    def test_no_divergence_when_decreasing(self):
        is_div = _detect_divergence([2.0, 1.5, 1.0, 0.5])
        assert is_div is False

    def test_short_history_no_divergence(self):
        is_div = _detect_divergence([1.0, 1.5])
        assert is_div is False


class TestGenerateTuningPlan:
    def test_overfitting_suggestion(self):
        history = {
            "train_loss": [2.0, 1.0, 0.3, 0.1, 0.05],
            "val_loss": [2.1, 1.5, 1.8, 2.0, 2.2],
            "train_acc": [30, 50, 70, 85, 95],
            "val_acc": [28, 45, 42, 40, 38],
        }
        plan = generate_tuning_plan(history)
        cats = [s["category"] for s in plan["suggestions"]]
        assert "regularization" in cats

    def test_underfitting_suggestion(self):
        history = {
            "train_loss": [2.0, 1.95, 1.93, 1.92, 1.91],
            "val_loss": [2.1, 2.05, 2.03, 2.02, 2.01],
            "train_acc": [30, 31, 31, 31, 31],
            "val_acc": [28, 29, 29, 29, 29],
        }
        plan = generate_tuning_plan(history)
        cats = [s["category"] for s in plan["suggestions"]]
        assert "capacity" in cats

    def test_divergence_suggestion(self):
        history = {
            "train_loss": [0.5, 0.6, 0.8, 1.2],
            "val_loss": [0.6, 0.7, 0.9, 1.3],
            "train_acc": [80, 70, 50, 30],
            "val_acc": [75, 65, 45, 25],
        }
        plan = generate_tuning_plan(history)
        cats = [s["category"] for s in plan["suggestions"]]
        assert "learning_rate" in cats

    def test_accuracy_suggestion(self):
        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_acc": [50, 55, 60],
            "val_acc": [45, 50, 55],
        }
        eval_report = {"metrics": {"accuracy": 55.0}}
        plan = generate_tuning_plan(history, eval_report, target_accuracy=80.0)
        cats = [s["category"] for s in plan["suggestions"]]
        assert "accuracy" in cats

    def test_class_balance_suggestion(self):
        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_acc": [50, 55, 60],
            "val_acc": [45, 50, 55],
        }
        eval_report = {
            "metrics": {
                "per_class_f1": {0: 0.9, 1: 0.85, 2: 0.3},
                "accuracy": 70.0,
            }
        }
        plan = generate_tuning_plan(history, eval_report)
        cats = [s["category"] for s in plan["suggestions"]]
        assert "class_balance" in cats

    def test_healthy_training_fine_tuning(self):
        history = {
            "train_loss": [1.0, 0.5, 0.2, 0.1],
            "val_loss": [1.1, 0.6, 0.3, 0.2],
            "train_acc": [50, 70, 85, 92],
            "val_acc": [48, 68, 82, 90],
        }
        plan = generate_tuning_plan(history)
        cats = [s["category"] for s in plan["suggestions"]]
        assert "fine_tuning" in cats

    def test_summary_fields(self):
        history = {
            "train_loss": [1.0, 0.5, 0.2],
            "val_loss": [1.1, 0.6, 0.3],
            "train_acc": [50, 70, 85],
            "val_acc": [48, 68, 82],
        }
        plan = generate_tuning_plan(history)
        assert plan["summary"]["num_epochs"] == 3
        assert plan["summary"]["final_train_loss"] == 0.2
        assert plan["summary"]["final_val_loss"] == 0.3
        assert plan["summary"]["final_train_acc"] == 85
        assert plan["summary"]["final_val_acc"] == 82

    def test_empty_history(self):
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }
        plan = generate_tuning_plan(history)
        assert plan["summary"]["num_epochs"] == 0
        assert plan["summary"]["final_train_loss"] is None
        # Should still get fine_tuning suggestion
        assert len(plan["suggestions"]) >= 1
