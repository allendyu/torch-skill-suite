#!/usr/bin/env python3
"""
PyTorch training loop for image classification (P0).

Consumes data_contract.yaml + model_contract.yaml, builds model and dataloader,
runs training with checkpointing.

Usage:
    # Synthetic data (smoke test)
    python train.py --data-contract data.yaml --model-contract model.yaml --synthetic

    # Real data (ImageFolder)
    python train.py --data-contract data.yaml --model-contract model.yaml --data-dir path/to/data

    # Resume from checkpoint
    python train.py --data-contract data.yaml --model-contract model.yaml --resume checkpoint.pt
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add shared package to path
_SHARED_PYTHON = Path(__file__).resolve().parent.parent.parent.parent.parent / "shared" / "python"
if str(_SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SHARED_PYTHON))

from torch_skill_shared.yaml_utils import load_yaml
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
# Optimizer / Scheduler / Loss builders
# ---------------------------------------------------------------------------

def build_optimizer(model, config=None):
    """Build optimizer from config."""
    cfg = config or {}
    name = cfg.get("name", "adam")
    lr = cfg.get("lr", 0.001)
    weight_decay = cfg.get("weight_decay", 0.0)

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = cfg.get("momentum", 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config=None):
    """Build learning rate scheduler from config."""
    cfg = config or {}
    name = cfg.get("name", "step")
    step_size = cfg.get("step_size", 10)
    gamma = cfg.get("gamma", 0.1)

    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "cosine":
        t_max = cfg.get("t_max", 50)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def build_criterion(task_type="classification", config=None):
    """Build loss function."""
    cfg = config or {}
    if task_type in ("classification", "segmentation"):
        return nn.CrossEntropyLoss()
    elif task_type == "regression":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown task_type for loss: {task_type}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Minimal PyTorch training loop with checkpointing."""

    def __init__(self, model, device, config=None):
        cfg = config or {}
        self.model = model.to(device)
        self.device = device
        self.task_type = cfg.get("task_type", "classification")
        self.optimizer = build_optimizer(model, cfg.get("optimizer"))
        self.scheduler = build_scheduler(self.optimizer, cfg.get("scheduler"))
        self.criterion = build_criterion(self.task_type, cfg.get("loss"))
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.history = defaultdict(list)

    def _forward_pass(self, inputs):
        """Handle both tensor and dict inputs for model forward.
        Also unwraps dict outputs (e.g. DeepLabV3 returns {'out': ...}).
        """
        if isinstance(inputs, dict):
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        else:
            outputs = self.model(inputs.to(self.device))
        if isinstance(outputs, dict):
            return outputs.get("out", outputs)
        return outputs

    def _get_batch_size(self, inputs):
        if isinstance(inputs, dict):
            return next(iter(inputs.values())).size(0)
        return inputs.size(0)

    def train_epoch(self, dataloader):
        """Run one training epoch. Returns average loss and accuracy (or None for regression)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        is_classification = self.task_type in ("classification", "segmentation")

        for batch in dataloader:
            inputs, labels = batch[0], batch[1]
            labels = labels.to(self.device)
            batch_size = self._get_batch_size(inputs)

            self.optimizer.zero_grad()
            outputs = self._forward_pass(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total += batch_size
            if is_classification:
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        if is_classification and total > 0:
            accuracy = 100.0 * correct / total
        else:
            accuracy = None
        return avg_loss, accuracy

    def train(self, train_loader, epochs=5, val_loader=None, checkpoint_dir=None):
        """Run full training loop.

        Args:
            train_loader: Training DataLoader.
            epochs: Number of epochs.
            val_loader: Optional validation DataLoader.
            checkpoint_dir: Directory to save checkpoints.

        Returns:
            Dict of training history.
        """
        start_epoch = self.current_epoch
        end_epoch = start_epoch + epochs
        for epoch in range(start_epoch, end_epoch):
            self.current_epoch = epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if train_acc is not None:
                log_parts = [f"Epoch {epoch+1:3d}/{end_epoch} | loss={train_loss:.4f} acc={train_acc:.1f}%"]
            else:
                log_parts = [f"Epoch {epoch+1:3d}/{end_epoch} | loss={train_loss:.4f}"]

            if val_loader is not None:
                val_loss, val_acc = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                if val_acc is not None:
                    log_parts.append(f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}%")
                else:
                    log_parts.append(f"val_loss={val_loss:.4f}")

            if self.scheduler is not None:
                self.scheduler.step()
                log_parts.append(f"lr={self.scheduler.get_last_lr()[0]:.2e}")

            print(" | ".join(log_parts))

            if checkpoint_dir and train_loss < self.best_loss:
                self.best_loss = train_loss
                self.save_checkpoint(f"{checkpoint_dir}/best_model.pt")

            if checkpoint_dir:
                self.save_checkpoint(f"{checkpoint_dir}/last_model.pt")

        self.current_epoch += epochs
        return dict(self.history)

    def _validate(self, dataloader):
        """Run validation. Returns average loss and accuracy (or None for regression)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        is_classification = self.task_type in ("classification", "segmentation")

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch[0], batch[1]
                labels = labels.to(self.device)
                batch_size = self._get_batch_size(inputs)

                outputs = self._forward_pass(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * batch_size
                total += batch_size
                if is_classification:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        if is_classification and total > 0:
            accuracy = 100.0 * correct / total
        else:
            accuracy = None
        return avg_loss, accuracy

    def save_checkpoint(self, path):
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "history": dict(self.history),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load training checkpoint and restore state.

        Uses full unpickling to restore optimizer/scheduler state.
        Only load checkpoints from trusted sources.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.history = defaultdict(list, checkpoint.get("history", {}))
        return checkpoint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch model for image classification.")
    parser.add_argument("--data-contract", required=True, help="Path to data_contract.yaml")
    parser.add_argument("--model-contract", required=True, help="Path to model_contract.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic random data")
    parser.add_argument("--data-dir", help="Path to real data directory (ImageFolder)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Directory for checkpoints")
    parser.add_argument("--resume", help="Resume from checkpoint path")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # Load contracts
    data_contract = load_yaml(args.data_contract)
    model_contract = load_yaml(args.model_contract)

    input_spec = data_contract.get("input_spec", model_contract.get("input_spec", {}))
    num_classes = (model_contract.get("head_spec", {}).get("num_classes")
                   or data_contract.get("output_spec", {}).get("num_classes", 10))
    task_type = model_contract.get("task_type", "classification")

    print(f"Task: {task_type} / {model_contract.get('data_type', 'image')}")
    print(f"Backbone: {model_contract['model_spec']['backbone']}")
    print(f"Classes: {num_classes}")
    print(f"Input: {input_spec.get('shape', [3, 224, 224])}")

    # Build model
    print("\nBuilding model...")
    model = build_model_from_contract(model_contract)
    trainable, total = sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")

    # DataLoader
    output_dim = model_contract.get("head_spec", {}).get("output_dim", 1)
    preprocessing = data_contract.get("preprocessing", [])
    if args.synthetic or (args.data_dir is None and not args.data_dir):
        print(f"Using synthetic data ({200} samples)")
        train_loader = _create_synthetic_dataloader_for_contract(
            model_contract, input_spec, num_classes, output_dim,
            num_samples=200, batch_size=args.batch_size
        )
        val_loader = _create_synthetic_dataloader_for_contract(
            model_contract, input_spec, num_classes, output_dim,
            num_samples=40, batch_size=args.batch_size
        )
    else:
        print(f"Using ImageFolder data from: {args.data_dir}")
        train_loader = create_imagefolder_dataloader(
            args.data_dir, input_spec, preprocessing=preprocessing,
            batch_size=args.batch_size, is_train=True
        )
        # For validation, try a 'val' sibling directory
        val_dir = str(Path(args.data_dir).parent / "val")
        if Path(val_dir).exists():
            val_loader = create_imagefolder_dataloader(
                val_dir, input_spec, preprocessing=preprocessing,
                batch_size=args.batch_size, is_train=False
            )
        else:
            val_loader = None
            print("  No validation directory found, skipping validation")

    # Trainer config
    trainer_config = {
        "task_type": task_type,
        "optimizer": {"name": "adam", "lr": args.lr, "weight_decay": 0.0},
        "scheduler": {"name": "step", "step_size": 10, "gamma": 0.1},
    }

    trainer = Trainer(model, device, trainer_config)

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    print(f"\nTraining {args.epochs} epoch(s)...")
    print("-" * 60)
    history = trainer.train(
        train_loader,
        epochs=args.epochs,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
    )
    print("-" * 60)

    # Summary
    final_acc = history['train_acc'][-1]
    acc_str = f" acc={final_acc:.1f}%" if final_acc is not None else ""
    print(f"\nFinal: loss={history['train_loss'][-1]:.4f}{acc_str}")
    loss_decreased = history["train_loss"][-1] < history["train_loss"][0]
    print(f"Loss decreased: {'YES' if loss_decreased else 'NO'}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
