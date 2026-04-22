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


# ---------------------------------------------------------------------------
# YAML helpers (fallback when PyYAML is unavailable)
# ---------------------------------------------------------------------------

try:
    import yaml
except ImportError:
    yaml = None


class SimpleYAMLParser:
    def __init__(self, text):
        self.lines = self._prepare_lines(text)

    def _strip_inline_comment(self, line):
        in_single, in_double = False, False
        escaped = False
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if ch == "#" and not in_single and not in_double:
                if i == 0 or line[i - 1].isspace():
                    return line[:i].rstrip()
        return line.rstrip()

    def _prepare_lines(self, text):
        return [self._strip_inline_comment(l) for l in text.splitlines()]

    def parse(self):
        result = {}
        stack = [(None, 0, result)]
        for line in self.lines:
            content = line.rstrip()
            if not content or content.isspace():
                continue
            indent = len(line) - len(line.lstrip())
            key, value = self._parse_line(content)
            if key is None:
                continue
            while stack and stack[-1][1] >= indent:
                stack.pop()
            parent = stack[-1][2]
            if value is not None:
                parent[key] = value
            else:
                new = {}
                parent[key] = new
                stack.append((key, indent, new))
        return result

    def _parse_line(self, line):
        content = line.strip()
        if content.startswith("- "):
            content = content[2:]
            key = None
        elif ":" in content:
            parts = content.split(":", 1)
            key = parts[0].strip().strip("'\"")
            rest = parts[1].strip() if len(parts) > 1 else ""
            if not rest:
                return key, None
            content = rest
        else:
            return None, None
        return key, self._parse_value(content)

    def _parse_value(self, text):
        if text in ("true", "True", "yes"):
            return True
        if text in ("false", "False", "no"):
            return False
        if text in ("null", "None", "~"):
            return None
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        try:
            return int(text)
        except (ValueError, TypeError):
            pass
        try:
            return float(text)
        except (ValueError, TypeError):
            pass
        return text


def _load_yaml(path):
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return SimpleYAMLParser(fh.read()).parse()


# ---------------------------------------------------------------------------
# Model building (delegates to torch-model templates)
# ---------------------------------------------------------------------------

def _add_template_path():
    """Add torch-model templates to sys.path."""
    script_dir = Path(__file__).resolve().parent
    # Add the torch-model skill directory (parent of templates/) so that
    # "from templates.image_classification.resnet import ..." works.
    model_skill_dir = script_dir / ".." / ".." / "torch-model"
    if model_skill_dir.exists():
        path = str(model_skill_dir.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def build_model_from_contract(model_contract):
    """Build a PyTorch model from a model_contract dict."""
    _add_template_path()
    architecture = model_contract["model_spec"]["architecture"]
    backbone = model_contract["model_spec"]["backbone"]
    head_spec = model_contract.get("head_spec", {})

    config = {
        "architecture": architecture,
        "backbone": backbone,
        "pretrained": model_contract["model_spec"].get("pretrained", False),
        "in_channels": model_contract["model_spec"].get("in_channels", 3),
        "head": {
            "type": head_spec.get("type", "linear_cls"),
            "num_classes": head_spec.get("num_classes", 1000),
            "pooling": head_spec.get("pooling", "avg"),
            "dropout": head_spec.get("dropout", 0.0),
        },
    }

    if architecture == "resnet":
        from templates.image_classification.resnet import build_resnet
        return build_resnet(config)
    elif architecture == "efficientnet":
        from templates.image_classification.efficientnet import build_efficientnet
        return build_efficientnet(config)
    elif architecture == "mlp":
        from templates.tabular_classification.mlp import build_mlp
        config["in_features"] = model_contract["model_spec"].get("in_features", 10)
        config["head"] = {
            "type": head_spec.get("type", "linear_cls"),
            "num_classes": head_spec.get("num_classes"),
            "output_dim": head_spec.get("output_dim"),
            "dropout": head_spec.get("dropout", 0.0),
        }
        return build_mlp(config)
    elif architecture == "bert":
        from templates.text_classification.bert import build_transformer
        config["head"] = {
            "type": head_spec.get("type", "pooled_linear_cls"),
            "num_classes": head_spec.get("num_classes", 2),
            "dropout": head_spec.get("dropout", 0.1),
        }
        return build_transformer(config)
    elif architecture == "deeplabv3":
        from templates.image_segmentation.deeplabv3 import build_deeplabv3
        config["head"] = {"num_classes": head_spec.get("num_classes", 21)}
        return build_deeplabv3(config)
    elif architecture == "unet":
        from templates.image_segmentation.unet import build_unet
        config["head"] = {"num_classes": head_spec.get("num_classes", 21)}
        return build_unet(config)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


# ---------------------------------------------------------------------------
# Synthetic DataLoader (for smoke testing)
# ---------------------------------------------------------------------------

def create_synthetic_dataloader(input_spec, num_classes, num_samples=200, batch_size=16):
    """Create a DataLoader with random synthetic data.

    Args:
        input_spec: Dict with 'shape' and 'dtype'.
        num_classes: Number of output classes (for classification; ignored for regression).
        num_samples: Total number of synthetic samples.
        batch_size: Batch size.

    Returns:
        DataLoader yielding (inputs, labels) tuples.
    """
    shape = input_spec.get("shape", [3, 224, 224])
    dtype_str = input_spec.get("dtype", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)

    if dtype in (torch.int64, torch.int32, torch.long):
        inputs = torch.randint(0, 1000, (num_samples, *shape), dtype=torch.long)
    else:
        inputs = torch.randn(num_samples, *shape, dtype=dtype)

    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_synthetic_text_dataloader(input_spec, num_classes, num_samples=100, batch_size=8, seq_length=None):
    """Create a DataLoader with random token-like data for text classification.

    Args:
        input_spec: Dict with 'shape' for [max_seq_length] and 'dtype'.
        num_classes: Number of output classes.
        num_samples: Total number of synthetic samples.
        batch_size: Batch size.
        seq_length: Sequence length (defaults to input_spec max_seq_length or 128).

    Returns:
        DataLoader yielding (dict_of_tensors, labels) tuples where dict has
        'input_ids' and 'attention_mask' keys.
    """
    if seq_length is None:
        seq_length = input_spec.get("max_seq_length", input_spec.get("shape", [128])[0])

    vocab_size = input_spec.get("vocab_size", 30522)
    input_ids = torch.randint(1, vocab_size, (num_samples, seq_length), dtype=torch.long)
    input_ids[:, 0] = 101  # [CLS] token
    attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)

    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(input_ids, attention_mask, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Wrap to return dict format
    return _DictDataLoaderWrapper(loader)


class _DictDataLoaderWrapper:
    """Wraps a DataLoader to yield (dict, labels) instead of (input_ids, mask, labels)."""

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for input_ids, attention_mask, labels in self._loader:
            yield {"input_ids": input_ids, "attention_mask": attention_mask}, labels

    def __len__(self):
        return len(self._loader)


def create_synthetic_segmentation_dataloader(input_spec, num_classes, num_samples=40, batch_size=4, image_size=None):
    """Create a DataLoader with random images and integer mask labels for segmentation.

    Args:
        input_spec: Dict with 'shape' for [C, H, W] and 'dtype'.
        num_classes: Number of segmentation classes.
        num_samples: Total number of synthetic samples.
        batch_size: Batch size.
        image_size: (H, W) override. Defaults to input_spec shape.

    Returns:
        DataLoader yielding (images, masks) tuples.
    """
    shape = input_spec.get("shape", [3, 224, 224])
    dtype_str = input_spec.get("dtype", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)

    if image_size is None:
        image_size = (shape[1], shape[2]) if len(shape) >= 3 else (224, 224)
    C = shape[0] if len(shape) >= 1 else 3

    inputs = torch.randn(num_samples, C, *image_size, dtype=dtype)
    masks = torch.randint(0, num_classes, (num_samples, *image_size), dtype=torch.long)
    dataset = TensorDataset(inputs, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_synthetic_regression_dataloader(input_spec, output_dim=1, num_samples=200, batch_size=16):
    """Create a DataLoader with random synthetic data for regression tasks.

    Args:
        input_spec: Dict with 'shape' and 'dtype'.
        output_dim: Number of regression targets.
        num_samples: Total number of synthetic samples.
        batch_size: Batch size.

    Returns:
        DataLoader yielding (inputs, targets) tuples with float32 labels.
    """
    shape = input_spec.get("shape", [10])
    dtype_str = input_spec.get("dtype", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)

    inputs = torch.randn(num_samples, *shape, dtype=dtype)
    targets = torch.randn(num_samples, output_dim, dtype=torch.float32)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_imagefolder_dataloader(data_dir, input_spec, preprocessing=None, batch_size=16, is_train=True):
    """Create a DataLoader from an ImageFolder directory.

    Args:
        data_dir: Path to ImageFolder-structured directory (with class subdirs).
        input_spec: Dict with 'shape' for target [C, H, W] after transforms.
        preprocessing: List of preprocessing steps from data_contract.
        batch_size: Batch size.
        is_train: Whether this is a training loader (enables shuffle).

    Returns:
        DataLoader yielding (inputs, labels) tuples.
    """
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform_list = []
    target_shape = input_spec.get("shape", [3, 224, 224])
    if len(target_shape) >= 2:
        h, w = target_shape[1], target_shape[2]
        transform_list.append(transforms.Resize((h, w)))

    transform_list.append(transforms.ToTensor())

    # Apply normalization from contract preprocessing
    if preprocessing:
        for step in preprocessing:
            name = step.get("name", "") if isinstance(step, dict) else ""
            params = step.get("params", {}) if isinstance(step, dict) else {}
            if name == "normalize":
                mean = params.get("mean", [0.485, 0.456, 0.406])
                std = params.get("std", [0.229, 0.224, 0.225])
                transform_list.append(transforms.Normalize(mean=mean, std=std))

    if is_train:
        # Add basic augmentation
        transform_list.insert(1, transforms.RandomHorizontalFlip())

    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2)


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
        """Load training checkpoint and restore state."""
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
    data_contract = _load_yaml(args.data_contract)
    model_contract = _load_yaml(args.model_contract)

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
    preprocessing = data_contract.get("preprocessing", [])
    if args.synthetic or (args.data_dir is None and not args.data_dir):
        print(f"Using synthetic data ({200} samples)")
        train_loader = create_synthetic_dataloader(
            input_spec, num_classes, num_samples=200, batch_size=args.batch_size
        )
        val_loader = create_synthetic_dataloader(
            input_spec, num_classes, num_samples=40, batch_size=args.batch_size
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
    print(f"\nFinal: loss={history['train_loss'][-1]:.4f} acc={history['train_acc'][-1]:.1f}%")
    loss_decreased = history["train_loss"][-1] < history["train_loss"][0]
    print(f"Loss decreased: {'YES' if loss_decreased else 'NO'}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
