"""Shared model builder and synthetic dataloader factories.

Used by torch-train, torch-eval-tune, and torch-infer-deploy to avoid
cross-skill imports via fragile sys.path manipulation.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Template path discovery
# ---------------------------------------------------------------------------

def _ensure_template_path():
    """Add torch-model templates to sys.path so template imports work.

    Walks up from this file (shared/python/torch_skill_shared/model_builder.py)
    to find the project root where .claude/skills/ lives.
    """
    # shared/python/torch_skill_shared/ -> shared/python/ -> shared/ -> project root
    root = Path(__file__).resolve().parent.parent.parent.parent
    model_skill_dir = root / ".claude" / "skills" / "torch-model"
    if model_skill_dir.exists():
        path = str(model_skill_dir.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model_from_contract(model_contract: dict):
    """Build a PyTorch model from a model_contract dict.

    Supports architectures: resnet, efficientnet, mlp, bert, deeplabv3, unet.
    """
    _ensure_template_path()

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
# Synthetic DataLoader factories
# ---------------------------------------------------------------------------

def create_synthetic_dataloader(input_spec, num_classes, num_samples=200, batch_size=16):
    """Create a DataLoader with random synthetic image data."""
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


def create_synthetic_text_dataloader(input_spec, num_classes, num_samples=100,
                                      batch_size=8, seq_length=None):
    """Create a DataLoader with random token-like data for text classification."""
    if seq_length is None:
        seq_length = input_spec.get("max_seq_length", input_spec.get("shape", [128])[0])

    vocab_size = input_spec.get("vocab_size", 30522)
    input_ids = torch.randint(1, vocab_size, (num_samples, seq_length), dtype=torch.long)
    input_ids[:, 0] = 101  # [CLS] token
    attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)

    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(input_ids, attention_mask, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
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


def create_synthetic_segmentation_dataloader(input_spec, num_classes, num_samples=40,
                                               batch_size=4, image_size=None):
    """Create a DataLoader with random images and integer mask labels for segmentation."""
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


def create_synthetic_regression_dataloader(input_spec, output_dim=1, num_samples=200,
                                             batch_size=16):
    """Create a DataLoader with random synthetic data for regression tasks."""
    shape = input_spec.get("shape", [10])
    dtype_str = input_spec.get("dtype", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)

    inputs = torch.randn(num_samples, *shape, dtype=dtype)
    targets = torch.randn(num_samples, output_dim, dtype=torch.float32)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_imagefolder_dataloader(data_dir, input_spec, preprocessing=None, batch_size=16, is_train=True):
    """Create a DataLoader from an ImageFolder directory.

    Requires torchvision.
    """
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    transform_list = []
    target_shape = input_spec.get("shape", [3, 224, 224])
    if len(target_shape) >= 2:
        h, w = target_shape[1], target_shape[2]
        transform_list.append(transforms.Resize((h, w)))

    transform_list.append(transforms.ToTensor())

    if preprocessing:
        for step in preprocessing:
            name = step.get("name", "") if isinstance(step, dict) else ""
            params = step.get("params", {}) if isinstance(step, dict) else {}
            if name == "normalize":
                mean = params.get("mean", [0.485, 0.456, 0.406])
                std = params.get("std", [0.229, 0.224, 0.225])
                transform_list.append(transforms.Normalize(mean=mean, std=std))

    if is_train:
        transform_list.insert(1, transforms.RandomHorizontalFlip())

    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=2)


def create_example_input(model_contract, batch_size=2, device="cpu"):
    """Create example input tensors for tracing based on model_contract.

    Returns a tensor or tuple of tensors matching the model's forward signature.
    """
    input_spec = model_contract.get("input_spec", {})
    shape = input_spec.get("shape", [3, 224, 224])
    dtype_str = input_spec.get("dtype", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)
    architecture = model_contract.get("model_spec", {}).get("architecture", "")

    if architecture == "bert":
        max_seq_length = input_spec.get("max_seq_length", 128)
        input_ids = torch.randint(1, 1000, (batch_size, max_seq_length), dtype=torch.long)
        attention_mask = torch.ones(batch_size, max_seq_length, dtype=torch.long)
        return (input_ids.to(device), attention_mask.to(device))
    else:
        if dtype in (torch.int64, torch.int32, torch.long):
            tensor = torch.randint(0, 1000, (batch_size, *shape), dtype=dtype)
        else:
            tensor = torch.randn(batch_size, *shape, dtype=dtype)
        return tensor.to(device)


def _create_synthetic_dataloader_for_contract(model_contract, input_spec, num_classes,
                                                output_dim=1, num_samples=200, batch_size=16):
    """Create the appropriate synthetic DataLoader based on model architecture."""
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
