#!/usr/bin/env python3
"""
Resolve a model contract from a data contract and optional project spec.

Usage:
    python resolve_model.py --data-contract path/to/data_contract.yaml
    python resolve_model.py --data-contract path/to/data_contract.yaml --project-spec path/to/project_spec.yaml
    python resolve_model.py --data-contract path/to/data_contract.yaml --output path/to/model_contract.yaml

Currently only supports the P0 route: image + classification.
"""

import argparse
import sys
from pathlib import Path

# Add shared package to path
_SHARED_PYTHON = Path(__file__).resolve().parent.parent.parent.parent.parent / "shared" / "python"
if str(_SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SHARED_PYTHON))

from torch_skill_shared.yaml_utils import load_yaml, emit_yaml


# ---- Backbone selection rules for P0 image_classification ----

RESNET_FEATURE_DIMS = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
}

EFFICIENTNET_FEATURE_DIMS = {
    "efficientnet_b0": 1280,
}

MODEL_SIZE_MB = {
    "resnet18": 45,
    "resnet34": 85,
    "resnet50": 98,
    "efficientnet_b0": 21,
}


def select_backbone(data_contract, project_spec=None):
    """Select the best backbone for image classification based on constraints.

    Args:
        data_contract: Parsed data_contract dict.
        project_spec: Optional project_spec dict with constraints.

    Returns:
        Tuple of (backbone_name, architecture, feature_dim, model_size_mb, latency_tier).
    """
    constraints = {}
    if project_spec and isinstance(project_spec, dict):
        constraints = project_spec.get("constraints", {})

    latency_ms = constraints.get("latency_ms")
    model_size_limit = constraints.get("model_size_mb")

    # Rule 1: Low latency or small model → efficientnet_b0 or resnet18
    if (latency_ms is not None and latency_ms <= 50) or (model_size_limit is not None and model_size_limit <= 50):
        if model_size_limit is not None and model_size_limit <= 30:
            return ("efficientnet_b0", "efficientnet", 1280, 21, "fast")
        return ("resnet18", "resnet", 512, 45, "fast")

    # Rule 2: Small model but not extreme → resnet18
    if model_size_limit is not None and model_size_limit <= 100:
        return ("resnet18", "resnet", 512, 45, "fast")

    # Rule 3: Accuracy priority (large model allowed) → resnet50
    if model_size_limit is not None and model_size_limit >= 200:
        return ("resnet50", "resnet", 2048, 98, "accurate")

    # Rule 4: Default balanced → resnet34
    return ("resnet34", "resnet", 512, 85, "balanced")


def select_tabular_backbone(data_contract, project_spec=None):
    """Select backbone for tabular data. Currently only MLP is supported.

    Returns:
        Tuple of (backbone_name, architecture, feature_dim, model_size_mb, latency_tier).
    """
    return ("mlp", "mlp", None, 5, "fast")


def select_text_backbone(data_contract, project_spec=None):
    """Select the best transformer backbone for text classification.

    Returns:
        Tuple of (backbone_name, architecture, hidden_size, model_size_mb, latency_tier).
    """
    constraints = {}
    if project_spec and isinstance(project_spec, dict):
        constraints = project_spec.get("constraints", {})

    latency_ms = constraints.get("latency_ms")
    model_size_limit = constraints.get("model_size_mb")

    # Lightweight: low latency or small model
    if (latency_ms is not None and latency_ms <= 30) or (model_size_limit is not None and model_size_limit <= 100):
        return ("distilbert-base-uncased", "bert", 768, 260, "fast")

    # Default: balanced
    return ("bert-base-uncased", "bert", 768, 440, "balanced")


def select_segmentation_backbone(data_contract, project_spec=None):
    """Select the best backbone for image segmentation.

    Returns:
        Tuple of (backbone_name, architecture, model_size_mb, latency_tier).
    """
    constraints = {}
    if project_spec and isinstance(project_spec, dict):
        constraints = project_spec.get("constraints", {})

    model_size_limit = constraints.get("model_size_mb")

    # Lightweight
    if model_size_limit is not None and model_size_limit <= 50:
        return ("deeplabv3_mobilenet_v3_large", "deeplabv3", 30, "fast")

    # Default
    return ("deeplabv3_resnet50", "deeplabv3", 200, "balanced")


def _build_image_classification_contract(data_contract, project_spec):
    """Build model_contract for image_classification route."""
    input_spec = data_contract.get("input_spec", {})
    output_spec = data_contract.get("output_spec", {})

    shape = input_spec.get("shape", [3, 224, 224])
    dtype = input_spec.get("dtype", "float32")
    channels_first = input_spec.get("channels_first", True)
    num_classes = output_spec.get("num_classes")
    if num_classes is None:
        raise ValueError("output_spec.num_classes is required for classification")

    backbone, architecture, feature_dim, model_size, latency_tier = select_backbone(
        data_contract, project_spec
    )

    return {
        "task_type": "classification",
        "data_type": "image",
        "input_spec": {"shape": shape, "dtype": dtype, "channels_first": channels_first},
        "model_spec": {
            "family": "cnn", "architecture": architecture, "backbone": backbone,
            "pretrained": True,
            "in_channels": shape[0] if len(shape) >= 1 else 3,
            "feature_dim": feature_dim,
        },
        "head_spec": {"type": "linear_cls", "num_classes": num_classes, "pooling": "avg", "dropout": 0.0},
        "forward_spec": {"input_tensor_name": "images", "output_tensor_name": "logits", "output_shape": ["batch", num_classes]},
        "compatibility": {"expected_target_type": "categorical", "expected_loss": "cross_entropy", "target_dtype": "int64", "output_activation": "none"},
        "constraints": {"latency_tier": latency_tier, "model_size_mb": model_size},
        "artifacts": {"template_name": f"image_classification/{architecture}", "smoke_test_required": True},
        "metadata": {"route": "image_classification", "priority": "P0", "resolver_version": "0.2.0"},
    }


def _build_tabular_classification_contract(data_contract, project_spec):
    """Build model_contract for tabular_classification route."""
    input_spec = data_contract.get("input_spec", {})
    output_spec = data_contract.get("output_spec", {})

    shape = input_spec.get("shape", [10])
    dtype = input_spec.get("dtype", "float32")
    num_features = shape[0] if len(shape) >= 1 else 10
    num_classes = output_spec.get("num_classes")
    if num_classes is None:
        raise ValueError("output_spec.num_classes is required for classification")

    backbone, architecture, feature_dim, model_size, latency_tier = select_tabular_backbone(
        data_contract, project_spec
    )

    return {
        "task_type": "classification",
        "data_type": "tabular",
        "input_spec": {"shape": shape, "dtype": dtype},
        "model_spec": {
            "family": "mlp", "architecture": architecture, "backbone": backbone,
            "pretrained": False, "in_features": num_features, "feature_dim": feature_dim,
        },
        "head_spec": {"type": "linear_cls", "num_classes": num_classes, "dropout": 0.2},
        "forward_spec": {"input_tensor_name": "features", "output_tensor_name": "logits", "output_shape": ["batch", num_classes]},
        "compatibility": {"expected_target_type": "categorical", "expected_loss": "cross_entropy", "target_dtype": "int64", "output_activation": "none"},
        "constraints": {"latency_tier": latency_tier, "model_size_mb": model_size},
        "artifacts": {"template_name": f"tabular_classification/{architecture}", "smoke_test_required": True},
        "metadata": {"route": "tabular_classification", "priority": "P1", "resolver_version": "0.2.0"},
    }


def _build_tabular_regression_contract(data_contract, project_spec):
    """Build model_contract for tabular_regression route."""
    input_spec = data_contract.get("input_spec", {})
    output_spec = data_contract.get("output_spec", {})

    shape = input_spec.get("shape", [10])
    dtype = input_spec.get("dtype", "float32")
    num_features = shape[0] if len(shape) >= 1 else 10
    output_dim = output_spec.get("output_dim", 1)

    backbone, architecture, feature_dim, model_size, latency_tier = select_tabular_backbone(
        data_contract, project_spec
    )

    return {
        "task_type": "regression",
        "data_type": "tabular",
        "input_spec": {"shape": shape, "dtype": dtype},
        "model_spec": {
            "family": "mlp", "architecture": architecture, "backbone": backbone,
            "pretrained": False, "in_features": num_features, "feature_dim": feature_dim,
        },
        "head_spec": {"type": "linear_regression", "output_dim": output_dim, "dropout": 0.2},
        "forward_spec": {"input_tensor_name": "features", "output_tensor_name": "predictions", "output_shape": ["batch", output_dim]},
        "compatibility": {"expected_target_type": "continuous", "expected_loss": "mse", "target_dtype": "float32", "output_activation": "none"},
        "constraints": {"latency_tier": latency_tier, "model_size_mb": model_size},
        "artifacts": {"template_name": f"tabular_regression/{architecture}", "smoke_test_required": True},
        "metadata": {"route": "tabular_regression", "priority": "P1", "resolver_version": "0.2.0"},
    }


def _build_text_classification_contract(data_contract, project_spec):
    """Build model_contract for text_classification route."""
    input_spec = data_contract.get("input_spec", {})
    output_spec = data_contract.get("output_spec", {})

    max_seq_length = input_spec.get("max_seq_length", 512)
    dtype = input_spec.get("dtype", "int64")
    num_classes = output_spec.get("num_classes")
    if num_classes is None:
        raise ValueError("output_spec.num_classes is required for classification")

    backbone, architecture, hidden_size, model_size, latency_tier = select_text_backbone(
        data_contract, project_spec
    )

    return {
        "task_type": "classification",
        "data_type": "text",
        "input_spec": {"shape": [max_seq_length], "dtype": dtype, "max_seq_length": max_seq_length},
        "model_spec": {
            "family": "transformer_encoder", "architecture": architecture, "backbone": backbone,
            "pretrained": True, "hidden_size": hidden_size,
        },
        "head_spec": {"type": "pooled_linear_cls", "num_classes": num_classes, "pooling": "cls_token", "dropout": 0.1},
        "forward_spec": {"input_tensor_name": "input_ids", "output_tensor_name": "logits", "output_shape": ["batch", num_classes]},
        "compatibility": {"expected_target_type": "categorical", "expected_loss": "cross_entropy", "target_dtype": "int64", "output_activation": "none"},
        "constraints": {"latency_tier": latency_tier, "model_size_mb": model_size},
        "artifacts": {"template_name": f"text_classification/{architecture}", "smoke_test_required": True},
        "metadata": {"route": "text_classification", "priority": "P1", "resolver_version": "0.3.0"},
    }


def _build_image_segmentation_contract(data_contract, project_spec):
    """Build model_contract for image_segmentation route."""
    input_spec = data_contract.get("input_spec", {})
    output_spec = data_contract.get("output_spec", {})

    shape = input_spec.get("shape", [3, 224, 224])
    dtype = input_spec.get("dtype", "float32")
    channels_first = input_spec.get("channels_first", True)
    num_classes = output_spec.get("num_classes")
    if num_classes is None:
        raise ValueError("output_spec.num_classes is required for segmentation")

    backbone, architecture, model_size, latency_tier = select_segmentation_backbone(
        data_contract, project_spec
    )

    return {
        "task_type": "segmentation",
        "data_type": "image",
        "input_spec": {"shape": shape, "dtype": dtype, "channels_first": channels_first},
        "model_spec": {
            "family": "cnn_encoder_decoder", "architecture": architecture, "backbone": backbone,
            "pretrained": True,
            "in_channels": shape[0] if len(shape) >= 1 else 3,
        },
        "head_spec": {"type": "segmentation_head", "num_classes": num_classes},
        "forward_spec": {"input_tensor_name": "images", "output_tensor_name": "masks", "output_shape": ["batch", num_classes, "H", "W"]},
        "compatibility": {"expected_target_type": "mask", "expected_loss": "cross_entropy", "target_dtype": "int64", "output_activation": "none"},
        "constraints": {"latency_tier": latency_tier, "model_size_mb": model_size},
        "artifacts": {"template_name": f"image_segmentation/{architecture}", "smoke_test_required": True},
        "metadata": {"route": "image_segmentation", "priority": "P1", "resolver_version": "0.3.0"},
    }


def resolve(data_contract, project_spec=None):
    """Resolve a model contract from data contract.

    Args:
        data_contract: Parsed data_contract dict.
        project_spec: Optional project_spec dict.

    Returns:
        Dict representing the model_contract.

    Raises:
        ValueError: If the data_contract cannot be resolved (unsupported route).
    """
    data_type = data_contract.get("data_type", "")
    task_type = data_contract.get("task_type", "")

    # ---- Route matching ----
    if data_type == "image" and task_type == "classification":
        return _build_image_classification_contract(data_contract, project_spec)
    elif data_type == "image" and task_type == "segmentation":
        return _build_image_segmentation_contract(data_contract, project_spec)
    elif data_type == "text" and task_type == "classification":
        return _build_text_classification_contract(data_contract, project_spec)
    elif data_type == "tabular" and task_type == "classification":
        return _build_tabular_classification_contract(data_contract, project_spec)
    elif data_type == "tabular" and task_type == "regression":
        return _build_tabular_regression_contract(data_contract, project_spec)
    else:
        raise ValueError(
            f"Unsupported route: data_type='{data_type}', task_type='{task_type}'. "
            f"Supported routes: image+classification, image+segmentation, text+classification, tabular+classification, tabular+regression."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Resolve model contract from data contract."
    )
    parser.add_argument("--data-contract", required=True, help="Path to data_contract.yaml")
    parser.add_argument("--project-spec", help="Path to project_spec.yaml (optional)")
    parser.add_argument("--output", "-o", help="Output path for model_contract.yaml (default: stdout)")
    args = parser.parse_args()

    data_contract = load_yaml(args.data_contract)
    project_spec = None
    if args.project_spec:
        project_spec = load_yaml(args.project_spec)

    try:
        model_contract = resolve(data_contract, project_spec)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    yaml_output = emit_yaml(model_contract)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(f"# Model contract generated by torch-model resolver (P0)\n{yaml_output}\n")
        print(f"Model contract written to {args.output}")
    else:
        print(f"# Model contract generated by torch-model resolver (P0)\n{yaml_output}")


if __name__ == "__main__":
    main()
