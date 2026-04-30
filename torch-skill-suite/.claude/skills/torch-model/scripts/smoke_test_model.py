#!/usr/bin/env python3
"""
Smoke test a model contract by building the model and running a dummy forward pass.

Usage:
    python smoke_test_model.py --model-contract path/to/model_contract.yaml
    python smoke_test_model.py --model-contract path/to/model_contract.yaml --batch-size 4

The script:
    1. Reads the model_contract
    2. Builds the model via templates
    3. Creates a dummy input based on input_spec
    4. Runs forward pass
    5. Checks output shape matches forward_spec.output_shape
    6. Reports parameter count
"""

import argparse
import sys
from pathlib import Path

import torch

# Add shared package to path
_SHARED_PYTHON = Path(__file__).resolve().parent.parent.parent.parent.parent / "shared" / "python"
if str(_SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SHARED_PYTHON))

from torch_skill_shared.yaml_utils import load_yaml


def _add_template_path():
    """Add the templates directory to sys.path so imports work."""
    script_dir = Path(__file__).resolve().parent
    templates_dir = script_dir / ".." / "templates"
    templates_path = str(templates_dir.resolve())
    if templates_path not in sys.path:
        sys.path.insert(0, templates_path)
    # Also add the parent so 'import templates' works
    parent_path = str((script_dir / "..").resolve())
    if parent_path not in sys.path:
        sys.path.insert(0, parent_path)


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def create_dummy_input(input_spec, batch_size=2):
    """Create a dummy input tensor from input_spec.

    Args:
        input_spec: Dict with 'shape' and 'dtype'.
        batch_size: Batch size to prepend.

    Returns:
        torch.Tensor with shape (batch_size, *shape).
    """
    shape = input_spec.get("shape", [3, 224, 224])
    dtype_str = input_spec.get("dtype", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)

    full_shape = [batch_size] + list(shape)
    if dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
        return torch.randint(0, 255, full_shape, dtype=dtype)
    else:
        return torch.randn(full_shape, dtype=dtype)


def build_model_from_contract(model_contract):
    """Build a PyTorch model from a model_contract dict.

    Args:
        model_contract: Parsed model_contract.

    Returns:
        nn.Module instance.
    """
    architecture = model_contract["model_spec"]["architecture"]
    backbone = model_contract["model_spec"]["backbone"]
    head_spec = model_contract.get("head_spec", {})

    config = {
        "architecture": architecture,
        "backbone": backbone,
        "pretrained": model_contract["model_spec"].get("pretrained", True),
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
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def check_output_shape(output, expected_shape, batch_size):
    """Check if actual output shape matches expected.

    Args:
        output: Model output tensor.
        expected_shape: List from forward_spec.output_shape (e.g. ['batch', 10]).
        batch_size: Actual batch size used.

    Returns:
        Tuple of (passed, message).
    """
    expected = []
    for dim in expected_shape:
        if dim == "batch":
            expected.append(batch_size)
        elif isinstance(dim, int):
            expected.append(dim)
        else:
            expected.append(dim)

    actual = list(output.shape)
    if len(actual) != len(expected):
        return False, f"Rank mismatch: expected {expected}, got {actual}"

    for i, (exp, act) in enumerate(zip(expected, actual)):
        if isinstance(exp, int) and exp != act:
            return False, f"Dimension {i} mismatch: expected {expected}, got {actual}"

    return True, f"Output shape {actual} matches expected {expected}"


def smoke_test(model_contract, batch_size=2):
    """Run a smoke test on a model contract.

    Args:
        model_contract: Parsed model_contract dict.
        batch_size: Batch size for dummy input.

    Returns:
        True if all checks pass, False otherwise.
    """
    _add_template_path()

    print(f"Model: {model_contract['model_spec']['backbone']}")
    print(f"Task:  {model_contract['task_type']} / {model_contract['data_type']}")
    print(f"Input: {model_contract['input_spec']}")
    print(f"Head:  {model_contract['head_spec']}")
    print()

    # Build model
    print("Building model...")
    try:
        model = build_model_from_contract(model_contract)
        print("  OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # Count parameters
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    print()

    # Create dummy input
    input_spec = model_contract["input_spec"]
    dummy = create_dummy_input(input_spec, batch_size=batch_size)
    print(f"Dummy input shape: {list(dummy.shape)}, dtype: {dummy.dtype}")

    # Forward pass
    print("Running forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        print(f"  Output shape: {list(output.shape)}")
    except Exception as e:
        print(f"  FORWARD FAILED: {e}")
        return False

    # Check output shape
    forward_spec = model_contract.get("forward_spec", {})
    expected_shape = forward_spec.get("output_shape", ["batch", model_contract["head_spec"].get("num_classes", 1000)])
    passed, msg = check_output_shape(output, expected_shape, batch_size)
    if passed:
        print(f"  Shape check: PASSED — {msg}")
    else:
        print(f"  Shape check: FAILED — {msg}")
        return False

    print()
    print("Smoke test PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test a model contract.")
    parser.add_argument("--model-contract", required=True, help="Path to model_contract.yaml")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for dummy input (default: 2)")
    args = parser.parse_args()

    model_contract = load_yaml(args.model_contract)
    success = smoke_test(model_contract, batch_size=args.batch_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
