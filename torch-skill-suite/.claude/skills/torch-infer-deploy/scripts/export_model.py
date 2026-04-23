#!/usr/bin/env python3
"""
Export a trained PyTorch model to TorchScript or ONNX.

Loads a model from model_contract.yaml and a checkpoint .pt file, then exports
to the specified format. Supports image classification (ResNet, EfficientNet),
image segmentation (DeepLabV3, UNet), tabular (MLP), and text (BERT) models.

Usage:
    # Export to TorchScript (default)
    python export_model.py --model-contract model.yaml --checkpoint best_model.pt --output-dir ./exported

    # Export to ONNX
    python export_model.py --model-contract model.yaml --checkpoint best_model.pt --format onnx
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

try:
    import yaml
except ImportError:
    yaml = None


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

class SimpleYAMLParser:
    def __init__(self, text):
        self.lines = self._prepare_lines(text)

    def _strip_inline_comment(self, line):
        in_single, in_double = False, False
        for ch in line:
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if ch == "#" and not in_single and not in_double:
                idx = line.index(ch)
                if idx == 0 or line[idx - 1].isspace():
                    return line[:idx].rstrip()
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
        if ":" in content:
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


def _emit_yaml(data):
    if yaml is not None:
        return yaml.safe_dump(data, default_flow_style=False, allow_unicode=True)
    lines = []
    def _emit(obj, indent=0):
        prefix = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{prefix}{k}:")
                    _emit(v, indent + 1)
                else:
                    lines.append(f"{prefix}{k}: {_yaml_value(v)}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}-")
                    _emit(item, indent + 1)
                else:
                    lines.append(f"{prefix}- {_yaml_value(item)}")
    _emit(data)
    return "\n".join(lines) + "\n"


def _yaml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, str):
        if any(c in v for c in ": #{}[]&*!|>'\"%@`"):
            return f"'{v}'"
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Model building (reuse from torch-train via cross-skill import)
# ---------------------------------------------------------------------------

def _add_template_path():
    script_dir = Path(__file__).resolve().parent
    train_scripts_dir = script_dir / ".." / ".." / "torch-train" / "scripts"
    model_skill_dir = script_dir / ".." / ".." / "torch-model"
    for d in [train_scripts_dir, model_skill_dir]:
        resolved = d.resolve()
        if resolved.exists() and str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))


def build_model_from_contract(model_contract):
    """Build model from contract, delegating to torch-train's builder."""
    _add_template_path()
    from train import build_model_from_contract as _build
    return _build(model_contract)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from a checkpoint file.

    Uses strict=False to handle head dimension mismatches gracefully.
    Returns (epoch, best_loss) metadata from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: {len(missing)} missing key(s) in checkpoint: {missing}")
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected key(s) in checkpoint: {unexpected}")
    model.to(device)
    model.eval()
    epoch = checkpoint.get("epoch", 0) if "model_state_dict" in checkpoint else 0
    best_loss = checkpoint.get("best_loss", float("inf")) if "model_state_dict" in checkpoint else float("inf")
    return epoch, best_loss


# ---------------------------------------------------------------------------
# Example input creation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model wrapping for export (handles architecture-specific forward signatures)
# ---------------------------------------------------------------------------

def _wrap_model_for_export(model, model_contract):
    """Wrap model to produce a single-tensor output for export.

    DeepLabV3 returns dict{"out": tensor, ...} which needs unwrapping.
    Other architectures return a single tensor directly.
    """
    architecture = model_contract.get("model_spec", {}).get("architecture", "")

    if architecture == "deeplabv3":
        class _ExtractOut(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def forward(self, x):
                return self.wrapped(x)["out"]
        return _ExtractOut(model)

    return model


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------

def export_torchscript(model, example_inputs, output_path, mode="trace"):
    """Export model to TorchScript via tracing or scripting.

    Args:
        model: PyTorch model (will be set to eval mode).
        example_inputs: Tensor or tuple of tensors for tracing.
        output_path: Path to save the .pt file.
        mode: 'trace' for torch.jit.trace, 'script' for torch.jit.script,
              'trace_bert' for tracing with relaxed checks (transformers).

    Returns:
        The traced/scripted model.
    """
    model.eval()
    if mode == "script":
        traced_model = torch.jit.script(model)
    elif mode == "trace_bert":
        traced_model = torch.jit.trace(model, example_inputs, check_trace=False, strict=False)
    else:
        traced_model = torch.jit.trace(model, example_inputs)
    torch.jit.save(traced_model, output_path)
    print(f"TorchScript model saved to: {output_path}")
    return traced_model


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model, example_inputs, output_path,
                input_names=None, output_names=None,
                dynamic_axes=None, opset_version=17):
    """Export model to ONNX format.

    Args:
        model: PyTorch model (will be set to eval mode).
        example_inputs: Tensor or tuple of tensors.
        output_path: Path to save the .onnx file.
        input_names: List of input tensor names.
        output_names: List of output tensor names.
        dynamic_axes: Dict mapping tensor names to axis mappings.
        opset_version: ONNX opset version.
    """
    model.eval()
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Handle tuple input for ONNX export
    if isinstance(example_inputs, tuple):
        input_names = [f"input_{i}" for i in range(len(example_inputs))]
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}
        dynamic_axes["output"] = {0: "batch_size"}

    torch.onnx.export(
        model,
        example_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True,
    )
    print(f"ONNX model saved to: {output_path}")

    # Validate with onnxruntime if available
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        print(f"ONNX validation passed: {len(session.get_inputs())} input(s), {len(session.get_outputs())} output(s)")
    except ImportError:
        print("Note: onnxruntime not available; skipping ONNX runtime validation.")
        print("  Install with: pip install onnxruntime")


# ---------------------------------------------------------------------------
# Export validation
# ---------------------------------------------------------------------------

def validate_exported_model(exported_model, example_inputs, original_output=None):
    """Validate that the exported model runs and produces reasonable output."""
    with torch.no_grad():
        if isinstance(example_inputs, tuple):
            exported_output = exported_model(*example_inputs)
        else:
            exported_output = exported_model(example_inputs)

    if isinstance(exported_output, torch.Tensor):
        print(f"Exported model output shape: {list(exported_output.shape)}")
        print(f"Exported model output range: [{exported_output.min().item():.4f}, {exported_output.max().item():.4f}]")
        if exported_output.numel() > 0:
            assert not torch.isnan(exported_output).any(), "Output contains NaN!"
            assert not torch.isinf(exported_output).any(), "Output contains Inf!"
    return True


# ---------------------------------------------------------------------------
# Main export pipeline
# ---------------------------------------------------------------------------

def _auto_select_torchscript_mode(model_contract):
    """Auto-select TorchScript mode based on model architecture.

    Transformer models (bert) use dynamic control flow and **kwargs that
    prevent scripting, so we use tracing with relaxed checks.
    """
    architecture = model_contract.get("model_spec", {}).get("architecture", "")
    return "trace_bert" if architecture == "bert" else "trace"


def export(model_contract_path, checkpoint_path, output_dir, format="torchscript",
           opset_version=17, batch_size=2, device="auto"):
    """Full export pipeline: build, load, wrap, export, validate.

    Returns the output directory path.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model contract: {model_contract_path}")
    model_contract = _load_yaml(model_contract_path)

    print(f"Building model for architecture: {model_contract.get('model_spec', {}).get('architecture', 'unknown')}")
    model = build_model_from_contract(model_contract)

    print(f"Loading checkpoint: {checkpoint_path}")
    epoch, best_loss = load_checkpoint(model, checkpoint_path, device)

    print(f"Creating example input (batch_size={batch_size})")
    example_inputs = create_example_input(model_contract, batch_size=batch_size, device=device)

    wrapped_model = _wrap_model_for_export(model, model_contract)

    # Get original output for comparison
    with torch.no_grad():
        if isinstance(example_inputs, tuple):
            original_output = wrapped_model(*example_inputs)
        else:
            original_output = wrapped_model(example_inputs)

    if format == "torchscript":
        output_path = output_dir / "model.torchscript.pt"
        mode = model_contract.get("deploy_config", {}).get("torchscript_mode") or _auto_select_torchscript_mode(model_contract)
        exported = export_torchscript(wrapped_model, example_inputs, str(output_path), mode=mode)
        validate_exported_model(exported, example_inputs, original_output)
        export_format = "torchscript"
    elif format == "onnx":
        output_path = output_dir / "model.onnx"
        export_onnx(wrapped_model, example_inputs, str(output_path), opset_version=opset_version)
        export_format = "onnx"
    else:
        raise ValueError(f"Unsupported export format: {format}")

    # Write export report
    report = {
        "export": {
            "format": export_format,
            "output_path": str(output_path),
            "opset_version": opset_version if format == "onnx" else None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "model": {
            "architecture": model_contract.get("model_spec", {}).get("architecture"),
            "backbone": model_contract.get("model_spec", {}).get("backbone"),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "epoch": epoch,
            "best_loss": best_loss,
        },
        "validation": {
            "output_shape": list(original_output.shape) if isinstance(original_output, torch.Tensor) else "multi_output",
        },
    }
    report_path = output_dir / "export_report.yaml"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(_emit_yaml(report))
    print(f"Export report saved to: {report_path}")

    print(f"\nExport completed successfully!")
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Export trained PyTorch model to TorchScript or ONNX.")
    parser.add_argument("--model-contract", required=True, help="Path to model_contract.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--output-dir", default="./exported", help="Output directory for exported model")
    parser.add_argument("--format", default="torchscript", choices=["torchscript", "onnx"],
                        help="Export format (default: torchscript)")
    parser.add_argument("--opset-version", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--batch-size", type=int, default=2, help="Example batch size for tracing (default: 2)")
    parser.add_argument("--device", default="auto", help="Device to use: 'auto', 'cpu', or 'cuda' (default: auto)")
    args = parser.parse_args()

    export(
        model_contract_path=args.model_contract,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        format=args.format,
        opset_version=args.opset_version,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
