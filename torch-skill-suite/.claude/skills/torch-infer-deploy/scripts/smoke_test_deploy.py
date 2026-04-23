#!/usr/bin/env python3
"""
End-to-end smoke test for the deploy pipeline.

Builds a model from contract, exports it to TorchScript or ONNX, loads the
exported model, runs inference with synthetic data, and validates output shape.

Usage:
    # With a real checkpoint
    python smoke_test_deploy.py --model-contract model.yaml --checkpoint best_model.pt

    # With synthetic model (no checkpoint needed)
    python smoke_test_deploy.py --model-contract model.yaml --synthetic
"""

import argparse
import sys
import tempfile
from pathlib import Path

import torch


def _add_script_paths():
    """Add sibling scripts to sys.path for imports."""
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


def smoke_test_export(model_contract_path, checkpoint_path=None, output_dir=None,
                      format="torchscript", batch_size=2, device="auto"):
    """Run the full export → inference → validation smoke test.

    Args:
        model_contract_path: Path to model_contract.yaml.
        checkpoint_path: Optional path to checkpoint .pt file.
        output_dir: Directory for exported model (temp dir if None).
        format: Export format ('torchscript' or 'onnx').
        batch_size: Batch size for synthetic test data.
        device: Device to use.

    Returns:
        0 on success, 1 on failure.
    """
    _add_script_paths()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from export_model import (
        build_model_from_contract,
        create_example_input,
        _wrap_model_for_export,
        export_torchscript,
        _auto_select_torchscript_mode,
        validate_exported_model,
        _load_yaml,
    )
    from local_infer import load_exported_model, run_inference, apply_postprocessing

    own_model = checkpoint_path is None

    print(f"=" * 60)
    print(f"Smoke Test: {format.upper()} export")
    print(f"  Model contract: {model_contract_path}")
    print(f"  Checkpoint: {checkpoint_path or '(synthetic model)'}")
    print(f"  Device: {device}")
    print(f"=" * 60)

    # 1. Load contract and build model
    print("\n[1/5] Loading model contract...")
    model_contract = _load_yaml(model_contract_path)
    architecture = model_contract.get("model_spec", {}).get("architecture", "unknown")
    backbone = model_contract.get("model_spec", {}).get("backbone", "unknown")
    print(f"  Architecture: {architecture}/{backbone}")

    print("\n[2/5] Building model...")
    model = build_model_from_contract(model_contract)
    model.to(device)
    model.eval()

    if own_model:
        print("  (using randomly initialized weights)")
    else:
        print(f"  Loading checkpoint: {checkpoint_path}")
        from export_model import load_checkpoint
        load_checkpoint(model, checkpoint_path, device)

    # 2. Create example input and get reference output
    print("\n[3/5] Creating example input and getting reference output...")
    example_inputs = create_example_input(model_contract, batch_size=batch_size, device=device)
    wrapped = _wrap_model_for_export(model, model_contract)

    with torch.no_grad():
        if isinstance(example_inputs, tuple):
            reference_output = wrapped(*example_inputs)
        else:
            reference_output = wrapped(example_inputs)

    print(f"  Reference output shape: {list(reference_output.shape)}")

    # 3. Export model
    print(f"\n[4/5] Exporting to {format.upper()}...")
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="smoke_test_deploy_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "torchscript":
        output_path = output_dir / "smoke_model.torchscript.pt"
        mode = _auto_select_torchscript_mode(model_contract)
        exported = export_torchscript(wrapped, example_inputs, str(output_path), mode=mode)
        validate_exported_model(exported, example_inputs, reference_output)
    elif format == "onnx":
        output_path = output_dir / "smoke_model.onnx"
        from export_model import export_onnx
        export_onnx(wrapped, example_inputs, str(output_path))
    else:
        print(f"Error: Unsupported format '{format}'")
        return 1

    # 4. Load exported model and run inference
    print(f"\n[5/5] Loading exported model and running inference...")
    loaded_model = load_exported_model(str(output_path), device)
    output = run_inference(loaded_model, example_inputs, device)

    if isinstance(output, torch.Tensor):
        print(f"  Exported model output shape: {list(output.shape)}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # Validate shape matches reference
        assert output.shape == reference_output.shape, \
            f"Shape mismatch: exported {output.shape} vs reference {reference_output.shape}"
        print(f"  Shape validation: PASSED")

        # Validate no NaN/Inf
        assert not torch.isnan(output).any(), "Output contains NaN!"
        assert not torch.isinf(output).any(), "Output contains Inf!"
        print(f"  Value validation: PASSED")

    # 5. Test postprocessing
    print(f"\n  Running postprocessing smoke test...")
    postproc_config = {"type": "softmax_topk", "topk": 3}
    predictions = apply_postprocessing(output, postproc_config)
    print(f"  Postprocessing output: {predictions[:2]}...")

    print(f"\n{'=' * 60}")
    print(f"Smoke Test PASSED!")
    print(f"  Exported model: {output_path}")
    print(f"{'=' * 60}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test for model deployment.")
    parser.add_argument("--model-contract", required=True, help="Path to model_contract.yaml")
    parser.add_argument("--checkpoint", help="Path to checkpoint .pt file (optional; uses synthetic weights if omitted)")
    parser.add_argument("--output-dir", help="Output directory for exported model (temp dir if omitted)")
    parser.add_argument("--format", default="torchscript", choices=["torchscript", "onnx"],
                        help="Export format (default: torchscript)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for synthetic data (default: 2)")
    parser.add_argument("--device", default="auto", help="Device: 'auto', 'cpu', or 'cuda' (default: auto)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic model (no checkpoint needed; for testing export logic)")
    args = parser.parse_args()

    if not args.checkpoint and not args.synthetic:
        print("Error: either --checkpoint or --synthetic is required.")
        parser.print_help()
        sys.exit(1)

    exit_code = smoke_test_export(
        model_contract_path=args.model_contract,
        checkpoint_path=args.checkpoint if not args.synthetic else None,
        output_dir=args.output_dir,
        format=args.format,
        batch_size=args.batch_size,
        device=args.device,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
