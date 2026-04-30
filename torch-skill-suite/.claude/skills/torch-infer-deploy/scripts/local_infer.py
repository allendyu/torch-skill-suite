#!/usr/bin/env python3
"""
Run local inference with an exported TorchScript or ONNX model.

Loads an exported model (TorchScript .pt or ONNX .onnx), applies preprocessing
from a deploy_contract (or sensible defaults), runs inference, and applies
postprocessing. Supports single file, directory batch, and synthetic input.

Usage:
    # Inference with a single image
    python local_infer.py --model-path model.torchscript.pt --model-contract model.yaml --input image.jpg

    # Inference with synthetic data
    python local_infer.py --model-path model.torchscript.pt --model-contract model.yaml --synthetic

    # Batch inference on a directory
    python local_infer.py --model-path model.onnx --model-contract model.yaml --input /path/to/images/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

# Add shared package to path
_SHARED_PYTHON = Path(__file__).resolve().parent.parent.parent.parent.parent / "shared" / "python"
if str(_SHARED_PYTHON) not in sys.path:
    sys.path.insert(0, str(_SHARED_PYTHON))

from torch_skill_shared.yaml_utils import load_yaml
from torch_skill_shared.model_builder import create_example_input

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_exported_model(model_path, device="cpu"):
    """Load an exported TorchScript or ONNX model.

    Args:
        model_path: Path to model file (.pt/.torchscript.pt or .onnx).
        device: Device to load the model onto.

    Returns:
        For TorchScript: a torch.jit.ScriptModule.
        For ONNX: an onnxruntime.InferenceSession.
    """
    path = Path(model_path)
    if path.suffix == ".onnx":
        try:
            import onnxruntime as ort
        except ImportError:
            print("Error: onnxruntime is required to load ONNX models.")
            print("  Install with: pip install onnxruntime")
            sys.exit(1)
        session = ort.InferenceSession(str(path))
        return session
    else:
        try:
            model = torch.jit.load(str(path), map_location=device)
        except (RuntimeError, AssertionError):
            # Fall back to CPU if CUDA loading fails (e.g. broken driver)
            model = torch.jit.load(str(path), map_location="cpu")
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def create_preprocessing_pipeline(preprocessing_config, model_contract=None):
    """Build a torchvision transforms pipeline from config.

    Falls back to model_contract input_spec shape if resize not specified.
    Returns None if no transforms are needed (text, raw_tensor, or empty config).
    """
    if not preprocessing_config:
        return None

    pipeline = []

    input_format = preprocessing_config.get("input_format", "image_file")
    if input_format == "raw_tensor":
        return None
    if input_format != "image_file":
        # For text, numpy_array, etc. — no image transforms
        return None

    # Resize
    resize = preprocessing_config.get("resize")
    if resize:
        h = resize.get("height", 224)
        w = resize.get("width", 224)
        pipeline.append(transforms.Resize((h, w)))
    elif model_contract:
        shape = model_contract.get("input_spec", {}).get("shape", [3, 224, 224])
        if len(shape) >= 2:
            pipeline.append(transforms.Resize((shape[-2], shape[-1])))

    # ToTensor
    pipeline.append(transforms.ToTensor())

    # Normalize
    normalize = preprocessing_config.get("normalize")
    if normalize:
        mean = normalize.get("mean", [0.485, 0.456, 0.406])
        std = normalize.get("std", [0.229, 0.224, 0.225])
        pipeline.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(pipeline) if pipeline else None


def load_input(input_path, input_format="image_file"):
    """Load input data from a file path.

    Args:
        input_path: Path to input file or directory.
        input_format: Format hint ('image_file', 'numpy_array', 'text').

    Returns:
        A single input item or list of items (for directories).
    """
    path = Path(input_path)
    if path.is_dir():
        items = sorted(path.iterdir())
        results = []
        for item in items:
            if item.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                result = _load_single_file(item, input_format)
                if result is not None:
                    results.append((item.name, result))
        return results
    else:
        result = _load_single_file(path, input_format)
        return result if result is not None else None


def _load_single_file(path, input_format):
    path = Path(path)
    if input_format == "image_file":
        if Image is None:
            print("Error: PIL (Pillow) is required to load images.")
            print("  Install with: pip install Pillow")
            return None
        return Image.open(path).convert("RGB")
    elif input_format == "numpy_array":
        import numpy as np
        return np.load(str(path))
    elif input_format == "text":
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    else:
        return path.read_bytes()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, input_tensor, device="cpu"):
    """Run model inference, handling different model types.

    Args:
        model: TorchScript module or ONNX InferenceSession.
        input_tensor: Preprocessed input tensor(s).

    Returns:
        Raw model output tensor.
    """
    # Check if ONNX session
    if hasattr(model, "run"):
        input_name = model.get_inputs()[0].name
        if isinstance(input_tensor, torch.Tensor):
            import numpy as np
            input_np = input_tensor.cpu().numpy()
        else:
            input_np = input_tensor
        output = model.run(None, {input_name: input_np})
        return torch.tensor(output[0])
    else:
        with torch.no_grad():
            if isinstance(input_tensor, tuple):
                output = model(*input_tensor)
            else:
                output = model(input_tensor)

        # Handle dict output (e.g., DeepLabV3)
        if isinstance(output, dict):
            output = output.get("out", list(output.values())[0])
        return output


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def apply_postprocessing(output_tensor, postprocessing_config):
    """Apply postprocessing to model output.

    Handles both 2D (classification) and 4D (segmentation) tensors.
    For segmentation, argmax is applied along the channel dimension (dim=1)
    to produce per-pixel class predictions.

    Args:
        output_tensor: Raw model output tensor.
        postprocessing_config: Dict with 'type' and optional 'topk'.

    Returns:
        Postprocessed predictions.
    """
    post_type = postprocessing_config.get("type", "softmax_topk")
    topk = postprocessing_config.get("topk", 5)

    if output_tensor.dim() >= 3:
        # Segmentation output: apply argmax along channel dim
        indices = torch.argmax(output_tensor, dim=1)
        return indices.tolist()

    if post_type == "softmax_topk":
        probs = F.softmax(output_tensor, dim=-1)
        values, indices = torch.topk(probs, min(topk, probs.size(-1)), dim=-1)
        return format_predictions(indices, values, probs)
    elif post_type == "argmax":
        indices = torch.argmax(output_tensor, dim=-1)
        return format_predictions(indices, None, None)
    elif post_type == "sigmoid":
        probs = torch.sigmoid(output_tensor)
        return probs.tolist()
    elif post_type == "none":
        return output_tensor.tolist()
    else:
        return output_tensor.tolist()


def format_predictions(indices, values=None, probs=None):
    """Format predictions into a list of dicts per sample.

    Args:
        indices: Top-k indices tensor of shape (batch, k) or (batch,).
        values: Top-k values tensor of shape (batch, k) or None.
        probs: Full probability tensor of shape (batch, num_classes) or None.

    Returns:
        List of predictions per sample.
    """
    results = []
    if indices.dim() == 1:
        # argmax mode: single index per sample
        for i, idx in enumerate(indices.tolist()):
            results.append({"class": idx})
    else:
        # top-k mode
        indices_list = indices.tolist()
        if values is not None:
            values_list = values.tolist()
            for i in range(len(indices_list)):
                topk = []
                for k in range(len(indices_list[i])):
                    topk.append({"class": indices_list[i][k], "score": round(values_list[i][k], 4)})
                results.append({"topk": topk})
        else:
            for i in range(len(indices_list)):
                topk = [{"class": idx} for idx in indices_list[i]]
                results.append({"topk": topk})
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def infer(model_path, model_contract_path, deploy_contract_path=None,
          input_path=None, synthetic=False, batch_size=1, device="auto"):
    """Run the full inference pipeline."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model contract: {model_contract_path}")
    model_contract = load_yaml(model_contract_path)

    deploy_contract = {}
    if deploy_contract_path:
        print(f"Loading deploy contract: {deploy_contract_path}")
        deploy_contract = load_yaml(deploy_contract_path)

    print(f"Loading exported model: {model_path}")
    model = load_exported_model(model_path, device)

    # Build preprocessing pipeline
    preprocessing_config = deploy_contract.get("preprocessing", {})
    pipeline = create_preprocessing_pipeline(preprocessing_config, model_contract)

    # Get input
    if synthetic:
        print(f"Using synthetic input (batch_size={batch_size})")
        input_tensor = create_example_input(model_contract, batch_size=batch_size, device=device)
        output = run_inference(model, input_tensor, device)
    elif input_path:
        print(f"Loading input: {input_path}")
        loaded = load_input(input_path, preprocessing_config.get("input_format", "image_file"))

        if isinstance(loaded, list):
            # Batch of files
            all_results = []
            for name, data in loaded:
                if pipeline:
                    tensor = pipeline(data).unsqueeze(0).to(device)
                else:
                    tensor = data if isinstance(data, torch.Tensor) else torch.tensor(data)
                    if tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    tensor = tensor.to(device)
                output = run_inference(model, tensor, device)
                postproc_config = deploy_contract.get("postprocessing", {"type": "softmax_topk", "topk": 5})
                preds = apply_postprocessing(output, postproc_config)
                all_results.append({"file": name, "predictions": preds})
            return all_results
        else:
            if pipeline:
                tensor = pipeline(loaded).unsqueeze(0).to(device)
            else:
                tensor = loaded if isinstance(loaded, torch.Tensor) else torch.tensor(loaded)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                tensor = tensor.to(device)
            output = run_inference(model, tensor, device)
    else:
        print("No input provided. Use --input or --synthetic.")
        return None

    postproc_config = deploy_contract.get("postprocessing", {"type": "softmax_topk", "topk": 5})
    predictions = apply_postprocessing(output, postproc_config)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run local inference with an exported model.")
    parser.add_argument("--model-path", required=True, help="Path to exported model (.pt or .onnx)")
    parser.add_argument("--model-contract", required=True, help="Path to model_contract.yaml")
    parser.add_argument("--deploy-contract", help="Path to deploy_contract.yaml (optional)")
    parser.add_argument("--input", help="Path to input file or directory")
    parser.add_argument("--output", help="Path to save predictions (stdout if omitted)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic input")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for synthetic data (default: 1)")
    parser.add_argument("--device", default="auto", help="Device: 'auto', 'cpu', or 'cuda' (default: auto)")
    args = parser.parse_args()

    if not args.synthetic and not args.input:
        parser.print_help()
        print("\nError: either --input or --synthetic is required.")
        sys.exit(1)

    results = infer(
        model_path=args.model_path,
        model_contract_path=args.model_contract,
        deploy_contract_path=args.deploy_contract,
        input_path=args.input,
        synthetic=args.synthetic,
        batch_size=args.batch_size,
        device=args.device,
    )

    if results is not None:
        output = json.dumps(results, indent=2, ensure_ascii=False)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(output)
            print(f"Predictions saved to: {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
