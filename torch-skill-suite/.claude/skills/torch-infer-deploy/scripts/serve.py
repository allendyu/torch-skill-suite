#!/usr/bin/env python3
"""
Serve an exported PyTorch model via FastAPI REST API.

Loads a TorchScript or ONNX model at startup and exposes:
  - GET  /health  — Service health check
  - POST /predict — Run inference on uploaded image or JSON tensor

Usage:
    python serve.py --model-path ./exported/model.torchscript.pt --model-contract model.yaml
    python serve.py --model-path ./exported/model.torchscript.pt --model-contract model.yaml --port 8080 --reload
"""

import argparse
import io
import json
import sys
from pathlib import Path

import torch

try:
    import yaml
except ImportError:
    yaml = None

# FastAPI is an optional dependency
try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    import uvicorn
except ImportError:
    print("Error: fastapi and uvicorn are required for serve mode.")
    print("  Install with: pip install fastapi uvicorn")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# YAML helper
# ---------------------------------------------------------------------------

def _load_yaml(path):
    try:
        if yaml is not None:
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        result = {}
        for line in text.splitlines():
            if ":" in line and not line.strip().startswith("#"):
                parts = line.split(":", 1)
                key = parts[0].strip()
                val = parts[1].strip()
                if val in ("true", "True"):
                    val = True
                elif val in ("false", "False"):
                    val = False
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            val = val.strip("'\"")
                result[key] = val
        return result
    except (FileNotFoundError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------

class InferenceModel:
    """Container for the loaded model and its configuration."""

    def __init__(self, model, model_contract, preprocessing_config, postprocessing_config):
        self.model = model
        self.model_contract = model_contract
        self.preprocessing_config = preprocessing_config
        self.postprocessing_config = postprocessing_config
        self.is_onnx = hasattr(model, "run")

    def predict(self, input_tensor):
        """Run inference on a preprocessed input tensor.

        Args:
            input_tensor: Preprocessed torch.Tensor, shape (C, H, W) or (1, C, H, W).

        Returns:
            Postprocessed predictions.
        """
        device = next(self.model.parameters()).device if not self.is_onnx else "cpu"
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            if self.is_onnx:
                import numpy as np
                input_name = self.model.get_inputs()[0].name
                output = self.model.run(None, {input_name: input_tensor.cpu().numpy()})
                output_tensor = torch.tensor(output[0])
            else:
                output_tensor = self.model(input_tensor)
                if isinstance(output_tensor, dict):
                    output_tensor = output_tensor.get("out", list(output_tensor.values())[0])

        # Apply postprocessing
        from local_infer import apply_postprocessing
        return apply_postprocessing(output_tensor, self.postprocessing_config)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(model_path, model_contract_path, deploy_contract_path=None):
    """Create and configure the FastAPI application.

    Args:
        model_path: Path to exported model file.
        model_contract_path: Path to model_contract.yaml.
        deploy_contract_path: Optional path to deploy_contract.yaml.

    Returns:
        Configured FastAPI app with model loaded.
    """
    app = FastAPI(title="Torch Inference Service", version="0.1.0")

    # Load contracts
    model_contract = _load_yaml(model_contract_path)
    deploy_contract = _load_yaml(deploy_contract_path) if deploy_contract_path else {}

    preprocessing_config = deploy_contract.get("preprocessing", {})
    postprocessing_config = deploy_contract.get("postprocessing", {"type": "softmax_topk", "topk": 5})

    # Build preprocessing pipeline
    from local_infer import create_preprocessing_pipeline
    pipeline = create_preprocessing_pipeline(preprocessing_config, model_contract)

    # Load model (fail gracefully so health endpoint can report status)
    from local_infer import load_exported_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference_model = None
    model_load_error = None
    try:
        model = load_exported_model(model_path, device)
        inference_model = InferenceModel(model, model_contract, preprocessing_config, postprocessing_config)
    except Exception as e:
        model_load_error = str(e)

    # Store on app state
    app.state.model = inference_model
    app.state.pipeline = pipeline
    app.state.model_path = str(model_path)
    app.state.model_load_error = model_load_error

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        model_loaded = app.state.model is not None
        status = "ok" if model_loaded else "error"
        result = {
            "status": status,
            "model": app.state.model_path,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        if app.state.model_load_error:
            result["error"] = app.state.model_load_error
        return result

    @app.post("/predict", response_model=dict)
    async def predict(file: UploadFile = File(...)):
        """Run inference on an uploaded image file.

        Accepts image files (JPEG, PNG, etc.) and returns top-k predictions.
        """
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Read and preprocess the image
        contents = await file.read()
        if Image is None:
            raise HTTPException(status_code=500, detail="Pillow (PIL) is required for image processing")

        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

        if app.state.pipeline:
            try:
                input_tensor = app.state.pipeline(image)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")
        else:
            from torchvision import transforms
            input_tensor = transforms.ToTensor()(image)

        # Run inference
        try:
            predictions = app.state.model.predict(input_tensor)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        return {"predictions": predictions}

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Serve exported model via FastAPI.")
    parser.add_argument("--model-path", required=True, help="Path to exported model (.pt or .onnx)")
    parser.add_argument("--model-contract", required=True, help="Path to model_contract.yaml")
    parser.add_argument("--deploy-contract", help="Path to deploy_contract.yaml (optional)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    args = parser.parse_args()

    app = create_app(
        model_path=args.model_path,
        model_contract_path=args.model_contract,
        deploy_contract_path=args.deploy_contract,
    )

    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"  Model: {args.model_path}")
    print(f"  Docs:  http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, workers=args.workers)


if __name__ == "__main__":
    main()
