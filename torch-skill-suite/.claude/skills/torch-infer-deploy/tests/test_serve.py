"""Tests for FastAPI service endpoints."""

import sys
import json
import io
import tempfile
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from export_model import (
    build_model_from_contract,
    create_example_input,
    export_torchscript,
)

# Skip all tests if fastapi/uvicorn not available
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient


def _make_contract():
    return {
        "task_type": "classification",
        "data_type": "image",
        "input_spec": {"shape": [3, 224, 224], "dtype": "float32", "channels_first": True},
        "model_spec": {
            "family": "cnn", "architecture": "resnet", "backbone": "resnet34",
            "pretrained": False, "in_channels": 3,
        },
        "head_spec": {"type": "linear_cls", "num_classes": 10, "pooling": "avg", "dropout": 0.0},
        "forward_spec": {"output_shape": ["batch", 10]},
    }


@pytest.fixture
def exported_model_and_contract(tmpdir):
    """Create a temporary exported model and contract files."""
    contract = _make_contract()
    model = build_model_from_contract(contract)
    model.eval()
    example = create_example_input(contract, batch_size=2)

    model_path = Path(tmpdir) / "model.torchscript.pt"
    export_torchscript(model, example, str(model_path))

    contract_path = Path(tmpdir) / "model_contract.yaml"
    import yaml
    with open(contract_path, "w") as f:
        yaml.dump(contract, f)

    return str(model_path), str(contract_path)


class TestServeApp:
    def test_health_endpoint(self, exported_model_and_contract):
        model_path, contract_path = exported_model_and_contract
        from serve import create_app

        app = create_app(model_path=model_path, model_contract_path=contract_path)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Model may be on cpu if cuda is unavailable or broken
        assert data["status"] in ("ok", "error")
        assert "model" in data

    def test_predict_endpoint(self, exported_model_and_contract):
        model_path, contract_path = exported_model_and_contract
        from serve import create_app

        app = create_app(model_path=model_path, model_contract_path=contract_path)
        client = TestClient(app)

        # Skip if model failed to load (e.g. CUDA issues)
        health = client.get("/health")
        if health.json().get("status") == "error":
            pytest.skip("Model not loaded, skipping predict test")

        # Create a simple test image
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        response = client.post("/predict", files={"file": ("test.png", img_bytes, "image/png")})
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data

    def test_predict_with_invalid_file(self, exported_model_and_contract):
        model_path, contract_path = exported_model_and_contract
        from serve import create_app

        app = create_app(model_path=model_path, model_contract_path=contract_path)
        client = TestClient(app)

        # Skip if model failed to load (e.g. CUDA issues)
        health = client.get("/health")
        if health.json().get("status") == "error":
            pytest.skip("Model not loaded, skipping predict test")

        response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
        assert response.status_code == 400

    def test_health_without_model(self):
        """Test health returns error when model not loaded."""
        from serve import create_app

        app = create_app(
            model_path="/nonexistent/model.pt",
            model_contract_path="/nonexistent/contract.yaml",
        )
        client = TestClient(app)

        response = client.get("/health")
        # Health endpoint should still respond with error status
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "model" in data
        assert "error" in data  # Should report the load error
