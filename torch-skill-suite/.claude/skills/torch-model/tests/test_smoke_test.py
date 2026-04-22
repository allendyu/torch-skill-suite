"""Tests for model smoke testing (dummy forward pass)."""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from smoke_test_model import (
    create_dummy_input,
    check_output_shape,
    _add_template_path,
    build_model_from_contract,
)


def _make_model_contract(backbone="resnet34", num_classes=10, shape=None):
    return {
        "task_type": "classification",
        "data_type": "image",
        "input_spec": {
            "shape": shape or [3, 224, 224],
            "dtype": "float32",
            "channels_first": True,
        },
        "model_spec": {
            "family": "cnn",
            "architecture": "resnet",
            "backbone": backbone,
            "pretrained": False,
            "in_channels": 3,
            "feature_dim": 512,
        },
        "head_spec": {
            "type": "linear_cls",
            "num_classes": num_classes,
            "pooling": "avg",
            "dropout": 0.0,
        },
        "forward_spec": {
            "output_shape": ["batch", num_classes],
        },
    }


class TestCreateDummyInput:
    def test_float32_input(self):
        spec = {"shape": [3, 224, 224], "dtype": "float32"}
        tensor = create_dummy_input(spec, batch_size=2)
        assert tensor.shape == (2, 3, 224, 224)
        assert str(tensor.dtype) == "torch.float32"

    def test_int64_input(self):
        spec = {"shape": [512], "dtype": "int64"}
        tensor = create_dummy_input(spec, batch_size=4)
        assert tensor.shape == (4, 512)
        assert str(tensor.dtype) == "torch.int64"

    def test_custom_batch_size(self):
        spec = {"shape": [3, 224, 224], "dtype": "float32"}
        tensor = create_dummy_input(spec, batch_size=8)
        assert tensor.shape == (8, 3, 224, 224)


class TestCheckOutputShape:
    def test_exact_match(self):
        output = type("Output", (), {"shape": [2, 10]})()
        passed, msg = check_output_shape(output, ["batch", 10], 2)
        assert passed

    def test_rank_mismatch(self):
        output = type("Output", (), {"shape": [2, 10, 1]})()
        passed, msg = check_output_shape(output, ["batch", 10], 2)
        assert not passed
        assert "Rank mismatch" in msg

    def test_dimension_mismatch(self):
        output = type("Output", (), {"shape": [2, 5]})()
        passed, msg = check_output_shape(output, ["batch", 10], 2)
        assert not passed
        assert "Dimension" in msg


class TestBuildModel:
    def setup_method(self):
        _add_template_path()

    def test_build_resnet34(self):
        contract = _make_model_contract("resnet34", num_classes=5)
        model = build_model_from_contract(contract)
        import torch
        dummy = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (2, 5)

    def test_build_resnet18(self):
        contract = _make_model_contract("resnet18", num_classes=3)
        model = build_model_from_contract(contract)
        import torch
        dummy = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (2, 3)

    def test_build_efficientnet(self):
        contract = _make_model_contract("efficientnet_b0", num_classes=7)
        contract["model_spec"]["architecture"] = "efficientnet"
        contract["model_spec"]["feature_dim"] = 1280
        model = build_model_from_contract(contract)
        import torch
        dummy = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (2, 7)

    def test_build_resnet50(self):
        contract = _make_model_contract("resnet50", num_classes=100)
        contract["model_spec"]["feature_dim"] = 2048
        model = build_model_from_contract(contract)
        import torch
        dummy = torch.randn(2, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (2, 100)

    def test_unknown_architecture_raises(self):
        contract = _make_model_contract()
        contract["model_spec"]["architecture"] = "unknown"
        with pytest.raises(ValueError):
            build_model_from_contract(contract)
