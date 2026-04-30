"""Tests for model export pipeline."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from export_model import (
    build_model_from_contract,
    create_example_input,
    _wrap_model_for_export,
    load_checkpoint,
    export_torchscript,
    validate_exported_model,
)


def _make_model_contract(architecture="resnet", backbone="resnet34",
                         num_classes=10, shape=None, task_type="classification",
                         data_type="image"):
    if shape is None:
        shape = [3, 224, 224]
    contract = {
        "task_type": task_type,
        "data_type": data_type,
        "input_spec": {"shape": shape, "dtype": "float32", "channels_first": True},
        "model_spec": {
            "family": "cnn",
            "architecture": architecture,
            "backbone": backbone,
            "pretrained": False,
            "in_channels": shape[0] if len(shape) >= 3 else 3,
        },
        "head_spec": {"type": "linear_cls", "num_classes": num_classes, "pooling": "avg", "dropout": 0.0},
        "forward_spec": {
            "input_tensor_name": "images",
            "output_tensor_name": "logits",
            "output_shape": ["batch", num_classes],
        },
        "compatibility": {
            "expected_target_type": "categorical",
            "expected_loss": "cross_entropy",
            "target_dtype": "int64",
            "output_activation": "none",
        },
        "constraints": {"latency_tier": "balanced", "model_size_mb": 85},
        "artifacts": {"template_name": "image_classification/resnet", "smoke_test_required": True},
    }
    # Add extra fields for non-image architectures
    if architecture == "mlp":
        contract["model_spec"]["in_features"] = 10
        contract["input_spec"]["shape"] = [10]
        contract["head_spec"]["type"] = "linear_cls"
    if architecture == "bert":
        contract["data_type"] = "text"
        contract["model_spec"]["family"] = "transformer_encoder"
        contract["input_spec"]["shape"] = [128]
        contract["input_spec"]["max_seq_length"] = 128
        contract["input_spec"]["dtype"] = "int64"
        contract["model_spec"]["in_channels"] = 1  # dummy, not used by bert
        contract["head_spec"]["type"] = "pooled_linear_cls"
    if architecture == "deeplabv3":
        contract["task_type"] = "segmentation"
        contract["model_spec"]["architecture"] = "deeplabv3"
        contract["model_spec"]["backbone"] = "deeplabv3_resnet50"
    if architecture == "unet":
        contract["task_type"] = "segmentation"
        contract["model_spec"]["architecture"] = "unet"
        contract["model_spec"]["backbone"] = "unet"
    return contract


class TestBuildModelAndLoadCheckpoint:
    def test_build_resnet34(self):
        contract = _make_model_contract()
        model = build_model_from_contract(contract)
        assert model is not None
        # Verify forward pass works
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (2, 10)

    def test_build_efficientnet(self):
        contract = _make_model_contract(architecture="efficientnet", backbone="efficientnet_b0")
        model = build_model_from_contract(contract)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (2, 10)

    def test_build_mlp(self):
        contract = _make_model_contract(architecture="mlp", backbone="mlp", data_type="tabular")
        model = build_model_from_contract(contract)
        dummy = torch.randn(2, 10)
        out = model(dummy)
        assert out.shape == (2, 10)

    def test_build_unet(self):
        contract = _make_model_contract(architecture="unet", backbone="unet",
                                        task_type="segmentation", num_classes=21)
        model = build_model_from_contract(contract)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        assert out.shape == (2, 21, 224, 224)

    def test_build_deeplabv3(self):
        contract = _make_model_contract(architecture="deeplabv3", backbone="deeplabv3_resnet50",
                                        task_type="segmentation", num_classes=21)
        model = build_model_from_contract(contract)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        # DeepLabV3 returns dict
        assert isinstance(out, dict)
        assert "out" in out
        assert out["out"].shape[1] == 21

    def test_build_text_classifier(self):
        contract = _make_model_contract(architecture="bert", backbone="bert-tiny",
                                        data_type="text", task_type="classification",
                                        num_classes=5)
        model = build_model_from_contract(contract)
        input_ids = torch.randint(1, 1000, (2, 128), dtype=torch.long)
        attention_mask = torch.ones(2, 128, dtype=torch.long)
        out = model(input_ids, attention_mask)
        assert out.shape == (2, 5)

    def test_unknown_architecture_raises(self):
        contract = _make_model_contract(architecture="unknown", backbone="unknown")
        with pytest.raises(ValueError, match="Unsupported architecture"):
            build_model_from_contract(contract)


class TestCreateExampleInput:
    def test_image_input(self):
        contract = _make_model_contract()
        example = create_example_input(contract, batch_size=4)
        assert isinstance(example, torch.Tensor)
        assert example.shape == (4, 3, 224, 224)

    def test_text_input(self):
        contract = _make_model_contract(architecture="bert", backbone="bert-tiny",
                                        data_type="text", task_type="classification")
        example = create_example_input(contract, batch_size=2)
        assert isinstance(example, tuple)
        assert len(example) == 2
        assert example[0].shape == (2, 128)

    def test_custom_batch_size(self):
        contract = _make_model_contract()
        example = create_example_input(contract, batch_size=8)
        assert example.shape[0] == 8


class TestModelWrapping:
    def test_deeplabv3_wrapping(self):
        contract = _make_model_contract(architecture="deeplabv3", backbone="deeplabv3_resnet50",
                                        task_type="segmentation", num_classes=21)
        model = build_model_from_contract(contract)
        wrapped = _wrap_model_for_export(model, contract)
        dummy = torch.randn(2, 3, 224, 224)
        out = wrapped(dummy)
        # Wrapped output should be a tensor, not dict
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == 21

    def test_resnet_no_wrapping(self):
        contract = _make_model_contract()
        model = build_model_from_contract(contract)
        wrapped = _wrap_model_for_export(model, contract)
        assert wrapped is model  # No wrapper needed


class TestTorchScriptExport:
    def test_export_and_reload_resnet(self):
        contract = _make_model_contract()
        model = build_model_from_contract(contract)
        model.eval()
        example = create_example_input(contract, batch_size=2)

        # Get reference output
        with torch.no_grad():
            ref_out = model(example)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            exported = export_torchscript(model, example, f.name)
            validate_exported_model(exported, example, ref_out)

            # Reload and verify
            loaded = torch.jit.load(f.name)
            loaded.eval()
            with torch.no_grad():
                loaded_out = loaded(example)
            assert loaded_out.shape == ref_out.shape

    def test_export_mlp(self):
        contract = _make_model_contract(architecture="mlp", backbone="mlp", data_type="tabular")
        model = build_model_from_contract(contract)
        model.eval()
        example = create_example_input(contract, batch_size=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            exported = export_torchscript(model, example, f.name)
            validate_exported_model(exported, example)

    def test_export_deeplabv3(self):
        contract = _make_model_contract(architecture="deeplabv3", backbone="deeplabv3_resnet50",
                                        task_type="segmentation", num_classes=21)
        model = build_model_from_contract(contract)
        wrapped = _wrap_model_for_export(model, contract)
        wrapped.eval()
        example = create_example_input(contract, batch_size=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            exported = export_torchscript(wrapped, example, f.name)
            validate_exported_model(exported, example)
