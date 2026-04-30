"""Tests for local inference pipeline."""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from local_infer import (
    create_preprocessing_pipeline,
    apply_postprocessing,
    format_predictions,
    load_exported_model,
    run_inference,
)
from export_model import (
    build_model_from_contract,
    create_example_input,
    export_torchscript,
)


def _make_minimal_contract():
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


class TestPreprocessing:
    def test_imagenet_norm_pipeline(self):
        config = {
            "name": "imagenet_norm",
            "input_format": "image_file",
            "resize": {"height": 224, "width": 224},
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        }
        pipeline = create_preprocessing_pipeline(config)
        assert pipeline is not None

    def test_no_preprocessing_config(self):
        pipeline = create_preprocessing_pipeline({})
        assert pipeline is None  # No transforms for empty config

    def test_raw_tensor_format(self):
        config = {"input_format": "raw_tensor"}
        pipeline = create_preprocessing_pipeline(config)
        assert pipeline is None  # No transforms needed for raw tensors


class TestPostprocessing:
    def test_softmax_topk(self):
        output = torch.tensor([[0.1, 0.7, 0.05, 0.1, 0.05]])
        config = {"type": "softmax_topk", "topk": 3}
        results = apply_postprocessing(output, config)
        assert isinstance(results, list)
        assert len(results) == 1
        assert "topk" in results[0]
        assert len(results[0]["topk"]) == 3

    def test_argmax(self):
        output = torch.tensor([[0.1, 0.7, 0.05, 0.1, 0.05]])
        config = {"type": "argmax"}
        results = apply_postprocessing(output, config)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["class"] == 1

    def test_sigmoid(self):
        output = torch.tensor([[0.5, -0.5, 2.0]])
        config = {"type": "sigmoid"}
        results = apply_postprocessing(output, config)
        assert isinstance(results, list)
        assert len(results) == 1
        assert len(results[0]) == 3
        # Sigmoid values should be between 0 and 1
        for val in results[0]:
            assert 0 <= val <= 1

    def test_none(self):
        output = torch.tensor([[0.1, 0.7, 0.05]])
        config = {"type": "none"}
        results = apply_postprocessing(output, config)
        assert len(results) == 1
        assert len(results[0]) == 3
        assert results[0][0] == pytest.approx(0.1, abs=1e-6)
        assert results[0][1] == pytest.approx(0.7, abs=1e-6)
        assert results[0][2] == pytest.approx(0.05, abs=1e-6)

    def test_batch_softmax_topk(self):
        output = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
        config = {"type": "softmax_topk", "topk": 2}
        results = apply_postprocessing(output, config)
        assert len(results) == 2
        assert len(results[0]["topk"]) == 2
        assert len(results[1]["topk"]) == 2


class TestFormatPredictions:
    def test_argmax_format(self):
        indices = torch.tensor([1, 2, 0])
        results = format_predictions(indices)
        assert len(results) == 3
        assert results[0]["class"] == 1
        assert results[1]["class"] == 2
        assert results[2]["class"] == 0

    def test_topk_format(self):
        indices = torch.tensor([[1, 0], [2, 1]])
        values = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
        results = format_predictions(indices, values)
        assert len(results) == 2
        assert results[0]["topk"][0]["class"] == 1
        assert results[0]["topk"][0]["score"] == 0.9


class TestInferencePipeline:
    def _export_resnet(self, tmpdir):
        """Helper: export a resnet model to TorchScript and return path + contract."""
        contract = _make_minimal_contract()
        model = build_model_from_contract(contract)
        model.eval()
        example = create_example_input(contract, batch_size=2)
        output_path = str(Path(tmpdir) / "model.pt")
        export_torchscript(model, example, output_path)
        return output_path, contract

    def test_synthetic_inference(self, tmpdir):
        model_path, contract = self._export_resnet(str(tmpdir))
        loaded = load_exported_model(model_path)
        example = create_example_input(contract, batch_size=2)
        output = run_inference(loaded, example)
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 10)

    def test_full_pipeline_with_postprocessing(self, tmpdir):
        model_path, contract = self._export_resnet(str(tmpdir))
        loaded = load_exported_model(model_path)
        example = create_example_input(contract, batch_size=2)
        output = run_inference(loaded, example)
        results = apply_postprocessing(output, {"type": "softmax_topk", "topk": 3})
        assert len(results) == 2
