"""Tests for model resolution from data_contract."""

import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from resolve_model import resolve, select_backbone


def _make_data_contract(num_classes=10, shape=None):
    return {
        "data_type": "image",
        "task_type": "classification",
        "input_spec": {
            "shape": shape or [3, 224, 224],
            "dtype": "float32",
            "channels_first": True,
        },
        "output_spec": {
            "type": "categorical",
            "num_classes": num_classes,
        },
    }


class TestResolve:
    def test_basic_resolve(self):
        contract = resolve(_make_data_contract())
        assert contract["task_type"] == "classification"
        assert contract["data_type"] == "image"
        assert contract["model_spec"]["backbone"] == "resnet34"
        assert contract["head_spec"]["num_classes"] == 10
        assert contract["forward_spec"]["output_shape"] == ["batch", 10]

    def test_respects_num_classes(self):
        contract = resolve(_make_data_contract(num_classes=5))
        assert contract["head_spec"]["num_classes"] == 5
        assert contract["forward_spec"]["output_shape"] == ["batch", 5]

    def test_uses_input_shape(self):
        contract = resolve(_make_data_contract(shape=[1, 128, 128]))
        assert contract["input_spec"]["shape"] == [1, 128, 128]
        assert contract["model_spec"]["in_channels"] == 1

    def test_default_backbone_is_resnet34(self):
        contract = resolve(_make_data_contract())
        assert contract["model_spec"]["backbone"] == "resnet34"
        assert contract["constraints"]["latency_tier"] == "balanced"

    def test_low_latency_selects_resnet18(self):
        contract = resolve(_make_data_contract(), project_spec={"constraints": {"latency_ms": 30}})
        assert contract["model_spec"]["backbone"] == "resnet18"
        assert contract["constraints"]["latency_tier"] == "fast"

    def test_small_model_selects_efficientnet(self):
        contract = resolve(_make_data_contract(), project_spec={"constraints": {"model_size_mb": 25}})
        assert contract["model_spec"]["backbone"] == "efficientnet_b0"
        assert contract["constraints"]["latency_tier"] == "fast"

    def test_small_model_limit_selects_resnet18(self):
        contract = resolve(_make_data_contract(), project_spec={"constraints": {"model_size_mb": 80}})
        assert contract["model_spec"]["backbone"] == "resnet18"

    def test_large_model_limit_selects_resnet50(self):
        contract = resolve(_make_data_contract(), project_spec={"constraints": {"model_size_mb": 250}})
        assert contract["model_spec"]["backbone"] == "resnet50"
        assert contract["constraints"]["latency_tier"] == "accurate"

    def test_output_has_compatibility(self):
        contract = resolve(_make_data_contract())
        assert contract["compatibility"]["expected_loss"] == "cross_entropy"
        assert contract["compatibility"]["expected_target_type"] == "categorical"
        assert contract["compatibility"]["target_dtype"] == "int64"

    def test_output_has_artifacts(self):
        contract = resolve(_make_data_contract())
        assert "template_name" in contract["artifacts"]
        assert contract["artifacts"]["smoke_test_required"] is True

    def test_output_has_metadata(self):
        contract = resolve(_make_data_contract())
        assert contract["metadata"]["route"] == "image_classification"
        assert contract["metadata"]["priority"] == "P0"


class TestUnsupportedRoutes:
    def test_text_classification_raises(self):
        # text + regression is unsupported
        dc = _make_data_contract()
        dc["data_type"] = "text"
        dc["task_type"] = "regression"
        with pytest.raises(ValueError, match="Unsupported route"):
            resolve(dc)

    def test_image_detection_raises(self):
        dc = _make_data_contract()
        dc["task_type"] = "detection"
        with pytest.raises(ValueError, match="Unsupported route"):
            resolve(dc)

    def test_tabular_raises(self):
        # tabular + detection is unsupported
        dc = _make_data_contract()
        dc["data_type"] = "tabular"
        dc["task_type"] = "detection"
        with pytest.raises(ValueError, match="Unsupported route"):
            resolve(dc)


class TestMissingNumClasses:
    def test_missing_num_classes_raises(self):
        dc = _make_data_contract()
        del dc["output_spec"]["num_classes"]
        with pytest.raises(ValueError, match="num_classes"):
            resolve(dc)


class TestTabularClassification:
    def _make_tabular_dc(self, num_classes=5, num_features=20):
        return {
            "data_type": "tabular",
            "task_type": "classification",
            "input_spec": {"shape": [num_features], "dtype": "float32"},
            "output_spec": {"type": "categorical", "num_classes": num_classes},
        }

    def test_basic_resolve(self):
        contract = resolve(self._make_tabular_dc())
        assert contract["task_type"] == "classification"
        assert contract["data_type"] == "tabular"
        assert contract["model_spec"]["family"] == "mlp"
        assert contract["model_spec"]["architecture"] == "mlp"
        assert contract["model_spec"]["backbone"] == "mlp"
        assert contract["model_spec"]["in_features"] == 20
        assert contract["head_spec"]["type"] == "linear_cls"
        assert contract["head_spec"]["num_classes"] == 5
        assert contract["forward_spec"]["output_shape"] == ["batch", 5]

    def test_pretrained_is_false(self):
        contract = resolve(self._make_tabular_dc())
        assert contract["model_spec"]["pretrained"] is False

    def test_metadata_route(self):
        contract = resolve(self._make_tabular_dc())
        assert contract["metadata"]["route"] == "tabular_classification"
        assert contract["metadata"]["priority"] == "P1"

    def test_respects_num_features(self):
        contract = resolve(self._make_tabular_dc(num_features=50))
        assert contract["model_spec"]["in_features"] == 50
        assert contract["input_spec"]["shape"] == [50]


class TestTabularRegression:
    def _make_tabular_dc(self, output_dim=3, num_features=15):
        return {
            "data_type": "tabular",
            "task_type": "regression",
            "input_spec": {"shape": [num_features], "dtype": "float32"},
            "output_spec": {"type": "continuous", "output_dim": output_dim},
        }

    def test_basic_resolve(self):
        contract = resolve(self._make_tabular_dc())
        assert contract["task_type"] == "regression"
        assert contract["data_type"] == "tabular"
        assert contract["model_spec"]["family"] == "mlp"
        assert contract["model_spec"]["backbone"] == "mlp"
        assert contract["head_spec"]["type"] == "linear_regression"
        assert contract["head_spec"]["output_dim"] == 3
        assert contract["forward_spec"]["output_shape"] == ["batch", 3]

    def test_compatibility(self):
        contract = resolve(self._make_tabular_dc())
        assert contract["compatibility"]["expected_loss"] == "mse"
        assert contract["compatibility"]["expected_target_type"] == "continuous"
        assert contract["compatibility"]["target_dtype"] == "float32"

    def test_metadata_route(self):
        contract = resolve(self._make_tabular_dc())
        assert contract["metadata"]["route"] == "tabular_regression"
        assert contract["metadata"]["priority"] == "P1"

    def test_default_output_dim(self):
        dc = self._make_tabular_dc()
        del dc["output_spec"]["output_dim"]
        contract = resolve(dc)
        assert contract["head_spec"]["output_dim"] == 1


class TestTextClassification:
    def _make_text_dc(self, num_classes=5, max_seq_length=128):
        return {
            "data_type": "text",
            "task_type": "classification",
            "input_spec": {"shape": [max_seq_length], "dtype": "int64", "max_seq_length": max_seq_length},
            "output_spec": {"type": "categorical", "num_classes": num_classes},
        }

    def test_basic_resolve(self):
        contract = resolve(self._make_text_dc())
        assert contract["task_type"] == "classification"
        assert contract["data_type"] == "text"
        assert contract["model_spec"]["family"] == "transformer_encoder"
        assert contract["model_spec"]["architecture"] == "bert"
        assert contract["model_spec"]["pretrained"] is True
        assert contract["head_spec"]["type"] == "pooled_linear_cls"
        assert contract["head_spec"]["num_classes"] == 5
        assert contract["head_spec"]["pooling"] == "cls_token"
        assert contract["forward_spec"]["output_shape"] == ["batch", 5]

    def test_default_backbone_is_bert(self):
        contract = resolve(self._make_text_dc())
        assert contract["model_spec"]["backbone"] == "bert-base-uncased"
        assert contract["constraints"]["latency_tier"] == "balanced"

    def test_low_latency_selects_distilbert(self):
        contract = resolve(self._make_text_dc(), project_spec={"constraints": {"latency_ms": 20}})
        assert contract["model_spec"]["backbone"] == "distilbert-base-uncased"
        assert contract["constraints"]["latency_tier"] == "fast"

    def test_small_model_selects_distilbert(self):
        contract = resolve(self._make_text_dc(), project_spec={"constraints": {"model_size_mb": 80}})
        assert contract["model_spec"]["backbone"] == "distilbert-base-uncased"

    def test_metadata_route(self):
        contract = resolve(self._make_text_dc())
        assert contract["metadata"]["route"] == "text_classification"
        assert contract["metadata"]["priority"] == "P1"

    def test_respects_max_seq_length(self):
        contract = resolve(self._make_text_dc(max_seq_length=256))
        assert contract["input_spec"]["max_seq_length"] == 256
        assert contract["input_spec"]["shape"] == [256]


class TestImageSegmentation:
    def _make_seg_dc(self, num_classes=5, shape=None):
        return {
            "data_type": "image",
            "task_type": "segmentation",
            "input_spec": {"shape": shape or [3, 224, 224], "dtype": "float32", "channels_first": True},
            "output_spec": {"type": "mask", "num_classes": num_classes},
        }

    def test_basic_resolve(self):
        contract = resolve(self._make_seg_dc())
        assert contract["task_type"] == "segmentation"
        assert contract["data_type"] == "image"
        assert contract["model_spec"]["family"] == "cnn_encoder_decoder"
        assert contract["model_spec"]["architecture"] == "deeplabv3"
        assert contract["model_spec"]["backbone"] == "deeplabv3_resnet50"
        assert contract["head_spec"]["type"] == "segmentation_head"
        assert contract["head_spec"]["num_classes"] == 5
        assert contract["compatibility"]["expected_target_type"] == "mask"

    def test_default_backbone(self):
        contract = resolve(self._make_seg_dc())
        assert contract["constraints"]["latency_tier"] == "balanced"

    def test_small_model_selects_mobilenet(self):
        contract = resolve(self._make_seg_dc(), project_spec={"constraints": {"model_size_mb": 30}})
        assert contract["model_spec"]["backbone"] == "deeplabv3_mobilenet_v3_large"
        assert contract["constraints"]["latency_tier"] == "fast"

    def test_metadata_route(self):
        contract = resolve(self._make_seg_dc())
        assert contract["metadata"]["route"] == "image_segmentation"
        assert contract["metadata"]["priority"] == "P1"

    def test_respects_num_classes(self):
        contract = resolve(self._make_seg_dc(num_classes=12))
        assert contract["head_spec"]["num_classes"] == 12
