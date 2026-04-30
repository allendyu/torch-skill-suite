"""Tests for generate_dataset.py."""

import ast
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from generate_dataset import generate, GENERATORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_python(source: str) -> bool:
    """Check that source code is syntactically valid Python."""
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def _make_image_cls_contract():
    return {
        "data_type": "image",
        "task_type": "classification",
        "input_spec": {"shape": [3, 224, 224], "dtype": "float32", "channels_first": True},
        "output_spec": {"type": "categorical", "num_classes": 10, "label_map": {"0": "cat", "1": "dog"}},
        "splits": {"train": "data/train", "val": "data/val", "test": "data/test"},
        "preprocessing": [{"name": "normalize", "params": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}}],
    }


def _make_tabular_cls_contract():
    return {
        "data_type": "tabular",
        "task_type": "classification",
        "input_spec": {"shape": [20], "dtype": "float32", "feature_columns": [f"feat_{i}" for i in range(20)]},
        "output_spec": {"type": "categorical", "num_classes": 3, "target_column": "label"},
        "splits": {"train": "data/train.csv", "val": "data/val.csv"},
    }


def _make_tabular_reg_contract():
    return {
        "data_type": "tabular",
        "task_type": "regression",
        "input_spec": {"shape": [15], "dtype": "float32", "feature_columns": [f"f{i}" for i in range(15)]},
        "output_spec": {"type": "continuous", "output_dim": 1, "target_column": "price"},
        "splits": {"train": "data/train.csv", "val": "data/val.csv", "test": "data/test.csv"},
    }


def _make_text_cls_contract():
    return {
        "data_type": "text",
        "task_type": "classification",
        "input_spec": {"shape": [128], "dtype": "int64", "max_seq_length": 128, "text_column": "review"},
        "output_spec": {"type": "categorical", "num_classes": 2, "target_column": "sentiment"},
        "splits": {"train": "data/train.jsonl", "val": "data/val.jsonl"},
    }


def _make_seg_contract():
    return {
        "data_type": "image",
        "task_type": "segmentation",
        "input_spec": {"shape": [3, 512, 512], "dtype": "float32", "channels_first": True},
        "output_spec": {"type": "mask", "num_classes": 21},
        "splits": {"train": "data/train", "val": "data/val"},
    }


# ---------------------------------------------------------------------------
# Tests: route dispatch
# ---------------------------------------------------------------------------

class TestRouteDispatch:
    def test_all_supported_routes_have_generators(self):
        supported = [
            ("image", "classification"),
            ("tabular", "classification"),
            ("tabular", "regression"),
            ("text", "classification"),
            ("image", "segmentation"),
        ]
        for data_type, task_type in supported:
            assert (data_type, task_type) in GENERATORS, f"Missing generator for {data_type}+{task_type}"

    def test_unsupported_route_raises(self):
        with pytest.raises(ValueError, match="No generator"):
            generate({"data_type": "video", "task_type": "classification"})


# ---------------------------------------------------------------------------
# Tests: image classification
# ---------------------------------------------------------------------------

class TestImageClassification:
    def test_generates_valid_python(self):
        files = generate(_make_image_cls_contract())
        assert len(files) == 3
        for name, code in files.items():
            assert _is_valid_python(code), f"{name} is not valid Python"

    def test_dataset_includes_label_map(self):
        files = generate(_make_image_cls_contract())
        assert "LABEL_MAP" in files["dataset.py"]

    def test_transforms_include_normalize(self):
        files = generate(_make_image_cls_contract())
        assert "Normalize" in files["transforms.py"]
        assert "0.5" in files["transforms.py"]

    def test_dataloader_has_splits(self):
        files = generate(_make_image_cls_contract())
        assert "data/train" in files["dataloader.py"]
        assert "data/val" in files["dataloader.py"]

    def test_writes_to_output_dir(self, tmp_path):
        generate(_make_image_cls_contract(), output_dir=str(tmp_path))
        assert (tmp_path / "dataset.py").exists()
        assert (tmp_path / "transforms.py").exists()
        assert (tmp_path / "dataloader.py").exists()

    def test_component_filter(self, tmp_path):
        generate(_make_image_cls_contract(), output_dir=str(tmp_path), components=["dataset"])
        assert (tmp_path / "dataset.py").exists()
        assert not (tmp_path / "transforms.py").exists()

    def test_default_normalize_when_no_preprocessing(self):
        contract = _make_image_cls_contract()
        del contract["preprocessing"]
        files = generate(contract)
        assert "Normalize" in files["transforms.py"]


# ---------------------------------------------------------------------------
# Tests: tabular classification
# ---------------------------------------------------------------------------

class TestTabularClassification:
    def test_generates_valid_python(self):
        files = generate(_make_tabular_cls_contract())
        assert len(files) == 3
        for name, code in files.items():
            assert _is_valid_python(code), f"{name} is not valid Python"

    def test_dataset_has_correct_target_dtype(self):
        files = generate(_make_tabular_cls_contract())
        assert "torch.long" in files["dataset.py"]

    def test_feature_columns_in_dataset(self):
        files = generate(_make_tabular_cls_contract())
        assert "feat_0" in files["dataset.py"]
        assert "feat_19" in files["dataset.py"]

    def test_scaler_in_transforms(self):
        files = generate(_make_tabular_cls_contract())
        assert "StandardScaler" in files["transforms.py"]


# ---------------------------------------------------------------------------
# Tests: tabular regression
# ---------------------------------------------------------------------------

class TestTabularRegression:
    def test_generates_valid_python(self):
        files = generate(_make_tabular_reg_contract())
        assert len(files) == 3
        for name, code in files.items():
            assert _is_valid_python(code), f"{name} is not valid Python"

    def test_dataset_has_float32_target(self):
        files = generate(_make_tabular_reg_contract())
        assert "torch.float32" in files["dataset.py"]

    def test_task_type_is_regression(self):
        files = generate(_make_tabular_reg_contract())
        assert '"regression"' in files["dataloader.py"]


# ---------------------------------------------------------------------------
# Tests: text classification
# ---------------------------------------------------------------------------

class TestTextClassification:
    def test_generates_valid_python(self):
        files = generate(_make_text_cls_contract())
        assert len(files) == 3
        for name, code in files.items():
            assert _is_valid_python(code), f"{name} is not valid Python"

    def test_dataset_uses_text_column(self):
        files = generate(_make_text_cls_contract())
        assert "review" in files["dataset.py"]

    def test_transforms_has_tokenizer(self):
        files = generate(_make_text_cls_contract())
        assert "TextTokenizer" in files["transforms.py"]

    def test_dataloader_uses_jsonl(self):
        files = generate(_make_text_cls_contract())
        assert "train.jsonl" in files["dataloader.py"]


# ---------------------------------------------------------------------------
# Tests: image segmentation
# ---------------------------------------------------------------------------

class TestImageSegmentation:
    def test_generates_valid_python(self):
        files = generate(_make_seg_contract())
        assert len(files) == 3
        for name, code in files.items():
            assert _is_valid_python(code), f"{name} is not valid Python"

    def test_dataset_has_segmentation_class(self):
        files = generate(_make_seg_contract())
        assert "SegmentationDataset" in files["dataset.py"]

    def test_mask_transform_uses_nearest(self):
        files = generate(_make_seg_contract())
        assert "NEAREST" in files["transforms.py"]

    def test_respects_input_shape(self):
        files = generate(_make_seg_contract())
        assert "512" in files["transforms.py"]

    def test_num_classes_in_dataloader(self):
        files = generate(_make_seg_contract())
        assert "21" in files["dataloader.py"]
