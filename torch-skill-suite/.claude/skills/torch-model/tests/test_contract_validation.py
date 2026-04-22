"""Tests for model_contract validation."""

import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from validate_contract import (
    _validate_builtin,
    _load_yaml,
    validate_contract,
    find_schema,
    find_shared_catalog,
)


def _make_valid_contract():
    return {
        "task_type": "classification",
        "data_type": "image",
        "input_spec": {"shape": [3, 224, 224], "dtype": "float32"},
        "model_spec": {
            "family": "cnn",
            "architecture": "resnet",
            "backbone": "resnet34",
            "pretrained": True,
            "in_channels": 3,
        },
        "head_spec": {"type": "linear_cls", "num_classes": 10},
        "forward_spec": {"output_shape": ["batch", 10]},
        "compatibility": {"expected_target_type": "categorical"},
    }


class TestBuiltinValidation:
    def test_valid_contract_passes(self):
        errors = _validate_builtin(_make_valid_contract())
        assert len(errors) == 0

    def test_missing_required_fields(self):
        errors = _validate_builtin({})
        assert len(errors) >= 6  # 6 required top-level fields

    def test_missing_task_type(self):
        c = _make_valid_contract()
        del c["task_type"]
        errors = _validate_builtin(c)
        assert any("task_type" in e for e in errors)

    def test_invalid_task_type(self):
        c = _make_valid_contract()
        c["task_type"] = "unknown"
        errors = _validate_builtin(c)
        assert any("task_type" in e.lower() for e in errors)

    def test_invalid_data_type(self):
        c = _make_valid_contract()
        c["data_type"] = "unknown"
        errors = _validate_builtin(c)
        assert any("data_type" in e.lower() for e in errors)

    def test_missing_input_shape(self):
        c = _make_valid_contract()
        del c["input_spec"]["shape"]
        errors = _validate_builtin(c)
        assert any("shape" in e for e in errors)

    def test_missing_input_dtype(self):
        c = _make_valid_contract()
        del c["input_spec"]["dtype"]
        errors = _validate_builtin(c)
        assert any("dtype" in e for e in errors)

    def test_missing_model_spec_fields(self):
        c = _make_valid_contract()
        del c["model_spec"]["backbone"]
        errors = _validate_builtin(c)
        assert any("backbone" in e for e in errors)

    def test_missing_head_type(self):
        c = _make_valid_contract()
        del c["head_spec"]["type"]
        errors = _validate_builtin(c)
        assert any("type" in e for e in errors)

    def test_classification_requires_num_classes(self):
        c = _make_valid_contract()
        del c["head_spec"]["num_classes"]
        errors = _validate_builtin(c)
        assert any("num_classes" in e for e in errors)

    def test_classification_requires_categorical_target(self):
        c = _make_valid_contract()
        c["compatibility"] = {"expected_target_type": "continuous"}
        errors = _validate_builtin(c)
        assert any("categorical" in e for e in errors)

    def test_image_requires_3d_shape(self):
        c = _make_valid_contract()
        c["input_spec"]["shape"] = [224, 224]
        errors = _validate_builtin(c)
        assert any("3 dimension" in e for e in errors)

    def test_image_3d_shape_passes(self):
        c = _make_valid_contract()
        c["input_spec"]["shape"] = [3, 224, 224]
        errors = _validate_builtin(c)
        assert len(errors) == 0

    def test_regression_requires_output_dim(self):
        c = _make_valid_contract()
        c["task_type"] = "regression"
        c["head_spec"] = {"type": "linear_regression"}
        c["compatibility"] = {"expected_target_type": "continuous"}
        errors = _validate_builtin(c)
        assert any("output_dim" in e for e in errors)

    def test_missing_forward_output_shape(self):
        c = _make_valid_contract()
        del c["forward_spec"]["output_shape"]
        errors = _validate_builtin(c)
        assert any("output_shape" in e for e in errors)


class TestSharedCatalog:
    def test_all_catalog_entries_valid(self):
        catalog_path = find_shared_catalog()
        if not catalog_path:
            pytest.skip("Shared catalog not found")
        schema_path = find_schema()
        instance = _load_yaml(catalog_path)
        for name, entry in instance.items():
            if not isinstance(entry, dict):
                continue
            errors = validate_contract(str(catalog_path), schema_path) if False else _validate_builtin(entry)
            # Validate each entry directly
            errors = _validate_builtin(entry)
            assert len(errors) == 0, f"Catalog entry '{name}' has errors: {errors}"


class TestExampleContracts:
    def test_all_examples_valid(self):
        examples_dir = Path(__file__).resolve().parent.parent / "examples" / "contracts"
        if not examples_dir.exists():
            pytest.skip("Examples directory not found")
        for example_file in sorted(examples_dir.glob("*.yaml")):
            instance = _load_yaml(str(example_file))
            errors = _validate_builtin(instance)
            assert len(errors) == 0, f"Example '{example_file.name}' has errors: {errors}"
