"""Tests for deploy_contract validation."""

import json
import sys
import types
from pathlib import Path

import pytest

# Load validate_contract directly to avoid namespace collision with
# torch-model/scripts/validate_contract.py when running all-skill tests.
_script_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_script_dir))
_validate_mod_name = "validate_contract_deploy"
if _validate_mod_name not in sys.modules:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        _validate_mod_name, _script_dir / "validate_contract.py"
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_validate_mod_name] = _mod
    _spec.loader.exec_module(_mod)
else:
    _mod = sys.modules[_validate_mod_name]

_validate_builtin = _mod._validate_builtin
_load_yaml = _mod._load_yaml
find_schema = _mod.find_schema
find_shared_catalog = _mod.find_shared_catalog
validate_contract = _mod.validate_contract
VALID_EXPORT_FORMATS = _mod.VALID_EXPORT_FORMATS
VALID_SERVICE_TYPES = _mod.VALID_SERVICE_TYPES
VALID_INPUT_FORMATS = _mod.VALID_INPUT_FORMATS
VALID_POSTPROCESS_TYPES = _mod.VALID_POSTPROCESS_TYPES


def _make_valid_contract():
    return {
        "export": {"format": "torchscript"},
        "service": {"type": "fastapi"},
        "preprocessing": {"name": "imagenet_norm", "input_format": "image_file"},
        "postprocessing": {"type": "softmax_topk", "topk": 5},
    }


class TestBuiltinValidation:
    def test_valid_minimal_contract_passes(self):
        contract = {"export": {"format": "torchscript"}}
        errors = _validate_builtin(contract)
        assert errors == []

    def test_valid_full_contract_passes(self):
        errors = _validate_builtin(_make_valid_contract())
        assert errors == []

    def test_empty_contract_passes(self):
        errors = _validate_builtin({})
        assert errors == []

    def test_invalid_export_format(self):
        contract = {"export": {"format": "savedmodel"}}
        errors = _validate_builtin(contract)
        assert any("export.format" in e for e in errors)

    def test_invalid_service_type(self):
        contract = {"service": {"type": "grpc"}}
        errors = _validate_builtin(contract)
        assert any("service.type" in e for e in errors)

    def test_invalid_input_format(self):
        contract = {"preprocessing": {"input_format": "video_file"}}
        errors = _validate_builtin(contract)
        assert any("preprocessing.input_format" in e for e in errors)

    def test_invalid_postprocess_type(self):
        contract = {"postprocessing": {"type": "invalid"}}
        errors = _validate_builtin(contract)
        assert any("postprocessing.type" in e for e in errors)

    def test_missing_topk_for_softmax_topk(self):
        contract = {"postprocessing": {"type": "softmax_topk"}}
        errors = _validate_builtin(contract)
        assert any("topk" in e for e in errors)

    def test_onnx_opset_too_low(self):
        contract = {"export": {"format": "onnx", "opset_version": 9}}
        errors = _validate_builtin(contract)
        assert any("opset_version" in e for e in errors)

    def test_valid_onnx_opset(self):
        contract = {"export": {"format": "onnx", "opset_version": 17}}
        errors = _validate_builtin(contract)
        assert errors == []

    def test_valid_batch_service(self):
        contract = {"service": {"type": "batch", "max_batch_size": 64}}
        errors = _validate_builtin(contract)
        assert errors == []


class TestSharedExamples:
    def test_all_examples_valid(self):
        catalog_path = find_shared_catalog()
        if not catalog_path:
            pytest.skip("Shared catalog not found")
        instance = _load_yaml(catalog_path)
        schema_path = find_schema()
        all_errors = {}
        for name, entry in instance.items():
            if not isinstance(entry, dict):
                continue
            errors = validate_contract(catalog_path, schema_path)
            if errors:
                all_errors[name] = errors
        assert not all_errors, f"Examples with errors: {all_errors}"

    def test_catalog_exists(self):
        catalog_path = find_shared_catalog()
        assert catalog_path is not None, "Shared catalog not found"
        assert Path(catalog_path).exists()

    def test_schema_exists(self):
        schema_path = find_schema()
        assert schema_path is not None, "Schema not found"
        assert Path(schema_path).exists()
