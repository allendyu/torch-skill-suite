#!/usr/bin/env python3
"""
Validate deploy contract YAML files against the JSON schema.

Usage:
    python validate_contract.py --contract path/to/deploy_contract.yaml
    python validate_contract.py --contract path/to/deploy_contract.yaml --schema path/to/schema.json
    python validate_contract.py --validate-examples
    python validate_contract.py --validate-all

If optional YAML/jsonschema packages are unavailable, the script falls back to a
built-in YAML subset parser and built-in validation.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

try:
    import jsonschema
    import jsonschema.exceptions
except ImportError:
    jsonschema = None


VALID_EXPORT_FORMATS = {"torchscript", "onnx"}
VALID_SERVICE_TYPES = {"fastapi", "batch"}
VALID_INPUT_FORMATS = {"image_file", "numpy_array", "text", "raw_tensor"}
VALID_POSTPROCESS_TYPES = {"softmax_topk", "argmax", "sigmoid", "none"}


class SimpleYAMLParser:
    """Minimal YAML subset parser for environments without PyYAML."""

    def __init__(self, text):
        self.lines = self._prepare_lines(text)

    def _strip_inline_comment(self, line):
        in_single = False
        in_double = False
        for i, ch in enumerate(line):
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if ch == "#" and not in_single and not in_double:
                if i == 0 or line[i - 1].isspace():
                    return line[:i].rstrip()
        return line.rstrip()

    def _prepare_lines(self, text):
        prepared = []
        for raw_line in text.splitlines():
            stripped = self._strip_inline_comment(raw_line)
            if stripped or not raw_line.strip():
                prepared.append(stripped)
        return prepared

    def parse(self):
        result = {}
        current_doc = result
        stack = [(None, 0, None)]
        for lineno, line in enumerate(self.lines, start=1):
            content = line.rstrip()
            if not content or content.isspace():
                continue
            indent = len(line) - len(line.lstrip())
            key, sep, value = self._parse_line(content)
            if key is None:
                continue
            while stack and stack[-1][1] > indent:
                _, _, parent = stack.pop()
            current_doc = stack[-1][2] if stack[-1][2] is not None else result
            if value is not None:
                current_doc[key] = value
            else:
                new_doc = {}
                if isinstance(current_doc, list):
                    current_doc.append(new_doc)
                else:
                    current_doc[key] = new_doc
                current_doc = new_doc
                stack.append((key, indent, new_doc))
        return result

    def _parse_line(self, line):
        content = line.strip()
        if content.startswith("- "):
            content = content[2:]
            key = None
        elif ":" in content:
            parts = content.split(":", 1)
            key = parts[0].strip().strip("'").strip('"')
            rest = parts[1].strip() if len(parts) > 1 else ""
            if not rest:
                return key, None, None
            content = rest
        else:
            return None, None, None
        parsed = self._parse_value(content)
        return key, parsed, parsed

    def _parse_value(self, text):
        if text in ("true", "True", "yes"):
            return True
        if text in ("false", "False", "no"):
            return False
        if text in ("null", "None", "~"):
            return None
        if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
            return self._parse_json_like(text)
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        try:
            return int(text)
        except (ValueError, TypeError):
            pass
        try:
            return float(text)
        except (ValueError, TypeError):
            pass
        return text

    def _parse_json_like(self, text):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return self._parse_loose_list(text)

    def _parse_loose_list(self, text):
        text = text.strip("[] ")
        if not text:
            return []
        items = []
        depth = 0
        current = ""
        for ch in text:
            if ch in "{[(":
                depth += 1
            elif ch in "}])":
                depth -= 1
            if ch == "," and depth == 0:
                items.append(self._parse_value(current.strip()))
                current = ""
            else:
                current += ch
        if current.strip():
            items.append(self._parse_value(current.strip()))
        return items


def _load_yaml(path):
    if yaml is not None:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    return SimpleYAMLParser(text).parse()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_with_jsonschema(instance, schema):
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    return errors


def _validate_builtin(instance):
    """Built-in validation for deploy_contract structure."""
    errors = []

    # export section
    export = instance.get("export", {})
    if isinstance(export, dict):
        if "format" in export and export["format"] not in VALID_EXPORT_FORMATS:
            errors.append(f"export.format '{export['format']}' is invalid. Must be one of: {sorted(VALID_EXPORT_FORMATS)}")
        if export.get("format") == "onnx":
            opset = export.get("opset_version", 17)
            if isinstance(opset, (int, float)) and opset < 11:
                errors.append(f"export.opset_version must be >= 11 for ONNX, got {opset}")

    # service section
    service = instance.get("service", {})
    if isinstance(service, dict):
        if "type" in service and service["type"] not in VALID_SERVICE_TYPES:
            errors.append(f"service.type '{service['type']}' is invalid. Must be one of: {sorted(VALID_SERVICE_TYPES)}")

    # preprocessing section
    preproc = instance.get("preprocessing", {})
    if isinstance(preproc, dict):
        if "input_format" in preproc and preproc["input_format"] not in VALID_INPUT_FORMATS:
            errors.append(f"preprocessing.input_format '{preproc['input_format']}' is invalid. Must be one of: {sorted(VALID_INPUT_FORMATS)}")

    # postprocessing section
    postproc = instance.get("postprocessing", {})
    if isinstance(postproc, dict):
        if "type" in postproc and postproc["type"] not in VALID_POSTPROCESS_TYPES:
            errors.append(f"postprocessing.type '{postproc['type']}' is invalid. Must be one of: {sorted(VALID_POSTPROCESS_TYPES)}")
        if postproc.get("type") == "softmax_topk" and "topk" not in postproc:
            errors.append("postprocessing.topk is required when postprocessing.type is 'softmax_topk'")

    return errors


def validate_contract(contract_path, schema_path=None):
    instance = _load_yaml(contract_path)
    errors = []

    if schema_path and jsonschema is not None:
        schema = _load_json(schema_path)
        js_errors = _validate_with_jsonschema(instance, schema)
        errors = [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in js_errors]
    elif schema_path and jsonschema is None:
        schema = _load_json(schema_path)
        errors = _validate_builtin(instance)
    else:
        errors = _validate_builtin(instance)

    return errors


def find_schema():
    """Find the deploy_contract schema file relative to this script."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / ".." / ".." / ".." / ".." / "shared" / "schemas" / "deploy_contract.schema.json",
        script_dir / ".." / ".." / "shared" / "schemas" / "deploy_contract.schema.json",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return None


def find_shared_catalog():
    """Find the shared deploy_contract.example.yaml catalog."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / ".." / ".." / ".." / ".." / "shared" / "contracts" / "deploy_contract.example.yaml",
        script_dir / ".." / ".." / "shared" / "contracts" / "deploy_contract.example.yaml",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return None


def cmd_validate(args):
    contract_path = args.contract
    schema_path = args.schema or find_schema()
    if schema_path:
        print(f"Using schema: {schema_path}")
    errors = validate_contract(contract_path, schema_path)
    if errors:
        print(f"Validation FAILED ({len(errors)} error(s)):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("Validation PASSED")
        sys.exit(0)


def cmd_validate_examples():
    catalog_path = find_shared_catalog()
    if not catalog_path:
        print("Shared deploy contract examples not found.")
        sys.exit(1)
    schema_path = find_schema()
    instance = _load_yaml(catalog_path)
    all_errors = {}
    for name, entry in instance.items():
        if not isinstance(entry, dict):
            continue
        errors = []
        if schema_path and jsonschema is not None:
            schema = _load_json(schema_path)
            js_errors = _validate_with_jsonschema(entry, schema)
            errors = [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in js_errors]
        else:
            errors = _validate_builtin(entry)
        if errors:
            all_errors[name] = errors
    if all_errors:
        print(f"Validation FAILED ({len(all_errors)} example(s) with errors):")
        for name, errors in all_errors.items():
            print(f"  {name}:")
            for err in errors:
                print(f"    - {err}")
        sys.exit(1)
    else:
        print(f"All {len(instance)} example(s) passed validation.")
        sys.exit(0)


def cmd_validate_all():
    errors_occurred = False
    catalog_path = find_shared_catalog()
    if catalog_path:
        print("--- Validating deploy contract examples ---")
        instance = _load_yaml(catalog_path)
        schema_path = find_schema()
        for name, entry in instance.items():
            if not isinstance(entry, dict):
                continue
            errors = []
            if schema_path and jsonschema is not None:
                schema = _load_json(schema_path)
                js_errors = _validate_with_jsonschema(entry, schema)
                errors = [f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in js_errors]
            else:
                errors = _validate_builtin(entry)
            if errors:
                print(f"  FAIL  {name}: {errors[0]}")
                errors_occurred = True
            else:
                print(f"  PASS  {name}")
    if errors_occurred:
        sys.exit(1)
    else:
        print("All validations passed.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Validate deploy contract YAML files.")
    parser.add_argument("--contract", help="Path to deploy_contract.yaml to validate")
    parser.add_argument("--schema", help="Path to deploy_contract.schema.json (auto-detected if omitted)")
    parser.add_argument("--validate-examples", action="store_true", help="Validate all deploy contract examples")
    parser.add_argument("--validate-all", action="store_true", help="Validate all known deploy contracts")
    args = parser.parse_args()

    if args.validate_all:
        cmd_validate_all()
    elif args.validate_examples:
        cmd_validate_examples()
    elif args.contract:
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
