#!/usr/bin/env python3
"""
Validate model contract YAML files against the JSON schema.

Usage:
    python validate_contract.py --contract path/to/model_contract.yaml
    python validate_contract.py --contract path/to/model_contract.yaml --schema path/to/schema.json
    python validate_contract.py --validate-examples
    python validate_contract.py --validate-shared-examples

If optional YAML/jsonschema packages are unavailable, the script falls back to a
built-in YAML subset parser and built-in validation.
"""

import argparse
import json
import re
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


VALID_TASK_TYPES = {"classification", "detection", "segmentation", "regression", "generation", "translation"}
VALID_DATA_TYPES = {"image", "text", "time_series", "tabular", "audio", "video", "multimodal"}
VALID_TARGET_TYPES = {"categorical", "continuous", "bounding_box", "mask", "sequence"}
VALID_LATENCY_TIERS = {"fast", "balanced", "accurate"}


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
    """Built-in validation for model_contract structure."""
    errors = []

    # Required top-level fields
    for field in ["task_type", "data_type", "input_spec", "model_spec", "head_spec", "forward_spec"]:
        if field not in instance:
            errors.append(f"Missing required field: '{field}'")

    if "task_type" in instance and instance["task_type"] not in VALID_TASK_TYPES:
        errors.append(f"Invalid task_type '{instance.get('task_type')}'. Must be one of: {sorted(VALID_TASK_TYPES)}")

    if "data_type" in instance and instance["data_type"] not in VALID_DATA_TYPES:
        errors.append(f"Invalid data_type '{instance.get('data_type')}'. Must be one of: {sorted(VALID_DATA_TYPES)}")

    # input_spec validation
    input_spec = instance.get("input_spec", {})
    if isinstance(input_spec, dict):
        if "shape" not in input_spec:
            errors.append("input_spec: missing 'shape'")
        elif not isinstance(input_spec["shape"], list) or len(input_spec["shape"]) < 1:
            errors.append("input_spec.shape must be a non-empty list")
        if "dtype" not in input_spec:
            errors.append("input_spec: missing 'dtype'")

    # model_spec validation
    model_spec = instance.get("model_spec", {})
    if isinstance(model_spec, dict):
        for field in ["family", "architecture", "backbone", "pretrained", "in_channels"]:
            if field not in model_spec:
                errors.append(f"model_spec: missing '{field}'")
        if "in_channels" in model_spec and not isinstance(model_spec["in_channels"], int):
            errors.append("model_spec.in_channels must be an integer")

    # head_spec validation
    head_spec = instance.get("head_spec", {})
    if isinstance(head_spec, dict):
        if "type" not in head_spec:
            errors.append("head_spec: missing 'type'")

    # Conditional: classification requires num_classes
    if instance.get("task_type") == "classification":
        if isinstance(head_spec, dict) and "num_classes" not in head_spec:
            errors.append("head_spec: 'num_classes' is required for classification tasks")
        compat = instance.get("compatibility", {})
        if isinstance(compat, dict) and compat.get("expected_target_type") != "categorical":
            errors.append("compatibility.expected_target_type must be 'categorical' for classification")

    # Conditional: regression requires output_dim
    if instance.get("task_type") == "regression":
        if isinstance(head_spec, dict) and "output_dim" not in head_spec:
            errors.append("head_spec: 'output_dim' is required for regression tasks")

    # Conditional: image requires 3D shape
    if instance.get("data_type") == "image":
        if isinstance(input_spec, dict):
            shape = input_spec.get("shape", [])
            if isinstance(shape, list) and len(shape) != 3:
                errors.append(f"input_spec.shape must have 3 dimensions for image data, got {len(shape)}")

    # forward_spec validation
    forward_spec = instance.get("forward_spec", {})
    if isinstance(forward_spec, dict):
        if "output_shape" not in forward_spec:
            errors.append("forward_spec: missing 'output_shape'")

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
    """Find the model_contract schema file relative to this script."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / ".." / ".." / ".." / ".." / "shared" / "schemas" / "model_contract.schema.json",
        script_dir / ".." / ".." / "shared" / "schemas" / "model_contract.schema.json",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return None


def find_shared_contract():
    """Find the canonical shared model contract example."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / ".." / ".." / ".." / ".." / "shared" / "contracts" / "model_contract.example.yaml",
        script_dir / ".." / ".." / "shared" / "contracts" / "model_contract.example.yaml",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return None


def find_shared_examples_dir():
    """Find the shared scenario model contract examples directory."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / ".." / ".." / ".." / ".." / "shared" / "examples" / "contracts" / "model",
        script_dir / ".." / ".." / "shared" / "examples" / "contracts" / "model",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return str(resolved)
    return None


def find_examples_dir():
    """Find the examples directory for model contracts."""
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir / ".." / "examples" / "contracts"
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


def _validate_files(label, files):
    schema_path = find_schema()
    all_errors = []
    for example_file in files:
        errors = validate_contract(str(example_file), schema_path)
        if errors:
            all_errors.append((example_file.name, errors))
    if all_errors:
        print(f"Validation FAILED ({len(all_errors)} {label} file(s) with errors):")
        for name, errors in all_errors:
            print(f"  {name}:")
            for err in errors:
                print(f"    - {err}")
        return False
    print(f"All {len(files)} {label} file(s) passed validation.")
    return True


def cmd_validate_examples():
    examples_dir = find_examples_dir()
    if not examples_dir:
        print("No examples directory found.")
        sys.exit(1)
    example_files = sorted(Path(examples_dir).glob("*.yaml"))
    if not example_files:
        print(f"No YAML files found in {examples_dir}")
        sys.exit(1)
    sys.exit(0 if _validate_files("skill example", example_files) else 1)


def cmd_validate_shared_examples():
    files = []
    shared_contract = find_shared_contract()
    if shared_contract:
        files.append(Path(shared_contract))
    shared_examples_dir = find_shared_examples_dir()
    if shared_examples_dir:
        files.extend(sorted(Path(shared_examples_dir).glob("*.yaml")))
    if not files:
        print("No shared model contract examples found.")
        sys.exit(1)
    sys.exit(0 if _validate_files("shared example", files) else 1)


def cmd_validate_all():
    errors_occurred = False
    shared_contract = find_shared_contract()
    if shared_contract:
        print("--- Validating shared canonical example ---")
        errors = validate_contract(shared_contract, find_schema())
        if errors:
            print(f"  FAIL  {Path(shared_contract).name}: {errors[0]}")
            errors_occurred = True
        else:
            print(f"  PASS  {Path(shared_contract).name}")
    shared_examples_dir = find_shared_examples_dir()
    if shared_examples_dir:
        print("--- Validating shared scenario examples ---")
        for example_file in sorted(Path(shared_examples_dir).glob("*.yaml")):
            errors = validate_contract(str(example_file), find_schema())
            if errors:
                print(f"  FAIL  {example_file.name}: {errors[0]}")
                errors_occurred = True
            else:
                print(f"  PASS  {example_file.name}")
    examples_dir = find_examples_dir()
    if examples_dir:
        print("--- Validating skill examples ---")
        for example_file in sorted(Path(examples_dir).glob("*.yaml")):
            errors = validate_contract(str(example_file), find_schema())
            if errors:
                print(f"  FAIL  {example_file.name}: {errors[0]}")
                errors_occurred = True
            else:
                print(f"  PASS  {example_file.name}")
    if errors_occurred:
        sys.exit(1)
    else:
        print("All validations passed.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Validate model contract YAML files.")
    parser.add_argument("--contract", help="Path to model_contract.yaml to validate")
    parser.add_argument("--schema", help="Path to model_contract.schema.json (auto-detected if omitted)")
    parser.add_argument("--validate-examples", action="store_true", help="Validate all skill example contracts")
    parser.add_argument("--validate-shared-examples", action="store_true", help="Validate shared canonical and scenario examples")
    parser.add_argument("--validate-all", action="store_true", help="Validate all known model contracts")
    args = parser.parse_args()

    if args.validate_all:
        cmd_validate_all()
    elif args.validate_examples:
        cmd_validate_examples()
    elif args.validate_shared_examples:
        cmd_validate_shared_examples()
    elif args.contract:
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
