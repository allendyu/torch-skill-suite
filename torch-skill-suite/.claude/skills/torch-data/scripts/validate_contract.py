#!/usr/bin/env python3
"""
Validate data contract YAML files against the JSON schema.

Usage:
    python validate_contract.py --contract path/to/data_contract.yaml
    python validate_contract.py --contract path/to/data_contract.yaml --schema path/to/schema.json
    python validate_contract.py --validate-examples
    python validate_contract.py --validate-shared-catalog
    python validate_contract.py --validate-all

If optional YAML/schema packages are unavailable, the script falls back to a
built-in YAML subset parser and built-in validation for the current
`data_contract` structure.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    yaml = None

try:
    import jsonschema  # type: ignore
    import jsonschema.exceptions  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    jsonschema = None


VALID_DATA_TYPES = {"image", "text", "time_series", "tabular", "audio", "video", "multimodal"}
VALID_TASK_TYPES = {
    "classification",
    "detection",
    "segmentation",
    "regression",
    "generation",
    "translation",
    "clustering",
    "reinforcement_learning",
}
VALID_OUTPUT_TYPES = {"categorical", "continuous", "bounding_box", "mask", "sequence", "multiple"}
VALID_BBOX_FORMATS = {"xyxy", "xywh", "cxcywh"}


class SimpleYAMLParser:
    def __init__(self, text):
        self.lines = self._prepare_lines(text)

    def _strip_inline_comment(self, line):
        in_single = False
        in_double = False
        escaped = False
        for i, ch in enumerate(line):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
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
            cleaned = self._strip_inline_comment(raw_line)
            if not cleaned.strip():
                continue
            indent = len(cleaned) - len(cleaned.lstrip(" "))
            prepared.append((indent, cleaned.lstrip(" ")))
        return prepared

    def parse(self):
        if not self.lines:
            return None
        value, index = self._parse_block(0, self.lines[0][0])
        if index != len(self.lines):
            raise ValueError("Unexpected trailing YAML content.")
        return value

    def _parse_block(self, index, indent):
        if index >= len(self.lines):
            return None, index
        current_indent, content = self.lines[index]
        if current_indent != indent:
            raise ValueError(f"Unexpected indentation at line {index + 1}: {current_indent}, expected {indent}")
        if content.startswith("- "):
            return self._parse_sequence(index, indent)
        return self._parse_mapping(index, indent)

    def _parse_mapping(self, index, indent):
        mapping = {}
        while index < len(self.lines):
            current_indent, content = self.lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in mapping at line {index + 1}")
            if content.startswith("- "):
                break

            key, rest = self._split_key_value(content)
            key = self._parse_key(key)
            index += 1

            if rest == "":
                if index < len(self.lines) and self.lines[index][0] > indent:
                    value, index = self._parse_block(index, self.lines[index][0])
                else:
                    value = {}
            else:
                value = self._parse_value(rest)

            mapping[key] = value
        return mapping, index

    def _parse_sequence(self, index, indent):
        sequence = []
        while index < len(self.lines):
            current_indent, content = self.lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in sequence at line {index + 1}")
            if not content.startswith("- "):
                break

            item_text = content[2:].strip()
            index += 1

            if item_text == "":
                if index < len(self.lines) and self.lines[index][0] > indent:
                    item, index = self._parse_block(index, self.lines[index][0])
                else:
                    item = None
            elif self._has_top_level_colon(item_text):
                key, rest = self._split_key_value(item_text)
                item = {}
                key = self._parse_key(key)
                if rest == "":
                    if index < len(self.lines) and self.lines[index][0] > indent:
                        value, index = self._parse_block(index, self.lines[index][0])
                    else:
                        value = {}
                else:
                    value = self._parse_value(rest)
                item[key] = value

                if index < len(self.lines) and self.lines[index][0] > indent:
                    extra, index = self._parse_mapping(index, self.lines[index][0])
                    if not isinstance(extra, dict):
                        raise ValueError("Expected mapping continuation for sequence item.")
                    item.update(extra)
            else:
                item = self._parse_value(item_text)
                if index < len(self.lines) and self.lines[index][0] > indent:
                    nested, index = self._parse_block(index, self.lines[index][0])
                    if isinstance(item, dict) and isinstance(nested, dict):
                        item.update(nested)
                    else:
                        raise ValueError("Unsupported nested content under scalar sequence item.")

            sequence.append(item)
        return sequence, index

    def _split_top_level(self, text, delimiter=","):
        parts = []
        current = []
        depth_brace = 0
        depth_bracket = 0
        in_single = False
        in_double = False
        escaped = False

        for ch in text:
            if escaped:
                current.append(ch)
                escaped = False
                continue
            if ch == "\\":
                current.append(ch)
                escaped = True
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                current.append(ch)
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                current.append(ch)
                continue
            if not in_single and not in_double:
                if ch == "{":
                    depth_brace += 1
                elif ch == "}":
                    depth_brace -= 1
                elif ch == "[":
                    depth_bracket += 1
                elif ch == "]":
                    depth_bracket -= 1
                elif ch == delimiter and depth_brace == 0 and depth_bracket == 0:
                    parts.append("".join(current).strip())
                    current = []
                    continue
            current.append(ch)

        if current:
            parts.append("".join(current).strip())
        return parts

    def _has_top_level_colon(self, text):
        depth_brace = 0
        depth_bracket = 0
        in_single = False
        in_double = False
        escaped = False

        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if not in_single and not in_double:
                if ch == "{":
                    depth_brace += 1
                elif ch == "}":
                    depth_brace -= 1
                elif ch == "[":
                    depth_bracket += 1
                elif ch == "]":
                    depth_bracket -= 1
                elif ch == ":" and depth_brace == 0 and depth_bracket == 0:
                    return True
        return False

    def _split_key_value(self, text):
        depth_brace = 0
        depth_bracket = 0
        in_single = False
        in_double = False
        escaped = False

        for i, ch in enumerate(text):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                continue
            if not in_single and not in_double:
                if ch == "{":
                    depth_brace += 1
                elif ch == "}":
                    depth_brace -= 1
                elif ch == "[":
                    depth_bracket += 1
                elif ch == "]":
                    depth_bracket -= 1
                elif ch == ":" and depth_brace == 0 and depth_bracket == 0:
                    return text[:i].strip(), text[i + 1 :].strip()
        raise ValueError(f"Expected key/value pair, got: {text}")

    def _parse_key(self, text):
        value = self._parse_scalar(text)
        return value

    def _parse_inline_mapping(self, text):
        inner = text[1:-1].strip()
        if not inner:
            return {}
        result = {}
        for item in self._split_top_level(inner):
            key, value = self._split_key_value(item)
            result[self._parse_key(key)] = self._parse_value(value)
        return result

    def _parse_inline_sequence(self, text):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [self._parse_value(item) for item in self._split_top_level(inner)]

    def _parse_scalar(self, text):
        text = text.strip()
        if text == "":
            return ""
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            try:
                return ast.literal_eval(text)
            except Exception:
                return text[1:-1]
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", text):
            return float(text)
        return text

    def _parse_value(self, text):
        text = text.strip()
        if text.startswith("{") and text.endswith("}"):
            return self._parse_inline_mapping(text)
        if text.startswith("[") and text.endswith("]"):
            return self._parse_inline_sequence(text)
        return self._parse_scalar(text)



def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return SimpleYAMLParser(text).parse()



def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)



def _fallback_validate_contract(contract_data):
    errors = []

    if not isinstance(contract_data, dict):
        return False, ["Contract must be a mapping/object."]

    required_fields = ["data_type", "task_type", "input_spec", "output_spec", "splits", "data_format_option"]
    for field in required_fields:
        if field not in contract_data:
            errors.append(f"Missing required field: {field}")

    data_type = contract_data.get("data_type")
    if data_type is not None and data_type not in VALID_DATA_TYPES:
        errors.append(f"Invalid data_type: {data_type}")

    task_type = contract_data.get("task_type")
    if task_type is not None and task_type not in VALID_TASK_TYPES:
        errors.append(f"Invalid task_type: {task_type}")

    input_spec = contract_data.get("input_spec")
    if input_spec is not None and not isinstance(input_spec, dict):
        errors.append("input_spec must be an object.")

    output_spec = contract_data.get("output_spec")
    if output_spec is not None:
        if not isinstance(output_spec, dict):
            errors.append("output_spec must be an object.")
        else:
            output_type = output_spec.get("type")
            if output_type is not None and output_type not in VALID_OUTPUT_TYPES:
                errors.append(f"Invalid output_spec.type: {output_type}")
            bbox_format = output_spec.get("bbox_format")
            if bbox_format is not None and bbox_format not in VALID_BBOX_FORMATS:
                errors.append(f"Invalid output_spec.bbox_format: {bbox_format}")

    splits = contract_data.get("splits")
    if splits is not None:
        if not isinstance(splits, dict):
            errors.append("splits must be an object.")
        elif "train" not in splits:
            errors.append("splits must include train.")

    preprocessing = contract_data.get("preprocessing")
    if preprocessing is not None:
        if not isinstance(preprocessing, list):
            errors.append("preprocessing must be an array.")
        else:
            for idx, step in enumerate(preprocessing):
                if not isinstance(step, dict):
                    errors.append(f"preprocessing[{idx}] must be an object.")
                    continue
                if "name" not in step:
                    errors.append(f"preprocessing[{idx}] is missing required field: name")
                if "params" in step and not isinstance(step["params"], dict):
                    errors.append(f"preprocessing[{idx}].params must be an object.")

    data_format_option = contract_data.get("data_format_option")
    if data_format_option not in {None, "user_provided", "auto_inferred"}:
        errors.append(f"Invalid data_format_option: {data_format_option}")

    user_format_spec = contract_data.get("user_format_spec")
    inferred_format_spec = contract_data.get("inferred_format_spec")

    for field_name, value in (("user_format_spec", user_format_spec), ("inferred_format_spec", inferred_format_spec)):
        if value is not None:
            if not isinstance(value, dict):
                errors.append(f"{field_name} must be an object.")
            else:
                if "format_type" not in value:
                    errors.append(f"{field_name} is missing required field: format_type")
                if "details" not in value:
                    errors.append(f"{field_name} is missing required field: details")
                elif not isinstance(value["details"], dict):
                    errors.append(f"{field_name}.details must be an object.")

    if data_format_option == "user_provided":
        if user_format_spec is None:
            errors.append("data_format_option=user_provided requires user_format_spec.")
        if inferred_format_spec is not None:
            errors.append("data_format_option=user_provided must not include inferred_format_spec.")
    elif data_format_option == "auto_inferred":
        if user_format_spec is not None:
            errors.append("data_format_option=auto_inferred must not include user_format_spec.")

    if task_type == "classification" and isinstance(output_spec, dict):
        if output_spec.get("type") != "categorical":
            errors.append("classification requires output_spec.type=categorical.")
        if "num_classes" not in output_spec:
            errors.append("classification requires output_spec.num_classes.")

    if task_type == "detection" and isinstance(output_spec, dict):
        if output_spec.get("type") != "bounding_box":
            errors.append("detection requires output_spec.type=bounding_box.")
        if "bbox_format" not in output_spec:
            errors.append("detection requires output_spec.bbox_format.")

    if task_type == "segmentation" and isinstance(output_spec, dict):
        if output_spec.get("type") != "mask":
            errors.append("segmentation requires output_spec.type=mask.")
        if "mask_shape" not in output_spec:
            errors.append("segmentation requires output_spec.mask_shape.")
        if "num_classes" not in output_spec:
            errors.append("segmentation requires output_spec.num_classes.")

    if task_type == "regression" and isinstance(output_spec, dict):
        if output_spec.get("type") != "continuous":
            errors.append("regression requires output_spec.type=continuous.")
        if "output_dim" not in output_spec:
            errors.append("regression requires output_spec.output_dim.")

    if data_type == "tabular" and isinstance(input_spec, dict) and "num_features" not in input_spec:
        errors.append("tabular data requires input_spec.num_features.")

    if data_type == "audio" and isinstance(input_spec, dict) and "sample_rate" not in input_spec:
        errors.append("audio data requires input_spec.sample_rate.")

    if data_type == "video" and isinstance(input_spec, dict):
        if "fps" not in input_spec and "shape" not in input_spec:
            errors.append("video data requires input_spec.fps or input_spec.shape.")

    return len(errors) == 0, errors



def validate_contract(contract_data, schema_data):
    """Validate contract against schema, return (is_valid, errors)."""
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=contract_data, schema=schema_data)
            return True, []
        except jsonschema.exceptions.ValidationError as e:
            return False, [str(e)]
        except jsonschema.exceptions.SchemaError as e:
            return False, [f"Schema error: {e}"]

    return _fallback_validate_contract(contract_data)



def resolve_default_schema_path(script_dir):
    schema_path = script_dir / "../../../../shared/schemas/data_contract.schema.json"
    if schema_path.exists():
        return schema_path

    fallback = Path("data_contract.schema.json")
    if fallback.exists():
        return fallback

    return None



def validate_path(contract_path, schema):
    contract = load_yaml(contract_path)
    return validate_contract(contract, schema)



def validate_example_suite(examples_root, schema):
    example_files = sorted(examples_root.glob("*/data_contract.yaml"))
    if not example_files:
        return False, [f"No example contracts found under: {examples_root}"]

    failures = []
    for example_file in example_files:
        is_valid, errors = validate_path(example_file, schema)
        if is_valid:
            print(f"✓ {example_file}")
        else:
            failures.append((str(example_file), errors))
            print(f"✗ {example_file}")
            for err in errors:
                print(f"  - {err}")

    return len(failures) == 0, failures



def validate_shared_catalog(catalog_path, schema):
    catalog = load_yaml(catalog_path)
    if not isinstance(catalog, dict) or not catalog:
        return False, [(str(catalog_path), ["Shared catalog is empty or not a mapping."])]

    failures = []
    for example_name, contract in catalog.items():
        is_valid, errors = validate_contract(contract, schema)
        if is_valid:
            print(f"✓ {catalog_path}::{example_name}")
        else:
            failures.append((f"{catalog_path}::{example_name}", errors))
            print(f"✗ {catalog_path}::{example_name}")
            for err in errors:
                print(f"  - {err}")

    return len(failures) == 0, failures



def main():
    parser = argparse.ArgumentParser(description="Validate data contract YAML files.")
    parser.add_argument("--contract", help="Path to one data_contract.yaml file")
    parser.add_argument("--schema", default=None, help="Path to JSON schema (default: use built-in)")
    parser.add_argument("--validate-examples", action="store_true", help="Validate all standalone example contracts")
    parser.add_argument("--validate-shared-catalog", action="store_true", help="Validate each example entry in the shared catalog")
    parser.add_argument("--validate-all", action="store_true", help="Validate standalone examples and shared catalog examples")
    args = parser.parse_args()

    if not any([args.contract, args.validate_examples, args.validate_shared_catalog, args.validate_all]):
        parser.error("Provide --contract or one of --validate-examples, --validate-shared-catalog, --validate-all")

    script_dir = Path(__file__).parent

    if args.schema:
        schema_path = Path(args.schema)
    else:
        schema_path = resolve_default_schema_path(script_dir)
        if schema_path is None:
            print("Error: Default schema not found. Please provide --schema.")
            sys.exit(1)

    if not schema_path.exists():
        print(f"Error: Schema file not found: {schema_path}")
        sys.exit(1)

    schema = load_json(schema_path)
    success = True

    if yaml is None:
        print("! Optional dependency 'PyYAML' is not installed; using built-in YAML subset parser.")
    if jsonschema is None:
        print("! Optional dependency 'jsonschema' is not installed; using built-in fallback validation.")

    if args.contract:
        contract_path = Path(args.contract)
        if not contract_path.exists():
            print(f"Error: Contract file not found: {contract_path}")
            sys.exit(1)

        is_valid, errors = validate_path(contract_path, schema)
        if is_valid:
            print("✓ Contract is valid.")
        else:
            success = False
            print("✗ Contract validation failed:")
            for err in errors:
                print(f"  - {err}")

    if args.validate_examples or args.validate_all:
        examples_root = script_dir.parent / "examples"
        is_valid, failures = validate_example_suite(examples_root, schema)
        success = success and is_valid
        if is_valid:
            print("✓ All standalone example contracts are valid.")
        else:
            print(f"✗ Standalone example validation failed for {len(failures)} example(s).")

    if args.validate_shared_catalog or args.validate_all:
        catalog_path = script_dir / "../../../../shared/contracts/data_contract.example.yaml"
        is_valid, failures = validate_shared_catalog(catalog_path, schema)
        success = success and is_valid
        if is_valid:
            print("✓ All shared catalog examples are valid.")
        else:
            print(f"✗ Shared catalog validation failed for {len(failures)} example(s).")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
