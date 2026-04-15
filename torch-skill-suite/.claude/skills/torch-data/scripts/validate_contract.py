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
import math
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
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    if yaml is not None:
        return yaml.safe_load(text)
    return SimpleYAMLParser(text).parse()


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def path_to_string(path_parts):
    if not path_parts:
        return "<root>"
    parts = []
    for part in path_parts:
        if isinstance(part, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{part}]"
            else:
                parts.append(f"[{part}]")
        else:
            parts.append(str(part))
    return ".".join(parts)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_split_entry(entry):
    if isinstance(entry, str):
        return "string"
    if isinstance(entry, dict):
        return "object"
    if is_number(entry):
        return "number"
    return type(entry).__name__


def validate_positive_integer(value, field_name, errors, allow_negative_one=False):
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{field_name}: expected integer")
        return
    if allow_negative_one and value == -1:
        return
    if value < 1:
        errors.append(f"{field_name}: must be >= 1")


def validate_positive_number(value, field_name, errors):
    if value is None:
        return
    if not is_number(value):
        errors.append(f"{field_name}: expected number")
        return
    if value <= 0:
        errors.append(f"{field_name}: must be > 0")


def validate_positive_shape(values, field_name, errors):
    if values is None:
        return
    if not isinstance(values, list):
        errors.append(f"{field_name}: expected array")
        return
    for idx, item in enumerate(values):
        if not isinstance(item, int) or isinstance(item, bool):
            errors.append(f"{field_name}[{idx}]: expected integer")
        elif item < 1:
            errors.append(f"{field_name}[{idx}]: must be >= 1")


def semantic_validate_contract(contract_data, strict_handoff=False):
    errors = []

    if not isinstance(contract_data, dict):
        return ["<root>: contract must be a mapping/object"]

    input_spec = contract_data.get("input_spec") if isinstance(contract_data.get("input_spec"), dict) else None
    output_spec = contract_data.get("output_spec") if isinstance(contract_data.get("output_spec"), dict) else None
    splits = contract_data.get("splits") if isinstance(contract_data.get("splits"), dict) else None
    inferred_format_spec = contract_data.get("inferred_format_spec") if isinstance(contract_data.get("inferred_format_spec"), dict) else None
    user_format_spec = contract_data.get("user_format_spec") if isinstance(contract_data.get("user_format_spec"), dict) else None

    if input_spec is not None:
        validate_positive_shape(input_spec.get("shape"), "input_spec.shape", errors)
        validate_positive_integer(input_spec.get("vocab_size"), "input_spec.vocab_size", errors)
        validate_positive_integer(input_spec.get("num_features"), "input_spec.num_features", errors)
        validate_positive_integer(input_spec.get("sample_rate"), "input_spec.sample_rate", errors)
        validate_positive_integer(input_spec.get("fps"), "input_spec.fps", errors)
        validate_positive_integer(input_spec.get("sequence_length"), "input_spec.sequence_length", errors, allow_negative_one=True)
        validate_positive_number(input_spec.get("duration"), "input_spec.duration", errors)

    if output_spec is not None:
        validate_positive_integer(output_spec.get("num_classes"), "output_spec.num_classes", errors)
        validate_positive_shape(output_spec.get("mask_shape"), "output_spec.mask_shape", errors)
        validate_positive_integer(output_spec.get("output_dim"), "output_spec.output_dim", errors)

        label_map = output_spec.get("label_map")
        num_classes = output_spec.get("num_classes")
        if label_map is not None:
            if not isinstance(label_map, dict):
                errors.append("output_spec.label_map: expected object")
            else:
                parsed_keys = []
                for key in label_map.keys():
                    try:
                        parsed_keys.append(int(key))
                    except (TypeError, ValueError):
                        errors.append(f"output_spec.label_map.{key}: key must be integer-like")
                if parsed_keys:
                    expected_keys = list(range(len(parsed_keys)))
                    if sorted(parsed_keys) != expected_keys:
                        errors.append("output_spec.label_map: keys must be contiguous starting at 0")
                if isinstance(num_classes, int) and not isinstance(num_classes, bool) and len(label_map) != num_classes:
                    errors.append(
                        f"output_spec.label_map: expected {num_classes} entries to match output_spec.num_classes, got {len(label_map)}"
                    )

    if splits is not None:
        numeric_values = []
        split_kinds = set()
        for split_name in ("train", "val", "test"):
            if split_name not in splits:
                continue
            entry = splits[split_name]
            kind = normalize_split_entry(entry)
            split_kinds.add(kind)
            if kind == "number":
                numeric_values.append((split_name, entry))
                if entry <= 0 or entry > 1:
                    errors.append(f"splits.{split_name}: numeric split must be > 0 and <= 1")
        if "number" in split_kinds and len(split_kinds - {"number"}) > 0:
            errors.append("splits: mixed numeric and path/object split definitions are not allowed")
        if len(numeric_values) > 1:
            total = sum(value for _, value in numeric_values)
            if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
                errors.append(f"splits: numeric split proportions must sum to 1.0, got {total:.6f}")

    if contract_data.get("data_type") == "multimodal" and inferred_format_spec is not None:
        details = inferred_format_spec.get("details")
        if isinstance(details, dict) and "modalities" in details:
            modalities = details.get("modalities")
            if not isinstance(modalities, list) or len(modalities) < 2:
                errors.append("inferred_format_spec.details.modalities: multimodal contracts require at least two modalities")

    if strict_handoff and contract_data.get("data_format_option") == "auto_inferred":
        if inferred_format_spec is None:
            errors.append("inferred_format_spec: required when --strict-handoff is enabled for auto_inferred contracts")
        else:
            format_type = inferred_format_spec.get("format_type")
            details = inferred_format_spec.get("details")
            if not isinstance(format_type, str) or not format_type.strip():
                errors.append("inferred_format_spec.format_type: required when --strict-handoff is enabled")
            if not isinstance(details, dict) or not details:
                errors.append("inferred_format_spec.details: non-empty object required when --strict-handoff is enabled")

    if strict_handoff and contract_data.get("data_format_option") == "user_provided" and user_format_spec is not None:
        format_type = user_format_spec.get("format_type")
        details = user_format_spec.get("details")
        if not isinstance(format_type, str) or not format_type.strip():
            errors.append("user_format_spec.format_type: required when --strict-handoff is enabled")
        if not isinstance(details, dict) or not details:
            errors.append("user_format_spec.details: non-empty object required when --strict-handoff is enabled")

    return errors


def _fallback_validate_contract(contract_data):
    errors = []

    if not isinstance(contract_data, dict):
        return False, ["<root>: contract must be a mapping/object"]

    required_fields = ["data_type", "task_type", "input_spec", "output_spec", "splits", "data_format_option"]
    for field in required_fields:
        if field not in contract_data:
            errors.append(f"{field}: missing required field")

    data_type = contract_data.get("data_type")
    if data_type is not None and data_type not in VALID_DATA_TYPES:
        errors.append(f"data_type: invalid value {data_type}")

    task_type = contract_data.get("task_type")
    if task_type is not None and task_type not in VALID_TASK_TYPES:
        errors.append(f"task_type: invalid value {task_type}")

    input_spec = contract_data.get("input_spec")
    if input_spec is not None and not isinstance(input_spec, dict):
        errors.append("input_spec: must be an object")

    output_spec = contract_data.get("output_spec")
    if output_spec is not None:
        if not isinstance(output_spec, dict):
            errors.append("output_spec: must be an object")
        else:
            output_type = output_spec.get("type")
            if output_type is not None and output_type not in VALID_OUTPUT_TYPES:
                errors.append(f"output_spec.type: invalid value {output_type}")
            bbox_format = output_spec.get("bbox_format")
            if bbox_format is not None and bbox_format not in VALID_BBOX_FORMATS:
                errors.append(f"output_spec.bbox_format: invalid value {bbox_format}")

    splits = contract_data.get("splits")
    if splits is not None:
        if not isinstance(splits, dict):
            errors.append("splits: must be an object")
        elif "train" not in splits:
            errors.append("splits.train: missing required field")

    preprocessing = contract_data.get("preprocessing")
    if preprocessing is not None:
        if not isinstance(preprocessing, list):
            errors.append("preprocessing: must be an array")
        else:
            for idx, step in enumerate(preprocessing):
                if not isinstance(step, dict):
                    errors.append(f"preprocessing[{idx}]: must be an object")
                    continue
                if "name" not in step:
                    errors.append(f"preprocessing[{idx}].name: missing required field")
                if "params" in step and not isinstance(step["params"], dict):
                    errors.append(f"preprocessing[{idx}].params: must be an object")

    data_format_option = contract_data.get("data_format_option")
    if data_format_option not in {None, "user_provided", "auto_inferred"}:
        errors.append(f"data_format_option: invalid value {data_format_option}")

    user_format_spec = contract_data.get("user_format_spec")
    inferred_format_spec = contract_data.get("inferred_format_spec")

    for field_name, value in (("user_format_spec", user_format_spec), ("inferred_format_spec", inferred_format_spec)):
        if value is not None:
            if not isinstance(value, dict):
                errors.append(f"{field_name}: must be an object")
            else:
                if "format_type" not in value:
                    errors.append(f"{field_name}.format_type: missing required field")
                if "details" not in value:
                    errors.append(f"{field_name}.details: missing required field")
                elif not isinstance(value["details"], dict):
                    errors.append(f"{field_name}.details: must be an object")

    if data_format_option == "user_provided":
        if user_format_spec is None:
            errors.append("user_format_spec: required when data_format_option=user_provided")
        if inferred_format_spec is not None:
            errors.append("inferred_format_spec: must not be present when data_format_option=user_provided")
    elif data_format_option == "auto_inferred":
        if user_format_spec is not None:
            errors.append("user_format_spec: must not be present when data_format_option=auto_inferred")

    if task_type == "classification" and isinstance(output_spec, dict):
        if output_spec.get("type") != "categorical":
            errors.append("output_spec.type: classification requires categorical")
        if "num_classes" not in output_spec:
            errors.append("output_spec.num_classes: required for classification")

    if task_type == "detection" and isinstance(output_spec, dict):
        if output_spec.get("type") != "bounding_box":
            errors.append("output_spec.type: detection requires bounding_box")
        if "bbox_format" not in output_spec:
            errors.append("output_spec.bbox_format: required for detection")

    if task_type == "segmentation" and isinstance(output_spec, dict):
        if output_spec.get("type") != "mask":
            errors.append("output_spec.type: segmentation requires mask")
        if "mask_shape" not in output_spec:
            errors.append("output_spec.mask_shape: required for segmentation")
        if "num_classes" not in output_spec:
            errors.append("output_spec.num_classes: required for segmentation")

    if task_type == "regression" and isinstance(output_spec, dict):
        if output_spec.get("type") != "continuous":
            errors.append("output_spec.type: regression requires continuous")
        if "output_dim" not in output_spec:
            errors.append("output_spec.output_dim: required for regression")

    if data_type == "tabular" and isinstance(input_spec, dict) and "num_features" not in input_spec:
        errors.append("input_spec.num_features: required for tabular data")

    if data_type == "audio" and isinstance(input_spec, dict) and "sample_rate" not in input_spec:
        errors.append("input_spec.sample_rate: required for audio data")

    if data_type == "video" and isinstance(input_spec, dict):
        if "fps" not in input_spec and "shape" not in input_spec:
            errors.append("input_spec: video data requires fps or shape")

    return len(errors) == 0, errors


def collect_schema_errors(contract_data, schema_data):
    if jsonschema is None:
        return []

    try:
        validator = jsonschema.Draft202012Validator(schema_data)
    except jsonschema.exceptions.SchemaError as e:
        return [f"<schema>: invalid schema - {e.message}"]

    errors = []
    for error in sorted(validator.iter_errors(contract_data), key=lambda err: list(err.absolute_path)):
        path = path_to_string(error.absolute_path)
        errors.append(f"{path}: {error.message}")
    return errors


def validate_contract(contract_data, schema_data, strict_handoff=False):
    """Validate contract against schema and semantic rules, return (is_valid, errors)."""
    if jsonschema is not None:
        errors = collect_schema_errors(contract_data, schema_data)
    else:
        _, errors = _fallback_validate_contract(contract_data)

    errors.extend(semantic_validate_contract(contract_data, strict_handoff=strict_handoff))

    deduped_errors = []
    seen = set()
    for error in errors:
        if error not in seen:
            deduped_errors.append(error)
            seen.add(error)
    return len(deduped_errors) == 0, deduped_errors


def resolve_default_schema_path(script_dir):
    schema_path = script_dir / "../../../../shared/schemas/data_contract.schema.json"
    if schema_path.exists():
        return schema_path

    fallback = Path("data_contract.schema.json")
    if fallback.exists():
        return fallback

    return None


def validate_path(contract_path, schema, strict_handoff=False):
    contract = load_yaml(contract_path)
    return validate_contract(contract, schema, strict_handoff=strict_handoff)


def validate_example_suite(examples_root, schema, strict_handoff=False):
    example_files = sorted(examples_root.glob("*/data_contract.yaml"))
    if not example_files:
        return False, [f"No example contracts found under: {examples_root}"]

    failures = []
    for example_file in example_files:
        is_valid, errors = validate_path(example_file, schema, strict_handoff=strict_handoff)
        if is_valid:
            print(f"✓ {example_file}")
        else:
            failures.append((str(example_file), errors))
            print(f"✗ {example_file}")
            for err in errors:
                print(f"  - {err}")

    return len(failures) == 0, failures


def validate_shared_catalog(catalog_path, schema, strict_handoff=False):
    catalog = load_yaml(catalog_path)
    if not isinstance(catalog, dict) or not catalog:
        return False, [(str(catalog_path), ["Shared catalog is empty or not a mapping."])]

    failures = []
    for example_name, contract in catalog.items():
        is_valid, errors = validate_contract(contract, schema, strict_handoff=strict_handoff)
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
    parser.add_argument("--strict-handoff", action="store_true", help="Require downstream-ready contracts, including inferred_format_spec for auto_inferred entries")
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

        is_valid, errors = validate_path(contract_path, schema, strict_handoff=args.strict_handoff)
        if is_valid:
            print("✓ Contract is valid.")
        else:
            success = False
            print("✗ Contract validation failed:")
            for err in errors:
                print(f"  - {err}")

    if args.validate_examples or args.validate_all:
        examples_root = script_dir.parent / "examples"
        is_valid, failures = validate_example_suite(examples_root, schema, strict_handoff=args.strict_handoff)
        success = success and is_valid
        if is_valid:
            print("✓ All standalone example contracts are valid.")
        else:
            print(f"✗ Standalone example validation failed for {len(failures)} example(s).")

    if args.validate_shared_catalog or args.validate_all:
        catalog_path = script_dir / "../../../../shared/contracts/data_contract.example.yaml"
        is_valid, failures = validate_shared_catalog(catalog_path, schema, strict_handoff=args.strict_handoff)
        success = success and is_valid
        if is_valid:
            print("✓ All shared catalog examples are valid.")
        else:
            print(f"✗ Shared catalog validation failed for {len(failures)} example(s).")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
