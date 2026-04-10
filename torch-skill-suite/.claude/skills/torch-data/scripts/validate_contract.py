#!/usr/bin/env python3
"""
Validate a data_contract.yaml file against the JSON schema.

Usage:
    python validate_contract.py --contract path/to/data_contract.yaml
    python validate_contract.py --contract path/to/data_contract.yaml --schema path/to/schema.json
"""

import argparse
import json
import sys
import os
from pathlib import Path
import yaml
import jsonschema
import jsonschema.exceptions

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_contract(contract_data, schema_data):
    """Validate contract against schema, return (is_valid, errors)."""
    try:
        jsonschema.validate(instance=contract_data, schema=schema_data)
        return True, []
    except jsonschema.exceptions.ValidationError as e:
        return False, [str(e)]
    except jsonschema.exceptions.SchemaError as e:
        return False, [f"Schema error: {e}"]

def main():
    parser = argparse.ArgumentParser(description="Validate a data contract YAML file.")
    parser.add_argument("--contract", required=True, help="Path to data_contract.yaml")
    parser.add_argument("--schema", default=None, help="Path to JSON schema (default: use built-in)")
    args = parser.parse_args()

    # Load contract
    if not os.path.exists(args.contract):
        print(f"Error: Contract file not found: {args.contract}")
        sys.exit(1)
    contract = load_yaml(args.contract)

    # Load schema
    if args.schema:
        schema_path = args.schema
    else:
        # Default to the shared schema in the suite
        script_dir = Path(__file__).parent
        schema_path = script_dir / "../../../shared/schemas/data_contract.schema.json"
        if not schema_path.exists():
            # Fallback to current directory
            schema_path = Path("data_contract.schema.json")
            if not schema_path.exists():
                print("Error: Default schema not found. Please provide --schema.")
                sys.exit(1)

    if not os.path.exists(schema_path):
        print(f"Error: Schema file not found: {schema_path}")
        sys.exit(1)
    schema = load_json(schema_path)

    # Validate
    is_valid, errors = validate_contract(contract, schema)
    if is_valid:
        print("✓ Contract is valid.")
        sys.exit(0)
    else:
        print("✗ Contract validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()