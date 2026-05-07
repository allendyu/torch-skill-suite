import subprocess
import sys
from pathlib import Path

VALIDATE_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_contract.py"
SCHEMA_PATH = Path(__file__).resolve().parents[4] / "shared" / "schemas" / "data_contract.schema.json"
EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "examples"


def run_validate(args, expect_success):
    proc = subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), *args],
        capture_output=True,
        text=True,
    )
    if expect_success:
        assert proc.returncode == 0, proc.stdout + proc.stderr
    else:
        assert proc.returncode != 0, proc.stdout + proc.stderr
    return proc


def test_validate_examples_pass_by_default():
    proc = run_validate(["--validate-examples", "--schema", str(SCHEMA_PATH)], expect_success=True)
    assert "All standalone example contracts are valid" in proc.stdout


def test_validate_shared_examples_pass():
    proc = run_validate(["--validate-shared-examples", "--schema", str(SCHEMA_PATH)], expect_success=True)
    assert "All shared data contract examples are valid" in proc.stdout


def test_reports_multiple_errors(tmp_path):
    contract = tmp_path / "bad.yaml"
    contract.write_text(
        """
        data_type: image
        task_type: classification
        input_spec:
          shape: [3, 0, 224]
        output_spec:
          type: categorical
          num_classes: 0
          label_map:
            0: cat
            2: dog
        splits:
          train: 1.2
          val: 0.2
        data_format_option: auto_inferred
        """,
        encoding="utf-8",
    )
    proc = run_validate(["--contract", str(contract), "--schema", str(SCHEMA_PATH)], expect_success=False)
    assert "output_spec.num_classes" in proc.stdout
    assert "input_spec.shape" in proc.stdout
    assert "splits.train" in proc.stdout


def test_reports_label_map_num_classes_mismatch(tmp_path):
    contract = tmp_path / "label_map_mismatch.yaml"
    contract.write_text(
        """
        data_type: text
        task_type: classification
        input_spec:
          sequence_length: 32
          vocab_size: 100
          dtype: int64
        output_spec:
          type: categorical
          num_classes: 3
          label_map:
            0: negative
            1: positive
        splits:
          train: data/train.csv
          val: data/val.csv
        data_format_option: auto_inferred
        """,
        encoding="utf-8",
    )
    proc = run_validate(["--contract", str(contract), "--schema", str(SCHEMA_PATH)], expect_success=False)
    assert "output_spec.label_map" in proc.stdout
    assert "expected 3 entries" in proc.stdout


def test_strict_handoff_requires_inferred_format_spec(tmp_path):
    contract = tmp_path / "draft.yaml"
    contract.write_text(
        """
        data_type: text
        task_type: classification
        input_spec:
          sequence_length: 32
          vocab_size: 100
          dtype: int64
        output_spec:
          type: categorical
          num_classes: 2
        splits:
          train: data/train.csv
          val: data/val.csv
        data_format_option: auto_inferred
        """,
        encoding="utf-8",
    )
    run_validate(["--contract", str(contract), "--schema", str(SCHEMA_PATH)], expect_success=True)
    proc = run_validate(["--contract", str(contract), "--schema", str(SCHEMA_PATH), "--strict-handoff"], expect_success=False)
    assert "inferred_format_spec" in proc.stdout


def test_numeric_split_proportions_must_sum_to_one(tmp_path):
    contract = tmp_path / "bad_splits.yaml"
    contract.write_text(
        """
        data_type: tabular
        task_type: regression
        input_spec:
          num_features: 4
          dtype: float32
        output_spec:
          type: continuous
          output_dim: 1
        splits:
          train: 0.5
          val: 0.2
          test: 0.2
        data_format_option: user_provided
        user_format_spec:
          format_type: CSV
          details:
            target_column: y
        """,
        encoding="utf-8",
    )
    proc = run_validate(["--contract", str(contract), "--schema", str(SCHEMA_PATH)], expect_success=False)
    assert "numeric split proportions must sum to 1.0" in proc.stdout


def test_mixed_split_styles_are_rejected(tmp_path):
    contract = tmp_path / "mixed.yaml"
    contract.write_text(
        """
        data_type: tabular
        task_type: regression
        input_spec:
          num_features: 4
          dtype: float32
        output_spec:
          type: continuous
          output_dim: 1
        splits:
          train: 0.8
          val: data/val.csv
        data_format_option: user_provided
        user_format_spec:
          format_type: CSV
          details:
            target_column: y
        """,
        encoding="utf-8",
    )
    proc = run_validate(["--contract", str(contract), "--schema", str(SCHEMA_PATH)], expect_success=False)
    assert "mixed numeric and path/object split definitions" in proc.stdout
