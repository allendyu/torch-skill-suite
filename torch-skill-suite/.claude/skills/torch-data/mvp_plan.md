# torch-data MVP Implementation Plan

## Goal

Create a minimal viable `torch-data` skill that can:
1. Handle multiple data types (image, text, time_series, tabular, audio, video) and task types (classification, detection, segmentation, regression, etc.)
2. Support two data format options: user‑provided and auto‑inferred
3. Generate a `data_contract.yaml` that can be consumed by downstream skills (`torch-model`, `torch-train`)

## Directory Structure (after MVP)

```
.claude/skills/torch-data/
├── SKILL.md                         # Formal skill description (already done)
├── mvp_plan.md                      # This file
├── references/
│   ├── data_types.md               # Reference for supported data types
│   ├── task_types.md               # Reference for supported task types
│   └── data_contract_guide.md      # Step‑by‑step guide to fill data contract
├── examples/
│   ├── image_classification_user/
│   │   └── data_contract.yaml      # Example: image classification with user format
│   ├── text_classification_auto/
│   │   └── data_contract.yaml      # Example: text classification with auto‑inferred format
│   ├── time_series_regression_user/
│   │   └── data_contract.yaml      # Example: time‑series regression with user format
│   └── object_detection_auto/
│       └── data_contract.yaml      # Example: object detection with auto‑inferred format
└── scripts/
    ├── validate_contract.py        # Validate a data_contract.yaml against schema
    └── inspect_dataset.py          # Infer dataset format (auto‑inferred mode)
```

## Shared Schema Updates

The shared schema `shared/schemas/data_contract.schema.json` has been extended to:

- Add `data_type` and `task_type` enumerations
- Make `input_spec` and `output_spec` flexible objects with modality‑specific fields
- Introduce `data_format_option` (`user_provided` / `auto_inferred`) with corresponding `user_format_spec` and `inferred_format_spec`
- Keep `splits`, `preprocessing`, `metadata` as generic fields

The example contract `shared/contracts/data_contract.example.yaml` now contains five sub‑examples illustrating different data/task combinations and format options.

## First‑Batch Files Created

### 1. References
- `data_types.md` – details on image, text, time_series, tabular, audio, video, multimodal
- `task_types.md` – details on classification, detection, segmentation, regression, generation, translation, clustering, RL
- `data_contract_guide.md` – practical walkthrough of each field in the contract

### 2. Examples
Four concrete examples showing realistic use‑cases:

- **Image classification (user‑provided)**: ImageFolder‑style dataset with class folders
- **Text classification (auto‑inferred)**: CSV files with text and label columns
- **Time‑series regression (user‑provided)**: NPZ file with `X` and `y` arrays
- **Object detection (auto‑inferred)**: YOLO‑style directory with images and label `.txt` files

Each example includes a complete `data_contract.yaml` that can be used as a template.

### 3. Scripts
- `validate_contract.py` – command‑line tool to validate a YAML contract against the JSON schema
- `inspect_dataset.py` – prototype script that attempts to guess the format of a dataset (supports image classification, text classification, time series, detection)

## Next Steps for torch-data

1. **Extend inspection script** to cover more data types and formats (tabular, audio, video, segmentation, etc.)
2. **Add code‑generation script** that takes a `data_contract.yaml` and produces a PyTorch `Dataset` class + `DataLoader` setup.
3. **Integrate with skill logic** – when the skill is triggered, it should:
   - Ask for `data_type` / `task_type`
   - Ask for `data_format_option`
   - If `user_provided`, guide the user through `user_format_spec`
   - If `auto_inferred`, run inspection and present the inferred format for confirmation
   - Generate the final `data_contract.yaml`
   - Optionally generate Dataset/DataLoader code

4. **Add more examples** for segmentation, audio classification, video action recognition, multimodal tasks.

5. **Improve schema validation** with conditional fields (e.g., `bbox_format` required only when `task_type=detection`).

## Collaboration with Other Skills

Once `torch-data` produces a valid `data_contract.yaml`, the workflow can continue:

- `torch-model` will read `data_contract.yaml` to understand input shape, output dimensions, and task type, then generate an appropriate model.
- `torch-train` will consume both `data_contract.yaml` and `model_contract.yaml` to build a training loop.
- `torch-eval-tune` will use the same contract to define evaluation metrics.
- `torch-infer-deploy` will rely on the contract for preprocessing/postprocessing consistency.

## Verification Checklist

- [x] Updated `data_contract.schema.json`
- [x] Updated `data_contract.example.yaml`
- [x] Created `references/` with three guides
- [x] Created `examples/` with four realistic contracts
- [x] Created `scripts/` with validation and inspection tools
- [ ] Test validation script on example contracts
- [ ] Test inspection script on dummy datasets
- [ ] Ensure schema covers all required fields for each data/task combination

## Notes

- The MVP deliberately keeps `input_spec` and `output_spec` flexible; later iterations can tighten the schema with `oneOf` or `if/then` rules.
- The `auto_inferred` path is a convenience feature; users can always switch to `user_provided` if the inference is incorrect.
- All documentation is bilingual (English + Chinese) in the skill description; reference files are currently English‑only but can be translated if needed.