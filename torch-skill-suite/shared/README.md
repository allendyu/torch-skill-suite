# Shared Resources

Cross-skill resources used by every skill in the suite. All paths below are relative to `torch-skill-suite/`.

## Layout

```
shared/
├── schemas/           # JSON Schema files (authoritative contract definitions)
├── contracts/         # Canonical *.example.yaml schema-valid scaffolds (one per schema)
├── examples/contracts/  # Scenario contract examples by modality/task
├── python/torch_skill_shared/  # Reusable Python utilities consumed by every skill
└── route_map.yaml     # Authoritative (data_type, task_type) → model route mapping
```

## `schemas/`

JSON Schema files that define the contract surface between skills:

- `project_spec.schema.json` — global task definition (consumed optionally as the entry point)
- `data_contract.schema.json` — produced by `torch-data`, consumed by `torch-model` / `torch-train`
- `model_contract.schema.json` — produced by `torch-model`, consumed by `torch-train` / `torch-eval-tune` / `torch-infer-deploy`
- `deploy_contract.schema.json` — produced by `torch-infer-deploy`, captures runtime form

The schemas are the single source of truth for validation. Each skill ships a `validate_contract.py` that loads the relevant schema from this directory.

## `contracts/`

One canonical schema-valid example per schema, using the `*.example.yaml` suffix:

- `project_spec.example.yaml`
- `data_contract.example.yaml`
- `model_contract.example.yaml`
- `deploy_contract.example.yaml`

These are scaffolds: copy one, rename it to its operational name (`data_contract.yaml`, etc.), and edit. The `*.example.yaml` suffix lets these checked-in scaffolds coexist with workspace artifacts produced at runtime.

## `examples/contracts/`

Scenario-specific contract examples organized by skill stage:

- `data/` — one contract per modality/task combination (image segmentation, object detection, audio classification, video classification, multimodal paired classification, tabular classification, tabular regression, time-series regression, text classification)
- `model/` — backbone-specific model contracts (e.g., `image_classification_resnet18_model_contract.yaml`, `image_classification_efficientnet_b0_model_contract.yaml`)

Use these as recipes when building a contract for a real workspace.

## `python/torch_skill_shared/`

Python package with helpers consumed by every skill. Currently version `0.2.0`:

- `yaml_utils.py` — YAML loading and emission with PyYAML fallback semantics; used by every script that reads or writes a contract
- `model_builder.py` — shared model construction (`build_model_from_contract`) and synthetic DataLoader factories for the five supported routes; used by `torch-train`, `torch-eval-tune`, and `torch-infer-deploy`

The package is added to `sys.path` by the package-level `conftest.py` (for tests) and by an explicit path bootstrap in each skill's scripts (for CLI usage). When extending shared behavior, prefer adding to this package over duplicating logic in a single skill.

## `route_map.yaml`

Authoritative mapping from `(data_type, task_type)` to a model route. Each entry declares:

- match conditions consumed from `data_contract.yaml`
- normalized task semantics (input / target / output)
- candidate backbones, default backbone, head type, and selection rules
- training compatibility (loss, target dtype, output activation)
- per-skill support status (`supported` / `partial` / `planned` / `unsupported` / `not_applicable`)
- maturity flags (`contract`, `route_defined`, `model_resolver`, `template`, `smoke_test`)
- priority bucket: `P0` (current MVP) → `P3` (long-term)

Currently `supported` end-to-end across the five production skills:

- `image_classification` (P0)
- `text_classification`, `image_segmentation`, `tabular_classification`, `tabular_regression` (P1)

When adding a new route, update `route_map.yaml` first; downstream skills should match against this file, not against ad-hoc lists.

## Validation

From the package directory (`torch-skill-suite/`):

```bash
python .claude/skills/torch-data/scripts/validate_contract.py --contract shared/contracts/data_contract.example.yaml
python .claude/skills/torch-model/scripts/validate_contract.py --contract shared/contracts/model_contract.example.yaml
python .claude/skills/torch-infer-deploy/scripts/validate_contract.py --contract shared/contracts/deploy_contract.example.yaml
```

All four canonical examples (including `project_spec.example.yaml`) pass schema validation as of 2026-05-08.
