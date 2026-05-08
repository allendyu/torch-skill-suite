# Architecture

Torch Skill Suite is organized as six Claude Code skills connected by explicit YAML contracts. Each skill owns one stage of a PyTorch engineering workflow and hands off structured artifacts to the next stage.

## Skill boundaries

1. `torch-data` owns dataset inspection, preprocessing decisions, Dataset/DataLoader scaffolding (via `generate_dataset.py`), and `data_contract.yaml` generation.
2. `torch-model` consumes the data contract, resolves a route from `shared/route_map.yaml`, picks a backbone/head template, and produces `model_contract.yaml`.
3. `torch-train` consumes data and model contracts, then builds training loops, checkpointing, logging, and resume behavior via a `Trainer` class.
4. `torch-eval-tune` analyzes metrics, validation behavior, and tuning directions; emits `evaluation_report.yaml` and `tuning_plan.yaml`. It does not own the core training loop.
5. `torch-infer-deploy` packages trained models for local inference, TorchScript/ONNX export, FastAPI serving, and `deploy_contract.yaml`.
6. `torch-engineering` is currently a documentation stub: `SKILL.md` describes scope and boundaries, but no scripts, templates, or tests exist yet. It standardizes project structure, testing, CI, and maintainability concerns once activated.

## Contract layout

The authoritative schemas live in `shared/schemas/`:

- `project_spec.schema.json`
- `data_contract.schema.json`
- `model_contract.schema.json`
- `deploy_contract.schema.json`

Canonical checked-in examples live in `shared/contracts/` and use the `*.example.yaml` suffix. Skill-generated workspace artifacts normally use unsuffixed names such as `data_contract.yaml`, `model_contract.yaml`, and `deploy_contract.yaml`.

Scenario contracts (per-modality/task examples and recipes) live in `shared/examples/contracts/data/` and `shared/examples/contracts/model/`.

## Route map

`shared/route_map.yaml` is the authoritative `(data_type, task_type)` → model route table. Every entry declares:

- match conditions consumed from `data_contract.yaml`
- normalized task semantics (input/target/output)
- candidate backbones, default backbone, head type, and selection rules
- training compatibility (loss, target dtype, output activation)
- per-skill support status (`supported` / `partial` / `planned` / `unsupported`)
- maturity flags (contract / resolver / template / smoke test)
- priority bucket: `P0` (MVP) → `P3` (long-term)

Currently `supported` end-to-end across the five production skills:

- `image_classification` (P0)
- `text_classification`, `image_segmentation`, `tabular_classification`, `tabular_regression` (P1)

P2 routes (`image_detection`, `time_series_classification`, `time_series_regression`, `audio_classification`) and P3 routes (`video_classification`, `multimodal_classification`, `text_generation`, `time_series_forecasting`) have data-side support and route definitions, but no `torch-model` template implementations yet.

## Shared code

Reusable Python helpers live in `shared/python/torch_skill_shared/`:

- `yaml_utils` — YAML loading and emission with PyYAML fallback semantics
- `model_builder` — shared model construction and synthetic DataLoader factories used by all five production skills

Skill scripts add this package to `sys.path` via the package-level `conftest.py` and a per-script path bootstrap. Prefer these helpers over duplicating logic.

## Current validation status

As of 2026-05-08, the package-level test suite passes from the `torch-skill-suite/` directory with `199 passed`. The canonical data, model, deploy, and project-spec contract examples validate against their schemas. CI (`.github/workflows/ci.yml`) reproduces these checks on Python 3.12.

## Deployment export note

`torch-infer-deploy` currently supports TorchScript and ONNX export flows and the tests pass, but PyTorch emits deprecation warnings for `torch.jit.trace`, `torch.jit.save`, and `torch.jit.load`. Keep this path stable for current users while evaluating `torch.export` as the future export target.
