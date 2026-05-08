# Torch Skill Suite

A Claude Code skill suite for automating PyTorch deep learning engineering workflows.

## Included skills

- `torch-data` — data engineering (Dataset/DataLoader scaffolding, inspection, validation, code generation)
- `torch-model` — model construction (route resolution, backbone/head templates, smoke testing)
- `torch-train` — training loop generation (Trainer, optimizer/scheduler/loss, checkpointing)
- `torch-eval-tune` — evaluation and tuning suggestions (metrics, error analysis, tuning plans)
- `torch-infer-deploy` — inference and deployment (TorchScript/ONNX export, FastAPI serving, benchmarking)
- `torch-engineering` — engineering extension and standardization (currently `SKILL.md` only; no scripts yet)

## Shared resources

- `shared/schemas/` — JSON Schema files for contract validation
- `shared/contracts/` — canonical schema-valid `*.example.yaml` contract scaffolds (one per schema)
- `shared/examples/contracts/` — scenario contract examples by modality/task
- `shared/python/torch_skill_shared/` — reusable Python utilities (`yaml_utils`, `model_builder`)
- `shared/route_map.yaml` — authoritative `(data_type, task_type)` → model route mapping with priority and support status
- `docs/` — architecture, workflow, and MVP roadmap documentation

Runtime skill outputs normally use unsuffixed names such as `data_contract.yaml`, `model_contract.yaml`, and `deploy_contract.yaml`. The checked-in canonical examples use the `*.example.yaml` suffix so they can coexist with generated workspace artifacts.

## Recommended flow

1. `torch-data` → `data_contract.yaml`
2. `torch-model` → `model_contract.yaml`
3. `torch-train` → checkpoints + training history
4. `torch-eval-tune` → eval report + `tuning_plan.yaml`
5. `torch-infer-deploy` → exported model + serving artifacts + `deploy_contract.yaml`
6. `torch-engineering` → repository normalization (planned)

## Supported routes

The five production skills (`torch-data` → `torch-infer-deploy`) currently cover these routes end-to-end:

| Route | Priority | Status |
| --- | --- | --- |
| `image_classification` | P0 | supported |
| `text_classification` | P1 | supported |
| `image_segmentation` | P1 | supported |
| `tabular_classification` | P1 | supported |
| `tabular_regression` | P1 | supported |

P2/P3 routes (detection, time series, audio, video, multimodal, text generation, forecasting) have data-side coverage and route definitions but no model templates yet. See `shared/route_map.yaml` for the full priority matrix.

## Development checks

Run commands from this package directory:

```bash
python -m pytest
python .claude/skills/torch-data/scripts/validate_contract.py --contract shared/contracts/data_contract.example.yaml
python .claude/skills/torch-model/scripts/validate_contract.py --contract shared/contracts/model_contract.example.yaml
python .claude/skills/torch-infer-deploy/scripts/validate_contract.py --contract shared/contracts/deploy_contract.example.yaml
```

CI (`.github/workflows/ci.yml`) runs the same validators plus per-skill pytest suites and `flake8` lint on Python 3.12.

Current health check as of 2026-05-08: `199 passed`, and the canonical data, model, deploy, and project-spec examples validate successfully.

## Known warnings

The deployment tests currently emit PyTorch deprecation warnings for `torch.jit.trace`, `torch.jit.save`, and `torch.jit.load`. These warnings do not fail the suite, but the deployment path should keep TorchScript support working while tracking a future `torch.export` migration.
