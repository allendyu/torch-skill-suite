# MVP Roadmap

Torch Skill Suite is past the initial single-route MVP. Five end-to-end routes are now supported across the five production skills, with `torch-engineering` still a documentation-only stub. Per-skill roadmaps live in each skill's `mvp_plan.md`.

## Current baseline

As of 2026-05-08:

- Package tests pass from `torch-skill-suite/`: `199 passed`.
- Canonical examples validate successfully:
  - `shared/contracts/project_spec.example.yaml`
  - `shared/contracts/data_contract.example.yaml`
  - `shared/contracts/model_contract.example.yaml`
  - `shared/contracts/deploy_contract.example.yaml`
- Checked-in examples use `*.example.yaml`; generated workflow artifacts continue to use operational names like `data_contract.yaml`, `model_contract.yaml`, `train_config.yaml`, `tuning_plan.yaml`, and `deploy_contract.yaml`.

## Supported routes

The five production skills (`torch-data` → `torch-model` → `torch-train` → `torch-eval-tune` → `torch-infer-deploy`) are fully wired for these routes:

| Route | Priority | torch-data | torch-model | torch-train | torch-eval-tune | torch-infer-deploy |
| --- | --- | --- | --- | --- | --- | --- |
| `image_classification` | P0 | supported | supported | supported | supported | supported |
| `text_classification` | P1 | supported | supported | supported | supported | supported |
| `image_segmentation` | P1 | supported | supported | supported | supported | supported |
| `tabular_classification` | P1 | supported | supported | supported | supported | supported |
| `tabular_regression` | P1 | supported | supported | supported | supported | supported |

P2 routes (`image_detection`, `time_series_classification/regression`, `audio_classification`) and P3 routes (`video_classification`, `multimodal_classification`, `text_generation`, `time_series_forecasting`) have data-side support and route definitions but no `torch-model` template yet. See `shared/route_map.yaml` for the authoritative status.

## Skill maturity

| Skill | SKILL.md | Scripts | Templates | Tests |
| --- | --- | --- | --- | --- |
| `torch-data` | yes | yes | n/a (codegen-driven) | yes |
| `torch-model` | yes | yes | yes (5 routes) | yes |
| `torch-train` | yes | yes | n/a | yes |
| `torch-eval-tune` | yes | yes | n/a | yes |
| `torch-infer-deploy` | yes | yes | n/a | yes |
| `torch-engineering` | yes | not yet | not yet | not yet |

## MVP priorities

1. Keep the five P1+P0 routes green across data → model → train → eval → deploy.
2. Promote `torch-engineering` from documentation-only to having at least one runnable script (e.g., a project structure normalizer or quality-gate checker).
3. Expand scenario examples in `shared/examples/contracts/` without weakening schema strictness.
4. Pick the next route from `shared/route_map.yaml` (most natural candidates: `image_detection` or `time_series_classification`) and lift it to `supported` across all five production skills.
5. Track PyTorch export API changes: current TorchScript/ONNX tests pass, but `torch.jit.*` deprecation warnings indicate that `torch.export` should be evaluated before deployment graduates beyond MVP.

## Validation commands

Run from `torch-skill-suite/`:

```bash
python -m pytest
python .claude/skills/torch-data/scripts/validate_contract.py --contract shared/contracts/data_contract.example.yaml
python .claude/skills/torch-model/scripts/validate_contract.py --contract shared/contracts/model_contract.example.yaml
python .claude/skills/torch-infer-deploy/scripts/validate_contract.py --contract shared/contracts/deploy_contract.example.yaml
```

CI (`.github/workflows/ci.yml`) reproduces these checks on Python 3.12.
