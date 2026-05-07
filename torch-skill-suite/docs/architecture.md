# Architecture

Torch Skill Suite is organized as six Claude Code skills connected by explicit YAML contracts. Each skill owns one stage of a PyTorch engineering workflow and hands off structured artifacts to the next stage.

## Skill boundaries

1. `torch-data` owns dataset inspection, preprocessing decisions, Dataset/DataLoader scaffolding, and `data_contract.yaml` generation.
2. `torch-model` consumes the data contract, selects or scaffolds model structure, and produces `model_contract.yaml`.
3. `torch-train` consumes data and model contracts, then builds training loops, checkpointing, logging, and resume behavior.
4. `torch-eval-tune` analyzes metrics, validation behavior, and tuning directions without owning the core training loop.
5. `torch-infer-deploy` packages trained models for local inference, export, serving, and deployment contracts.
6. `torch-engineering` standardizes project structure, testing, CI, and maintainability concerns across the generated project.

## Contract layout

The authoritative schemas live in `shared/schemas/`:

- `project_spec.schema.json`
- `data_contract.schema.json`
- `model_contract.schema.json`
- `deploy_contract.schema.json`

Canonical checked-in examples live in `shared/contracts/` and use the `*.example.yaml` suffix. Skill-generated workspace artifacts normally use unsuffixed names such as `data_contract.yaml`, `model_contract.yaml`, and `deploy_contract.yaml`.

## Shared code

Reusable Python helpers live in `shared/python/torch_skill_shared/`. Skill scripts should prefer these helpers for shared behavior such as YAML loading and model-building utilities instead of duplicating logic.

## Current validation status

As of 2026-05-07, the package-level test suite passes from the `torch-skill-suite/` directory with `199 passed`. The canonical data, model, and deploy contract examples validate against their schemas.

## Deployment export note

`torch-infer-deploy` currently supports TorchScript-oriented flows and the tests pass, but PyTorch emits deprecation warnings for `torch.jit.trace`, `torch.jit.save`, and `torch.jit.load`. Keep this path stable for current users while evaluating `torch.export` as the future export target.
