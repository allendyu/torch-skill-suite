# Torch Skill Suite

A Claude Code skill suite for automating PyTorch deep learning engineering workflows.

## Included skills

- `torch-data` — data engineering
- `torch-model` — model construction
- `torch-train` — training loop generation
- `torch-eval-tune` — validation and tuning
- `torch-infer-deploy` — inference and deployment
- `torch-engineering` — engineering extension and standardization

## Shared resources

- `shared/schemas/` — JSON Schema files for contract validation
- `shared/contracts/` — canonical schema-valid `*.example.yaml` contract examples
- `shared/examples/contracts/` — scenario contract examples and recipes
- `shared/python/` — reusable Python utilities shared across skills
- `docs/` — architecture, workflow, and MVP status docs

Runtime skill outputs normally use unsuffixed names such as `data_contract.yaml`, `model_contract.yaml`, and `deploy_contract.yaml`. The checked-in canonical examples use the `*.example.yaml` suffix so they can coexist with generated workspace artifacts.

## Recommended flow

1. `torch-data`
2. `torch-model`
3. `torch-train`
4. `torch-eval-tune`
5. `torch-infer-deploy`
6. `torch-engineering`

## Development checks

Run commands from this package directory:

```bash
python -m pytest
python .claude/skills/torch-data/scripts/validate_contract.py --contract shared/contracts/data_contract.example.yaml
python .claude/skills/torch-model/scripts/validate_contract.py --contract shared/contracts/model_contract.example.yaml
python .claude/skills/torch-infer-deploy/scripts/validate_contract.py --contract shared/contracts/deploy_contract.example.yaml
```

Current health check as of 2026-05-07: `199 passed`, and the canonical data, model, and deploy contract examples validate successfully.

## Known warnings

The deployment tests currently emit PyTorch deprecation warnings for `torch.jit.trace`, `torch.jit.save`, and `torch.jit.load`. These warnings do not fail the suite, but the deployment path should keep TorchScript support working while tracking a future `torch.export` migration.
