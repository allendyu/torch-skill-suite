# MVP Roadmap

Torch Skill Suite is in MVP refinement with all six skills now present and initially implemented. The image-classification route remains the primary end-to-end path for validating the suite.

## Current baseline

As of 2026-05-07:

- Package tests pass from `torch-skill-suite/`: `199 passed`.
- Canonical examples validate successfully:
  - `shared/contracts/data_contract.example.yaml`
  - `shared/contracts/model_contract.example.yaml`
  - `shared/contracts/deploy_contract.example.yaml`
- The checked-in examples use `*.example.yaml`; generated workflow artifacts should continue to use operational names like `data_contract.yaml`, `model_contract.yaml`, and `deploy_contract.yaml`.

## MVP priorities

1. Keep the image-classification path green across data, model, train, eval, deploy, and engineering skills.
2. Expand scenario examples in `shared/examples/contracts/` without weakening schema strictness.
3. Keep shared utilities in `shared/python/torch_skill_shared/` as the source of reusable behavior across skills.
4. Replace placeholder example workspaces with runnable minimal projects when the corresponding route is ready.
5. Track PyTorch export API changes: current TorchScript tests pass, but `torch.jit.*` deprecation warnings indicate that `torch.export` should be evaluated before the deployment path graduates beyond MVP.

## Validation commands

Run from `torch-skill-suite/`:

```bash
python -m pytest
python .claude/skills/torch-data/scripts/validate_contract.py --contract shared/contracts/data_contract.example.yaml
python .claude/skills/torch-model/scripts/validate_contract.py --contract shared/contracts/model_contract.example.yaml
python .claude/skills/torch-infer-deploy/scripts/validate_contract.py --contract shared/contracts/deploy_contract.example.yaml
```
