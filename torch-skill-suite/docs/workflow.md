# Workflow

This is the recommended end-to-end workflow across the six skills.

## 1. Start with a project specification

Use `shared/contracts/project_spec.example.yaml` as a scaffold when a global task definition is useful. A project spec describes the task, data modality, expected outputs, and high-level constraints.

## 2. Build the data contract with `torch-data`

`torch-data` should inspect the existing dataset or project structure before generating code. Its main handoff artifact is `data_contract.yaml`, which captures input shape, task type, split strategy, preprocessing, and output specification.

Validate the canonical data example from the package directory:

```bash
python .claude/skills/torch-data/scripts/validate_contract.py --contract shared/contracts/data_contract.example.yaml
```

## 3. Resolve the model with `torch-model`

`torch-model` consumes a data contract and resolves the model family, backbone, prediction head, and loss compatibility. Its handoff artifact is `model_contract.yaml`.

```bash
python .claude/skills/torch-model/scripts/validate_contract.py --contract shared/contracts/model_contract.example.yaml
```

## 4. Generate training with `torch-train`

`torch-train` consumes the data and model contracts to produce a reliable training entrypoint, optimizer/scheduler setup, checkpointing, logging, and resume behavior. It should not change the model architecture decisions owned by `torch-model`.

## 5. Evaluate and tune with `torch-eval-tune`

`torch-eval-tune` reads validation outputs, metrics, and experiment artifacts to summarize model quality and propose tuning actions. Its role is diagnosis and prioritization rather than replacing the trainer.

## 6. Package inference and deployment with `torch-infer-deploy`

`torch-infer-deploy` prepares local inference, export artifacts, optional serving code, and `deploy_contract.yaml`.

```bash
python .claude/skills/torch-infer-deploy/scripts/validate_contract.py --contract shared/contracts/deploy_contract.example.yaml
```

TorchScript export is currently tested and passing, but PyTorch emits deprecation warnings for `torch.jit.*`. Future deployment work should evaluate `torch.export` while preserving current TorchScript/ONNX compatibility expectations.

## 7. Standardize with `torch-engineering`

Use `torch-engineering` to align structure, tests, configs, CI, and maintainability once the core data→model→train→eval→deploy path exists.

## Development validation

Run the suite from this directory because `pytest.ini` is package-local:

```bash
python -m pytest
```

Current health check as of 2026-05-07: `199 passed`, and the canonical data, model, and deploy contract examples validate successfully.
