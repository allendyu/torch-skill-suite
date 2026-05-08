# Workflow

This is the recommended end-to-end workflow across the six skills. Sequential execution is the canonical path, but skills can be invoked independently if their input contracts already exist.

## 1. Start with a project specification

Use `shared/contracts/project_spec.example.yaml` as a scaffold when a global task definition is useful. A project spec describes the task, data modality, expected outputs, and high-level constraints. Validation is enforced via `project_spec.schema.json`.

## 2. Build the data contract with `torch-data`

`torch-data` should inspect the existing dataset or project structure before generating code. Its main handoff artifact is `data_contract.yaml`, which captures input shape, task type, split strategy, preprocessing, and output specification.

Validate the canonical data example from the package directory:

```bash
python .claude/skills/torch-data/scripts/validate_contract.py --contract shared/contracts/data_contract.example.yaml
```

To go from a contract straight to runnable Dataset/DataLoader scaffolding, use `generate_dataset.py`:

```bash
python .claude/skills/torch-data/scripts/generate_dataset.py --data-contract data_contract.yaml --output-dir ./datasets
```

Supported codegen routes today: image classification (ImageFolder), text classification (JSONL/CSV), tabular classification/regression (CSV/TSV), image segmentation (image/mask pairs).

## 3. Resolve the model with `torch-model`

`torch-model` consumes a data contract, looks up the matching entry in `shared/route_map.yaml`, and produces a `model_contract.yaml` that names backbone, head, loss compatibility, and forward shape expectations.

```bash
python .claude/skills/torch-model/scripts/resolve_model.py --data-contract data_contract.yaml --output model_contract.yaml
python .claude/skills/torch-model/scripts/validate_contract.py --contract shared/contracts/model_contract.example.yaml
python .claude/skills/torch-model/scripts/smoke_test_model.py --model-contract model_contract.yaml
```

Templates ship for `image_classification`, `text_classification`, `image_segmentation`, `tabular_classification`, and `tabular_regression` under `.claude/skills/torch-model/templates/`.

## 4. Generate training with `torch-train`

`torch-train` consumes the data and model contracts to produce a reliable training entrypoint, optimizer/scheduler setup, checkpointing, logging, and resume behavior. It does not change the model architecture decisions owned by `torch-model`.

```bash
python .claude/skills/torch-train/scripts/train.py --data-contract data_contract.yaml --model-contract model_contract.yaml --epochs 10
python .claude/skills/torch-train/scripts/smoke_test_train.py
```

`e2e_cifar10.py` provides a real-data CIFAR-10 reference run for the image-classification route.

## 5. Evaluate and tune with `torch-eval-tune`

`torch-eval-tune` reads validation outputs, metrics, and experiment artifacts to summarize model quality and propose tuning actions. Its role is diagnosis and prioritization rather than replacing the trainer.

```bash
python .claude/skills/torch-eval-tune/scripts/evaluate.py --model-contract model_contract.yaml --checkpoint best_model.pt --output eval_report.yaml
python .claude/skills/torch-eval-tune/scripts/tune.py --history train_history.json --eval eval_report.yaml --output tuning_plan.yaml
```

## 6. Package inference and deployment with `torch-infer-deploy`

`torch-infer-deploy` prepares local inference, TorchScript/ONNX export, optional FastAPI serving, and `deploy_contract.yaml`.

```bash
python .claude/skills/torch-infer-deploy/scripts/export_model.py --model-contract model_contract.yaml --checkpoint best_model.pt --format torchscript
python .claude/skills/torch-infer-deploy/scripts/local_infer.py --model-path exported/model.torchscript.pt --model-contract model_contract.yaml --synthetic
python .claude/skills/torch-infer-deploy/scripts/serve.py --model-path exported/model.torchscript.pt --model-contract model_contract.yaml
python .claude/skills/torch-infer-deploy/scripts/validate_contract.py --contract shared/contracts/deploy_contract.example.yaml
```

TorchScript and ONNX exports are tested and passing, but PyTorch emits deprecation warnings for `torch.jit.*`. Future deployment work should evaluate `torch.export` while preserving current TorchScript/ONNX compatibility expectations.

## 7. Standardize with `torch-engineering`

`torch-engineering` is currently a documentation-only skill: it specifies how to align structure, tests, configs, CI, and maintainability across the suite, but does not yet ship scripts or templates. Use its `SKILL.md` to plan engineering work; expect scripts to land later.

## Development validation

Run the suite from this directory because `pytest.ini` is package-local:

```bash
python -m pytest
```

Current health check as of 2026-05-08: `199 passed`, and the canonical data, model, deploy, and project-spec contract examples validate successfully.
