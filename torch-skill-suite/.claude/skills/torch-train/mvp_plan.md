# torch-train Development Roadmap

## Strategic Positioning

`torch-train` is the third skill in the suite pipeline. It consumes `data_contract.yaml` from `torch-data` and `model_contract.yaml` from `torch-model`, and produces trained checkpoints plus training history for `torch-eval-tune` and `torch-infer-deploy`.

## Current Scope

Five routes are supported with both synthetic-data smoke tests and real-data training paths:

- `image_classification` (P0): ResNet (18/34/50), EfficientNet-B0; ImageFolder data path; CIFAR-10 reference run via `e2e_cifar10.py`
- `text_classification` (P1): pure-PyTorch TransformerEncoder; synthetic and real text datasets
- `image_segmentation` (P1): DeepLabV3 / UNet; synthetic and real image/mask datasets
- `tabular_classification` (P1): MLP; synthetic and CSV-driven datasets
- `tabular_regression` (P1): MLP regression head; CSV/TSV inputs

Optimizers: Adam (default), SGD. Schedulers: StepLR, CosineAnnealingLR, or none. Losses: CrossEntropyLoss for classification/segmentation, MSE for regression.

## Core Goals

Build `torch-train` into a skill that can:

1. Consume `data_contract.yaml` and `model_contract.yaml` reliably
2. Build model from `torch-model` templates via `torch_skill_shared.model_builder`
3. Create DataLoaders (synthetic factories for smoke tests, real loaders for production)
4. Set up optimizer / scheduler / loss based on the model contract's compatibility section
5. Run training loop with logging and history capture
6. Save and resume from checkpoints (model + optimizer + scheduler + epoch + history)

## Architecture

```
data_contract.yaml ──┐
                      ├──→ train.py / Trainer ──→ checkpoints/
model_contract.yaml ──┘                          ├── best_model.pt
                                                 ├── last_model.pt
                                                 └── train_history.json
```

## Key Modules

1. **Contract Ingestion** — Read data_contract + model_contract
2. **Model Building** — Delegate to `torch_skill_shared.model_builder`
3. **DataLoader Creation** — Synthetic factories (smoke tests) or real loaders (ImageFolder, CSV, JSONL, image/mask pairs)
4. **Trainer** — Training loop with optimizer/scheduler/loss
5. **Checkpointing** — Save/restore full training state, with resume support

## Scripts

- `train.py` — primary entrypoint; consumes both contracts and runs the loop
- `smoke_test_train.py` — synthetic-data smoke test for all five routes
- `e2e_cifar10.py` — real-data CIFAR-10 reference run for the image classification route

## Development Phases

### Phase 1 — P0: Synthetic Smoke Test (done)
- [x] `train.py` with synthetic data support across all five supported routes
- [x] `smoke_test_train.py` — verify training loop runs end-to-end
- [x] Trainer class with checkpointing
- [x] Optimizer / scheduler / loss builders driven by model contract
- [x] Tests

### Phase 2 — P1: Real Data Training (done)
- [x] ImageFolder DataLoader for image classification
- [x] CSV/JSONL DataLoaders for text and tabular routes
- [x] Image/mask pair DataLoader for segmentation
- [x] Real-data CIFAR-10 reference run (`e2e_cifar10.py`)
- [ ] Validation split handling beyond synthetic
- [ ] Early stopping
- [ ] Gradient clipping
- [ ] Mixed precision (AMP)

### Phase 3 — P2: Advanced Training
- [ ] Multi-GPU support (DDP)
- [ ] Gradient accumulation
- [ ] TensorBoard / wandb logging
- [ ] Hyperparameter config via YAML (`train_config.yaml` formal output)

## Handoff Contracts

### Input (from torch-data + torch-model)
- `data_contract.yaml` — input shape, output spec, splits, preprocessing
- `model_contract.yaml` — model spec, head spec, compatibility (loss, target dtype, output activation)

### Output (to torch-eval-tune + torch-infer-deploy)
- Checkpoint files (`best_model.pt`, `last_model.pt`)
- Training history (loss, accuracy/metric per epoch) as JSON

## Verification Checklist

- [x] `train.py` script
- [x] `smoke_test_train.py` script
- [x] `e2e_cifar10.py` reference run
- [x] Trainer class
- [x] Checkpoint save/load/resume
- [x] Tests for training loop across the five supported routes
- [x] Synthetic data factories shared via `torch_skill_shared.model_builder`
- [ ] Add early stopping, gradient clipping, AMP
- [ ] Emit a formal `train_config.yaml` artifact for reproducibility
