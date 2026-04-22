# torch-train Development Roadmap

## Strategic Positioning

`torch-train` is the third skill in the suite pipeline. It consumes `data_contract.yaml` from `torch-data` and `model_contract.yaml` from `torch-model`, producing a trained model checkpoint and training logs for `torch-eval-tune` and `torch-infer-deploy`.

## Current MVP Scope (P0)

Only the `image_classification` route is supported:

- **Input**: data_contract + model_contract
- **Data mode**: Synthetic random data (for smoke testing)
- **Model**: ResNet (18/34/50) or EfficientNet-B0
- **Optimizer**: Adam (default) or SGD
- **Scheduler**: StepLR, CosineAnnealingLR, or none
- **Loss**: CrossEntropyLoss
- **Checkpoint**: model + optimizer + scheduler + epoch + history
- **Output**: Trained checkpoint + training history

## Core Goals

Build `torch-train` into a skill that can:

1. Consume `data_contract.yaml` and `model_contract.yaml` reliably
2. Build model from torch-model templates
3. Create DataLoader (synthetic for smoke test, real for production)
4. Set up optimizer / scheduler / loss automatically
5. Run training loop with logging
6. Save and resume from checkpoints

## Architecture

```
data_contract.yaml ──┐
                      ├──→ train.py ──→ checkpoints/
model_contract.yaml ──┘                 ├── best_model.pt
                                        ├── last_model.pt
                                        └── training_history
```

## Key Modules

1. **Contract Ingestion** — Read data_contract + model_contract
2. **Model Building** — Delegate to torch-model templates
3. **DataLoader Creation** — Synthetic (P0) or real ImageFolder (P1)
4. **Trainer** — Training loop with optimizer/scheduler/loss
5. **Checkpointing** — Save/restore full training state

## Development Phases

### Phase 1 — P0: Synthetic Smoke Test (current)
- [x] `train.py` with synthetic data support
- [x] `smoke_test_train.py` — verify training loop works
- [x] Trainer class with checkpointing
- [x] Optimizer/scheduler/loss builders
- [x] Tests

### Phase 2 — P1: Real Data Training
- ImageFolder DataLoader from data_contract
- Data transforms from data_contract.preprocessing
- Validation split handling
- Early stopping
- Gradient clipping
- Mixed precision (AMP)

### Phase 3 — P2: Advanced Training
- Multi-GPU support (DDP)
- Gradient accumulation
- TensorBoard / wandb logging
- Hyperparameter config via YAML
- train_config.yaml output

## Handoff Contracts

### Input (from torch-data + torch-model)
- `data_contract.yaml` — input_spec.shape, output_spec.num_classes
- `model_contract.yaml` — model_spec, head_spec

### Output (to torch-eval-tune)
- Checkpoint files (best_model.pt, last_model.pt)
- Training history (loss, accuracy per epoch)

## Verification Checklist

- [x] `train.py` script
- [x] `smoke_test_train.py` script
- [x] Trainer class
- [x] Checkpoint save/load/resume
- [x] Tests for training loop
- [x] Synthetic data generation
