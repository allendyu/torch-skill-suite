# torch-eval-tune Development Roadmap

## Strategic Positioning

`torch-eval-tune` is the fourth skill in the suite pipeline. It consumes a trained checkpoint plus its `data_contract.yaml` / `model_contract.yaml` / training history, runs validation, and produces both a structured evaluation report and a prioritized tuning plan. Its role is **diagnosis and prioritization**, not retraining.

## Current Scope

Five routes are evaluable end-to-end (matching what `torch-train` produces):

- `image_classification`, `text_classification`, `image_segmentation` — classification-style metrics: accuracy, macro/micro precision/recall/F1, per-class breakdown, confusion matrix
- `tabular_classification` — same classification metrics
- `tabular_regression` — regression metrics: MSE, MAE, RMSE, R²

Inputs supported:

- Real validation data when paths are present in the data contract
- Synthetic factories from `torch_skill_shared.model_builder` (for smoke tests when no real data is available)

## Core Goals

Build `torch-eval-tune` into a skill that can:

1. Load a checkpoint built by `torch-train` and rebuild the model from `model_contract.yaml`
2. Run inference over a validation split and compute task-appropriate metrics
3. Surface failure modes (per-class metrics, confusion matrix, error clusters) rather than only headline scores
4. Read training history and detect overfitting, underfitting, plateau, and divergence patterns
5. Produce a short, prioritized tuning plan rather than an unranked checklist
6. Distinguish facts, interpretations, and suggested next actions in its outputs

## Architecture

```
checkpoint + model_contract.yaml ──┐
                                    ├──→ evaluate.py ──→ eval_report.yaml
data_contract.yaml + val data ─────┘

train_history.json ──┐
                      ├──→ tune.py ──→ tuning_plan.yaml
eval_report.yaml ────┘
```

## Scripts

- `evaluate.py` — load checkpoint, run validation, emit structured eval report
- `tune.py` — analyze training history + eval report, emit ranked tuning plan
- `smoke_test_eval.py` — end-to-end: train (synthetic) → evaluate → tune

## Development Phases

### Phase 1 — P0: Classification + Regression metrics (done)
- [x] `evaluate.py` with classification metrics (accuracy, P/R/F1, per-class, confusion matrix)
- [x] Regression metrics (MSE, MAE, RMSE, R²)
- [x] `tune.py` with overfitting / plateau / lr-too-high heuristics
- [x] `smoke_test_eval.py` covering all five routes
- [x] Structured `eval_report.yaml` and `tuning_plan.yaml` outputs
- [x] Tests

### Phase 2 — P1: Richer error analysis
- [ ] Top-K confusion clusters with class names
- [ ] Calibration metrics (ECE, reliability diagrams)
- [ ] Per-slice metrics (e.g., grouped by metadata field)
- [ ] Segmentation-specific metrics: per-class IoU, mIoU, boundary IoU
- [ ] Detection metrics path (mAP, AP@IoU thresholds) once `torch-model` supports detection
- [ ] Compare two checkpoints/runs in a single report

### Phase 3 — P2: Tuning depth
- [ ] Bayesian / Optuna integration template
- [ ] Suggest data-side actions (more augmentation, class rebalancing) versus model-side actions
- [ ] Cross-experiment comparison via a small experiment registry

## Handoff Contracts

### Input
- `model_contract.yaml` — to rebuild the model
- Checkpoint (`best_model.pt` or `last_model.pt`) from `torch-train`
- `data_contract.yaml` — to locate the validation split and preprocessing
- `train_history.json` — for tuning analysis

### Output
- `eval_report.yaml` — facts: metrics, per-class numbers, confusion matrix, sample counts
- `tuning_plan.yaml` — prioritized list of suggested actions, each with rationale and expected effect

## Boundaries

- Does **not** rebuild the trainer or change architecture decisions
- Does **not** export models or build serving — that belongs to `torch-infer-deploy`
- Treats large hyperparameter searches as a deliberate later phase, not the default

## Verification Checklist

- [x] `evaluate.py` script
- [x] `tune.py` script
- [x] `smoke_test_eval.py` script
- [x] Tests for evaluation + tuning logic
- [x] Eval and tuning outputs validated for the five supported routes
- [ ] Add detection / time-series / audio metrics once those routes ship in `torch-model`
- [ ] Add experiment-comparison mode (two runs in one report)
