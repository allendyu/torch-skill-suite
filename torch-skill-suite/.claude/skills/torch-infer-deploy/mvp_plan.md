# torch-infer-deploy Development Roadmap

## Strategic Positioning

`torch-infer-deploy` is the fifth skill in the suite pipeline. It consumes a trained checkpoint plus its `model_contract.yaml`, packages the model for local inference, exports it to TorchScript or ONNX, and optionally wraps it in a FastAPI service. Its role is to **take a trained model into inference reliably**, not to retrain it.

## Current Scope

Five routes are deployable end-to-end (matching what `torch-train` produces):

- `image_classification` — ResNet, EfficientNet
- `text_classification` — pure-PyTorch TransformerEncoder
- `image_segmentation` — DeepLabV3, UNet
- `tabular_classification` — MLP
- `tabular_regression` — MLP regression head

Capabilities:

- TorchScript export (default) and ONNX export (with `onnxruntime` validation)
- Local inference from a single file, a directory batch, or synthetic input
- FastAPI serving with `/health` and `/predict` endpoints (image upload or JSON tensor)
- `deploy_contract.yaml` validation against `shared/schemas/deploy_contract.schema.json`

## Core Goals

Build `torch-infer-deploy` into a skill that can:

1. Load a checkpoint and rebuild the model from `model_contract.yaml`
2. Align preprocessing/postprocessing with the training side
3. Export models to TorchScript or ONNX with shape verification
4. Serve models via a minimal FastAPI app (image upload or JSON tensor)
5. Provide latency / throughput benchmarking guidance
6. Produce or update `deploy_contract.yaml` to fix runtime assumptions

## Architecture

```
checkpoint + model_contract.yaml ──→ export_model.py ──→ exported/model.{torchscript.pt,onnx}
                                          │
                                          └──→ local_infer.py ──→ predictions
                                          │
                                          └──→ serve.py ──→ FastAPI app (/health, /predict)

deploy_contract.yaml ──→ validate_contract.py
```

## Scripts

- `export_model.py` — TorchScript and ONNX export with shape verification
- `local_infer.py` — local inference (single file, directory, synthetic)
- `serve.py` — FastAPI serving with image / tensor input
- `smoke_test_deploy.py` — end-to-end smoke test (export → load → infer)
- `validate_contract.py` — schema validation for `deploy_contract.yaml`

## Development Phases

### Phase 1 — P0: TorchScript + Local Inference (done)
- [x] `export_model.py` with TorchScript export
- [x] `local_infer.py` with single-file and synthetic input
- [x] `smoke_test_deploy.py`
- [x] `deploy_contract.yaml` schema and validator
- [x] Tests for export and inference

### Phase 2 — P1: ONNX + Serving (done)
- [x] ONNX export path with `onnxruntime` shape validation
- [x] `serve.py` FastAPI app with `/health` and `/predict`
- [x] Multi-route support: image classification, text classification, image segmentation, tabular tasks
- [x] Batch inference over a directory
- [ ] Quantization export path (dynamic / static / QAT)
- [ ] Benchmarking script with structured latency / throughput report

### Phase 3 — P2: Hardening
- [ ] Optional TensorRT export path
- [ ] Model versioning conventions in `deploy_contract.yaml`
- [ ] Streaming / batched serving for high-throughput cases
- [ ] Migrate from `torch.jit.*` to `torch.export` once API stabilizes

## Handoff Contracts

### Input
- `model_contract.yaml` — to rebuild the model and infer expected I/O shapes
- Checkpoint (`best_model.pt` or `last_model.pt`)
- Optional `deploy_contract.yaml` — preprocess/postprocess and runtime form
- Optional `data_contract.yaml` — to align preprocessing with training

### Output
- Exported artifacts under `./exported/` (`model.torchscript.pt`, `model.onnx`)
- `deploy_contract.yaml` capturing format, service type, preprocess, postprocess, batch behavior

## Boundaries

- Does **not** retrain or fine-tune
- Does **not** own evaluation metric strategy — that's `torch-eval-tune`
- Defers long-term repository normalization to `torch-engineering`

## Known warnings

Tests currently emit PyTorch deprecation warnings for `torch.jit.trace`, `torch.jit.save`, `torch.jit.load`, and `torch.jit.trace_method`. The deployment path should keep TorchScript working in the short term while planning a `torch.export` migration before the deployment path graduates beyond MVP.

## Verification Checklist

- [x] `export_model.py` (TorchScript + ONNX)
- [x] `local_infer.py` (single file, batch directory, synthetic)
- [x] `serve.py` (FastAPI `/health`, `/predict`)
- [x] `validate_contract.py` for `deploy_contract.yaml`
- [x] `smoke_test_deploy.py`
- [x] Tests for export, local inference, and serving
- [ ] Add quantization export path
- [ ] Add structured benchmarking script
- [ ] Plan migration from `torch.jit.*` to `torch.export`
