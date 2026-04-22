# torch-model Development Roadmap

## Strategic Positioning

`torch-model` is the second skill in the suite pipeline. It consumes `data_contract.yaml` from `torch-data` and produces `model_contract.yaml` for `torch-train`. Its role is to translate data specifications into concrete model architecture decisions.

At this stage, the goal is **narrow and reliable**: support one route well before expanding.

## Current MVP Scope (P0)

Only the `image_classification` route is implemented:

- **data_type**: image
- **task_type**: classification
- **backbones**: resnet18, resnet34, resnet50, efficientnet_b0
- **head**: linear_cls (global average pooling + linear)
- **output**: logits over num_classes

## Core Goals

Build `torch-model` into a skill that can:

1. Consume `data_contract.yaml` reliably
2. Match data/task combinations to model routes via `route_map.yaml`
3. Select appropriate backbone based on constraints
4. Generate `model_contract.yaml` as stable output
5. Produce runnable model scaffolding from templates
6. Validate model correctness via dummy forward pass

## Architecture

### Contract-Driven Design

```
data_contract.yaml ──→ resolve_model.py ──→ model_contract.yaml
                              │
                      route_map.yaml (match rules)
                              │
                      templates/ (code generation)
```

### Key Modules

1. **Contract Ingestion** — Extract model-relevant fields from `data_contract`
2. **Route Matching** — Match (data_type, task_type) against route_map
3. **Backbone Selection** — Choose backbone based on constraints
4. **Contract Emission** — Write `model_contract.yaml`
5. **Template Rendering** — Generate model code from contract
6. **Smoke Testing** — Dummy forward pass validation

## Development Principles

1. **Contract first, code second** — stabilize the schema before generating models
2. **Template-based, not free-form** — use predefined templates, not arbitrary generation
3. **One route at a time** — finish image_classification before expanding
4. **Validate at boundaries** — shape checks, dummy forward, output dimension checks
5. **Downstream compatibility** — model_contract must be consumable by torch-train

## Capability Layers

### Layer 1 — Contract Support (current)
- `model_contract.schema.json` with task-aware conditional validation
- Example contracts for all supported backbone variants
- Contract validation script

### Layer 2 — Model Resolution (current)
- `resolve_model.py` reads data_contract → outputs model_contract
- Backbone selection rules based on constraints
- Supports image + classification only

### Layer 3 — Template Generation (current)
- ResNet template (resnet18/34/50)
- EfficientNet template (efficientnet_b0)
- Common head modules
- `build_model(config)` interface

### Layer 4 — Smoke Testing (current)
- Dummy forward pass validation
- Output shape verification
- Parameter count reporting

## Support Matrix (Current)

| Route | data_type | task_type | Contract | Resolver | Template | Smoke Test |
|---|---:|---:|---:|---:|---:|
| image_classification | image | classification | yes | yes | yes | yes |

## Development Phases

### Phase 1 — P0: Image Classification MVP (current)
- [x] `model_contract.schema.json`
- [x] `model_contract.example.yaml`
- [x] `validate_contract.py`
- [x] `resolve_model.py` (image+classification only)
- [x] Model templates (resnet, efficientnet)
- [x] `smoke_test_model.py`
- [x] Tests

### Phase 2 — P1: Expand to Text, Segmentation, Tabular
- text_classification (BERT/RoBERTa/DistilBERT)
- image_segmentation (U-Net/DeepLabV3)
- tabular_classification (MLP/TabNet)
- tabular_regression (MLP/TabNet)

### Phase 3 — P2: Detection, Time Series, Audio
- image_detection (Faster R-CNN/SSD/YOLO)
- time_series_classification (LSTM/TCN)
- time_series_regression (LSTM/TCN)
- audio_classification (CNN10/ResNet1D)

### Phase 4 — P3: Video, Multimodal, Generation
- video_classification (R3D/TimeSformer)
- multimodal_classification (Two-Tower/CLIP)
- text_generation (T5/GPT2)

## Handoff Contracts

### Input (from torch-data)
Consumes from `data_contract.yaml`:
- `data_type`, `task_type`
- `input_spec.shape`, `input_spec.dtype`, `input_spec.channels_first`
- `output_spec.type`, `output_spec.num_classes`

### Output (to torch-train)
Produces `model_contract.yaml`:
- `model_spec` — backbone, pretrained, feature_dim
- `head_spec` — type, num_classes, pooling, dropout
- `forward_spec` — output_shape
- `compatibility` — expected_loss, target_dtype, output_activation

## Verification Checklist

- [x] `model_contract.schema.json` with conditional validation
- [x] `model_contract.example.yaml` with 4 examples
- [x] `validate_contract.py` script
- [x] `resolve_model.py` script
- [x] Model templates (resnet, efficientnet, heads)
- [x] `smoke_test_model.py` script
- [x] Tests for contract validation, resolution, and smoke testing
