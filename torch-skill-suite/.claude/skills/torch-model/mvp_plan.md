# torch-model Development Roadmap

## Strategic Positioning

`torch-model` is the second skill in the suite pipeline. It consumes `data_contract.yaml` from `torch-data` and produces `model_contract.yaml` for `torch-train`. Its role is to translate data specifications into concrete model architecture decisions via a route map.

## Current Scope

Five routes are supported with templates and smoke tests:

- **image_classification** (P0): resnet18/34/50, efficientnet_b0; head `linear_cls` with global average pooling
- **text_classification** (P1): pure-PyTorch `TransformerEncoder` backbones (no HuggingFace dependency); head `pooled_linear_cls`
- **image_segmentation** (P1): deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, unet; head `segmentation_head` producing `[B, num_classes, H, W]` logits
- **tabular_classification** (P1): mlp backbone (default), with tabnet/ft_transformer placeholders; head `linear_cls`
- **tabular_regression** (P1): same backbones as tabular classification; head `linear_regression` driven by `output_spec.output_dim`

Other routes (detection, time series, audio, video, multimodal, generation) are defined in `shared/route_map.yaml` but `model_resolver`/`template`/`smoke_test` are still `unsupported`.

## Core Goals

Build `torch-model` into a skill that can:

1. Consume `data_contract.yaml` reliably
2. Match data/task combinations to model routes via `shared/route_map.yaml`
3. Select an appropriate backbone based on constraints (latency, model size, num_features)
4. Generate `model_contract.yaml` as stable output
5. Produce runnable model scaffolding from templates
6. Validate model correctness via dummy forward pass

## Architecture

### Contract-Driven Design

```
data_contract.yaml ──→ resolve_model.py ──→ model_contract.yaml
                              │
                      shared/route_map.yaml (match rules + selection)
                              │
                      templates/ (code generation)
                              │
                      smoke_test_model.py (forward verification)
```

### Key Modules

1. **Contract Ingestion** — Extract model-relevant fields from `data_contract`
2. **Route Matching** — Match (data_type, task_type) against `route_map.yaml`
3. **Backbone Selection** — Choose backbone based on constraints
4. **Contract Emission** — Write `model_contract.yaml`
5. **Template Rendering** — Generate model code from contract
6. **Smoke Testing** — Dummy forward pass validation

## Development Principles

1. **Contract first, code second** — stabilize the schema before generating models
2. **Template-based, not free-form** — use predefined templates, not arbitrary generation
3. **One route at a time** — finish a route end-to-end before expanding
4. **Validate at boundaries** — shape checks, dummy forward, output dimension checks
5. **Downstream compatibility** — `model_contract` must be consumable by `torch-train` and `torch-infer-deploy`

## Capability Layers

### Layer 1 — Contract Support (current)
- `model_contract.schema.json` with task-aware conditional validation
- Example contracts under `shared/examples/contracts/model/` for image_classification (resnet18/50, efficientnet_b0)
- Skill-local example contracts under `examples/contracts/` for the five supported routes
- Contract validation script (`scripts/validate_contract.py`)

### Layer 2 — Model Resolution (current)
- `resolve_model.py` reads data_contract → outputs model_contract
- Backbone selection rules driven by `shared/route_map.yaml`
- Supports the five routes listed above

### Layer 3 — Template Generation (current)
- ResNet / EfficientNet templates (`templates/image_classification/`)
- DeepLabV3 / UNet templates (`templates/image_segmentation/`)
- TransformerEncoder template (`templates/text_classification/`)
- MLP template for tabular tasks (`templates/tabular_classification/`, `templates/tabular_regression/`)
- Common head modules (`templates/common/`)
- `build_model(config)` interface

### Layer 4 — Smoke Testing (current)
- Dummy forward pass validation across all five routes
- Output shape verification
- Parameter count reporting

## Support Matrix

| Route | data_type | task_type | Contract | Resolver | Template | Smoke Test |
| --- | --- | --- | :---: | :---: | :---: | :---: |
| image_classification | image | classification | yes | yes | yes | yes |
| text_classification | text | classification | yes | yes | yes | yes |
| image_segmentation | image | segmentation | yes | yes | yes | yes |
| tabular_classification | tabular | classification | yes | yes | yes | yes |
| tabular_regression | tabular | regression | yes | yes | yes | yes |
| image_detection | image | detection | yes | no | no | no |
| time_series_classification | time_series | classification | yes | no | no | no |
| time_series_regression | time_series | regression | yes | no | no | no |
| audio_classification | audio | classification | yes | no | no | no |
| video_classification | video | classification | yes | no | no | no |
| multimodal_classification | multimodal | classification | yes | no | no | no |
| text_generation | text | generation | partial | no | no | no |

## Development Phases

### Phase 1 — P0: Image Classification MVP (done)
- [x] `model_contract.schema.json`
- [x] `model_contract.example.yaml`
- [x] `validate_contract.py`
- [x] `resolve_model.py` (image+classification)
- [x] Model templates (resnet, efficientnet)
- [x] `smoke_test_model.py`
- [x] Tests

### Phase 2 — P1: Text, Segmentation, Tabular (done)
- [x] text_classification with pure-PyTorch TransformerEncoder
- [x] image_segmentation with DeepLabV3 / UNet
- [x] tabular_classification with MLP
- [x] tabular_regression with MLP
- [ ] Add tabnet / ft_transformer alternatives
- [ ] Add UNet-only variant (encoder/decoder split) and SegFormer

### Phase 3 — P2: Detection, Time Series, Audio
- [ ] image_detection (Faster R-CNN / SSD / YOLO)
- [ ] time_series_classification (LSTM / TCN / PatchTST)
- [ ] time_series_regression (LSTM / TCN / PatchTST)
- [ ] audio_classification (CNN10 / 1D ResNet)

### Phase 4 — P3: Video, Multimodal, Generation
- [ ] video_classification (R3D / TimeSformer / SlowFast)
- [ ] multimodal_classification (Two-Tower / CLIP-style)
- [ ] text_generation (T5 / BART / GPT2-style decoder)

## Handoff Contracts

### Input (from torch-data)
Consumes from `data_contract.yaml`:
- `data_type`, `task_type`
- `input_spec.shape`, `input_spec.dtype`, `input_spec.channels_first`, `input_spec.num_features`, `input_spec.sequence_length`
- `output_spec.type`, `output_spec.num_classes`, `output_spec.output_dim`, `output_spec.mask_shape`

### Output (to torch-train and torch-infer-deploy)
Produces `model_contract.yaml`:
- `model_spec` — backbone, pretrained, feature_dim
- `head_spec` — type, num_classes / output_dim, pooling, dropout
- `forward_spec` — output_shape, output activation
- `compatibility` — expected_loss, target_dtype, output_activation

## Verification Checklist

- [x] `model_contract.schema.json` with conditional validation
- [x] `model_contract.example.yaml` with multiple examples
- [x] `validate_contract.py` script
- [x] `resolve_model.py` script
- [x] Model templates for the five supported routes
- [x] `smoke_test_model.py` script
- [x] Tests for contract validation, resolution, and smoke testing
- [ ] Promote next route from P2 (recommended: `image_detection` or `time_series_classification`)

