# How to Fill the Data Contract

The `data_contract` is the central interface between `torch-data` and the rest of the skill suite. This guide walks you through each field and explains how to choose values based on your dataset and task.

## 1. Decide `data_type` and `task_type`

First, identify the primary modality of your input (`data_type`) and the machine‑learning task (`task_type`). Use the [Data Types](data_types.md) and [Task Types](task_types.md) references to pick the right values.

Example: If you have RGB images and want to classify them into 10 categories:
```yaml
data_type: image
task_type: classification
```

For multimodal data, pick `data_type: multimodal` and describe each modality explicitly in `input_spec` or the format details.

## 2. Fill `input_spec`

The `input_spec` describes the shape, dtype, and other characteristics of a **single input sample** after preprocessing.

### For images
```yaml
input_spec:
  shape: [3, 224, 224]   # channels, height, width
  dtype: float32
  channels_first: true    # PyTorch default
```

### For text
```yaml
input_spec:
  sequence_length: 512    # fixed length after padding/truncation
  vocab_size: 30522       # vocabulary size of your tokenizer
  dtype: int64
```

### For time series
```yaml
input_spec:
  shape: [100, 5]        # time_steps, features
  dtype: float32
```

### For tabular
```yaml
input_spec:
  num_features: 30       # number of feature columns
  dtype: float32
```

### For audio
```yaml
input_spec:
  sample_rate: 16000
  duration: 5.0          # seconds (optional)
  shape: [1, 80000]      # channels, samples (raw waveform)
  dtype: float32
```

### For video
```yaml
input_spec:
  shape: [3, 16, 224, 224]   # channels, frames, height, width
  fps: 30
  dtype: uint8
```

### For multimodal
```yaml
input_spec:
  image:
    shape: [3, 224, 224]
    dtype: float32
    channels_first: true
  text:
    sequence_length: 128
    vocab_size: 30522
    dtype: int64
```

When using multimodal input, prefer a per-modality nested structure and make alignment explicit in `user_format_spec.details`.

## 3. Fill `output_spec`

The `output_spec` describes the format of labels/targets.

### For classification
```yaml
output_spec:
  type: categorical
  num_classes: 10
  label_map:              # optional
    0: cat
    1: dog
```

### For detection
```yaml
output_spec:
  type: bounding_box
  bbox_format: xywh       # or xyxy, cxcywh
  num_classes: 80         # optional, if classifying each box
```

### For segmentation
```yaml
output_spec:
  type: mask
  mask_shape: [224, 224]  # height, width
  num_classes: 21
```

### For regression
```yaml
output_spec:
  type: continuous
  output_dim: 1           # scalar regression
```

### For generation/translation
```yaml
output_spec:
  type: sequence
```

At the current stage, the strongest validated patterns are classification, detection, segmentation, and regression. Generation and translation remain more flexible and may need additional format details.

## 4. Define `splits`

The `splits` section tells the data loader where to find each partition.

### Path‑based splits (most common)
```yaml
splits:
  train: data/train
  val: data/val
  test: data/test
```

### Proportion‑based splits (auto‑split from a single source)
```yaml
splits:
  train: 0.8
  val: 0.1
  test: 0.1
```

### Custom split files
```yaml
splits:
  train: splits/train.txt   # list of sample IDs or paths
  val: splits/val.txt
```

For multimodal or manifest-driven datasets, split entries often point to manifest files instead of raw asset directories.

## 5. List `preprocessing` steps

Preprocessing steps are applied in order. Each step has a `name` and optional `params`.

Example for images:
```yaml
preprocessing:
  - name: resize
    params: { size: [256, 256] }
  - name: random_crop
    params: { size: [224, 224] }
  - name: normalize
    params: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }
```

Example for text:
```yaml
preprocessing:
  - name: tokenize
    params: { tokenizer: "bert-base-uncased", max_length: 512 }
  - name: add_special_tokens
    params: {}
```

Example for audio:
```yaml
preprocessing:
  - name: resample
    params: { sample_rate: 16000 }
  - name: pad_or_trim
    params: { num_samples: 80000 }
  - name: to_mel_spectrogram
    params: { n_mels: 64, hop_length: 512 }
```

Example for video:
```yaml
preprocessing:
  - name: temporal_subsample
    params: { num_frames: 16, strategy: uniform }
  - name: resize
    params: { size: [224, 224] }
```

## 6. Choose `data_format_option`

This is a key decision: whether you will provide the exact format of the raw data (`user_provided`) or let `torch-data` infer it from the dataset (`auto_inferred`).

### Option A: `user_provided`
- You know exactly how your data is stored.
- You must describe it in `user_format_spec`.
- Use this when you have a custom directory layout, proprietary file format, or explicit annotation format.

```yaml
data_format_option: user_provided
user_format_spec:
  format_type: ImageFolder
  details:
    structure: "class folders inside split folders"
    extensions: [".jpg", ".png"]
```

### Option B: `auto_inferred`
- You point to the dataset and let `torch-data` examine its structure.
- The skill will attempt to guess the format (e.g., CSV, JSONL, image directory, audio folder, video folder, etc.) and fill `inferred_format_spec`.
- Use this when you want a quick start and don't want to manually specify every detail.

```yaml
data_format_option: auto_inferred
# inferred_format_spec will be filled automatically after inspection
```

Current inspection output may also include:
- `confidence`
- `warnings`
- `observed_fields`
- `missing_information`

Current recognized `auto_inferred` patterns include:
- image classification: split/class folders or flat image directories
- image detection: YOLO-style image/label folders, COCO-style JSON hints
- image segmentation: image/mask directory pairs, COCO-style JSON hints
- text classification: CSV, JSONL, text-file directories
- tabular: CSV, TSV, JSONL, JSON, Parquet-by-suffix
- time series: CSV, TSV, NPZ-by-suffix
- audio classification: folder-per-class audio data, audio + metadata manifest
- video classification: folder-per-class clip data, extracted frame directories, video + metadata manifest
- multimodal classification: CSV/TSV/JSON/JSONL manifests that contain multiple modalities such as image+text or audio+text

Important:
- Inspection is heuristic. Treat `inferred_format_spec` as a strong draft, not guaranteed truth.
- Example contracts may intentionally omit a realized `inferred_format_spec` at authoring time and instead include a commented “Expected inference” block.
- After inspection runs, the final contract can include `inferred_format_spec` explicitly.
- When warnings or missing information are returned, confirm those fields before generating downstream Dataset/DataLoader code.

## 7. Provide `metadata` (optional)

Add any extra information that might be useful for documentation or downstream steps.

```yaml
metadata:
  source: "My custom dataset"
  version: "1.0"
  license: "MIT"
  description: "Dataset of animal images collected in 2025"
```

## Full Example (Image Classification)

```yaml
data_type: image
task_type: classification
input_spec:
  shape: [3, 224, 224]
  dtype: float32
  channels_first: true
output_spec:
  type: categorical
  num_classes: 10
  label_map:
    0: cat
    1: dog
    2: bird
splits:
  train: data/train
  val: data/val
  test: data/test
preprocessing:
  - name: resize
    params: { size: [256, 256] }
  - name: random_crop
    params: { size: [224, 224] }
  - name: normalize
    params: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] }
data_format_option: user_provided
user_format_spec:
  format_type: ImageFolder
  details:
    structure: "class folders inside split folders"
    extensions: [".jpg", ".png"]
metadata:
  source: "Animal-10 dataset"
```

## Additional Example Patterns

### Image segmentation
Use paired image/mask directories and specify `mask_shape` plus class semantics.

### Tabular regression
Use `output_spec.type: continuous`, set `output_dim`, and identify the target column in `user_format_spec.details`.

### Audio classification
Set `sample_rate`, optional `duration`, and waveform or spectrogram shape assumptions. Let `auto_inferred` determine whether labels come from folder names or a manifest.

### Video classification
Include temporal information via `shape` and/or `fps`, then describe whether clips come from decoded videos or frame directories.

### Multimodal paired classification
Use nested `input_spec` entries per modality and define pairing in a manifest, not by implicit naming when possible.

## Next Steps

Once your `data_contract` is complete, the `torch-data` skill will:

1. Validate the contract against the schema.
2. If `data_format_option` is `auto_inferred`, inspect the dataset and fill `inferred_format_spec`.
3. Generate Dataset/DataLoader code that matches your specifications.
4. Produce a `data_contract.yaml` file that can be consumed by downstream skills.

If you're unsure about a field, start with a minimal contract and let the skill ask for missing information.
