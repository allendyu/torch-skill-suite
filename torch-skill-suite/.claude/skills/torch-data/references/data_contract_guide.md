# How to Fill the Data Contract

The `data_contract` is the central interface between `torch-data` and the rest of the skill suite. This guide walks you through each field and explains how to choose values based on your dataset and task.

## 1. Decide `data_type` and `task_type`

First, identify the primary modality of your input (`data_type`) and the machine‑learning task (`task_type`). Use the [Data Types](data_types.md) and [Task Types](task_types.md) references to pick the right values.

Example: If you have RGB images and want to classify them into 10 categories:
```yaml
data_type: image
task_type: classification
```

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
  num_classes: 80        # optional, if classifying each box
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
  type: image             # same as input modality
```

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
- The skill will attempt to guess the format (e.g., CSV, JSONL, image directory, etc.) and fill `inferred_format_spec`.
- Use this when you want a quick start and don't want to manually specify every detail.

```yaml
data_format_option: auto_inferred
# inferred_format_spec will be filled automatically after inspection
```

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

## Next Steps

Once your `data_contract` is complete, the `torch-data` skill will:

1. Validate the contract against the schema.
2. If `data_format_option` is `auto_inferred`, inspect the dataset and fill `inferred_format_spec`.
3. Generate Dataset/DataLoader code that matches your specifications.
4. Produce a `data_contract.yaml` file that can be consumed by `torch-model` and `torch-train`.

If you're unsure about a field, start with a minimal contract and let the skill ask for missing information.