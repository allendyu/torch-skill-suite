# Data Types Supported by torch-data

This document describes the data modalities that `torch-data` skill can handle, and how to specify them in the `data_contract`.

## Overview

The `data_type` field in `data_contract` determines the primary modality of your input data. Each data type has its own set of typical preprocessing steps, input shapes, and common storage formats.

## 1. Image

- **Description**: 2D pixel arrays, usually with color channels (RGB) or grayscale.
- **Typical tasks**: Classification, detection, segmentation, generation, super‑resolution.
- **Input spec fields**:
  - `shape`: `[channels, height, width]` (PyTorch default is channels‑first)
  - `dtype`: `uint8` (0–255) or `float32` (normalized)
  - `channels_first`: `true` (PyTorch) or `false` (OpenCV/TF default)
- **Common storage formats**:
  - Folder‑per‑class (`ImageFolder`)
  - COCO JSON (detection/segmentation)
  - YOLO‑style `.txt` per image
  - Plain directory of images with separate annotation files
- **Preprocessing typical steps**:
  - Resize / crop
  - Normalize (mean/std or scale to [0,1])
  - Color jitter, random flip, rotation (augmentation)
- **Example `data_type`: `"image"`**


## 2. Text

- **Description**: Sequences of tokens (words, sub‑words, characters).
- **Typical tasks**: Classification, sequence labeling, translation, generation, summarization.
- **Input spec fields**:
  - `sequence_length`: fixed length or `-1` for variable
  - `vocab_size`: size of the tokenizer's vocabulary
  - `dtype`: `int64` (token indices)
- **Common storage formats**:
  - CSV/TSV with text column
  - JSONL (one JSON object per line)
  - Plain text files (one document per file)
  - Hugging Face Datasets
- **Preprocessing typical steps**:
  - Tokenization (BERT, GPT, etc.)
  - Padding / truncation
  - Adding special tokens (CLS, SEP, etc.)
- **Example `data_type`: `"text"`**

## 3. Time Series

- **Description**: Ordered sequences of numerical values, often with multiple features per time step.
- **Typical tasks**: Forecasting, classification, anomaly detection, regression.
- **Input spec fields**:
  - `shape`: `[time_steps, features]`
  - `dtype`: `float32`
  - `sequence_length`: can be fixed or variable
- **Common storage formats**:
  - CSV/TSV (rows = time steps, columns = features)
  - NPZ / NumPy arrays
  - Parquet / Feather
- **Preprocessing typical steps**:
  - Normalization / standardization (per feature or global)
  - Sliding window creation
  - Handling missing values
- **Example `data_type`: `"time_series"`**

## 4. Tabular

- **Description**: Structured tables with rows (samples) and columns (features).
- **Typical tasks**: Classification, regression, ranking, clustering.
- **Input spec fields**:
  - `num_features`: number of feature columns (excluding target)
  - `dtype`: `float32` for numeric, `int64` for categorical
- **Common storage formats**:
  - CSV, Excel, Parquet
  - Pandas DataFrame (pickle)
  - SQL tables
- **Preprocessing typical steps**:
  - Missing value imputation
  - Feature scaling (standard, min‑max, robust)
  - One‑hot encoding for categoricals
- **Example `data_type`: `"tabular"`**

## 5. Audio

- **Description**: Sound waveforms or spectrograms.
- **Typical tasks**: Classification, speech recognition, source separation, music generation.
- **Input spec fields**:
  - `sample_rate`: Hz (e.g., 16000, 44100)
  - `duration`: seconds (optional)
  - `shape`: `[channels, samples]` for raw waveform, `[freq_bins, time_frames]` for spectrogram
  - `dtype`: `float32`
- **Common storage formats**:
  - WAV, MP3, FLAC files
  - Numpy arrays
  - Librosa‑compatible formats
- **Preprocessing typical steps**:
  - Resampling
  - STFT (spectrogram)
  - Mel‑scale conversion
  - Normalization
- **Example `data_type`: `"audio"`**

## 6. Video

- **Description**: Sequences of image frames, often with audio track.
- **Typical tasks**: Action recognition, captioning, tracking, generation.
- **Input spec fields**:
  - `shape`: `[frames, channels, height, width]` or `[channels, frames, height, width]`
  - `fps`: frames per second
  - `duration`: seconds
  - `dtype`: `uint8` or `float32`
- **Common storage formats**:
  - MP4, AVI, MKV files
  - Folder of frame images
  - HDF5 / NPZ arrays
- **Preprocessing typical steps**:
  - Frame extraction
  - Resize / crop
  - Temporal subsampling
- **Example `data_type`: `"video"`**

## 7. Multimodal

- **Description**: Combinations of the above (e.g., image + text, video + audio).
- **Typical tasks**: VQA, cross‑modal retrieval, multimodal classification.
- **Input spec fields**: Usually a dictionary of specs per modality.
- **Common storage formats**: JSON, HDF5, custom directories.
- **Preprocessing typical steps**: Modality‑specific pipelines plus alignment.
- **Example `data_type`: `"multimodal"`**

## Choosing the Right Data Type

- If your input is a single 2D array of pixels → `image`
- If your input is a sequence of tokens → `text`
- If your input is a sequence of numeric vectors ordered by time → `time_series`
- If your input is a fixed‑length vector of features → `tabular`
- If your input is a waveform or spectrogram → `audio`
- If your input is a sequence of image frames → `video`
- If your input combines two or more modalities → `multimodal`

Once you pick the `data_type`, fill the `input_spec` with the relevant fields and choose appropriate `preprocessing` steps.