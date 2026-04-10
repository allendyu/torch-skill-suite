# Task Types Supported by torch-data

This document describes the machine‑learning task types that `torch-data` can prepare data for, and how to specify the expected output format in the `output_spec`.

## Overview

The `task_type` field in `data_contract` defines what kind of prediction the model will make. Each task type dictates the structure of labels/targets and influences the choice of loss function, evaluation metrics, and model head.

## 1. Classification

- **Description**: Predict a discrete class label from a fixed set of categories.
- **Output spec**:
  - `type`: `"categorical"`
  - `num_classes`: integer ≥2
  - `label_map`: optional mapping from integer label to class name
- **Label storage**:
  - Integer values (0‑based)
  - One‑hot vectors (multi‑label)
  - Strings mapped via `label_map`
- **Example tasks**: Image classification, sentiment analysis, topic categorization.
- **Common datasets**: MNIST, CIFAR‑10, IMDB, AG News.

## 2. Detection

- **Description**: Localize objects in an image with bounding boxes and classify each box.
- **Output spec**:
  - `type`: `"bounding_box"`
  - `bbox_format`: `"xyxy"` (xmin, ymin, xmax, ymax), `"xywh"` (xmin, ymin, width, height), `"cxcywh"` (center x, center y, width, height)
  - (Optional) `num_classes`: if classification per box is required
- **Label storage**:
  - List of boxes per image, each box = [x1, y1, x2, y2, class_id]
  - YOLO‑style: one `.txt` per image, each line: `class x_center y_center width height` (normalized)
  - COCO JSON: `annotations` list with `bbox` and `category_id`
- **Example tasks**: Object detection, face detection, pedestrian detection.
- **Common datasets**: COCO, Pascal VOC, YOLO‑format custom datasets.

## 3. Segmentation

- **Description**: Assign a class label to each pixel of an image.
- **Output spec**:
  - `type`: `"mask"`
  - `mask_shape`: `[height, width]` of the mask (should match input after preprocessing)
  - `num_classes`: integer ≥2 (including background)
- **Label storage**:
  - Integer‑valued mask images (PNG) where pixel value = class index
  - RGB masks with color‑to‑class mapping
  - Polygon annotations (COCO‑style) that need rasterization
- **Example tasks**: Semantic segmentation, instance segmentation, medical image segmentation.
- **Common datasets**: Cityscapes, ADE20K, PASCAL VOC segmentation, medical imaging (e.g., BraTS).

## 4. Regression

- **Description**: Predict a continuous numerical value (scalar or vector).
- **Output spec**:
  - `type`: `"continuous"`
  - `output_dim`: dimension of the output (1 for scalar regression)
- **Label storage**:
  - Float value(s) per sample (CSV column, numpy array)
  - Can be single column or multiple columns for multi‑output regression
- **Example tasks**: Price prediction, age estimation, score prediction, time‑series forecasting.
- **Common datasets**: Boston Housing, California Housing, custom tabular regression.

## 5. Generation

- **Description**: Produce new data samples similar to the training distribution (unsupervised or conditional).
- **Output spec**:
  - `type`: `"sequence"` or `"image"` or `"audio"` (same as input modality)
  - Often no explicit labels; the target is the input itself (auto‑encoder) or a conditioning signal.
- **Label storage**:
  - For conditional generation: conditioning vector or class label
  - For unconditional generation: no labels needed
- **Example tasks**: Image generation, text generation, music synthesis, style transfer.
- **Common datasets**: CelebA, LSUN, WikiText, MIDI collections.

## 6. Translation

- **Description**: Transform input from one domain to another (e.g., language translation, image‑to‑image translation).
- **Output spec**:
  - `type`: `"sequence"` (for text) or `"image"` (for image translation)
  - Similar to generation but with paired source–target examples.
- **Label storage**:
  - Paired samples: source file and target file, or parallel corpus for text
- **Example tasks**: Machine translation, image colorization, sketch‑to‑photo, style transfer.
- **Common datasets**: WMT, Multi30K, paired image datasets (e.g., maps↔aerial).

## 7. Clustering (Unsupervised)

- **Description**: Group similar samples together without pre‑defined labels.
- **Output spec**:
  - `type`: `"none"` (no labels)
  - Usually no output spec needed; the contract only describes input.
- **Label storage**: None (unsupervised).
- **Example tasks**: Customer segmentation, anomaly detection, feature learning.
- **Common datasets**: Any unlabeled dataset.

## 8. Reinforcement Learning

- **Description**: Learn a policy by interacting with an environment (states, actions, rewards).
- **Output spec**:
  - `type`: `"action"`
  - `action_space`: discrete (integer) or continuous (vector)
- **Label storage**:
  - Usually not stored as static dataset; generated online by environment.
  - Can be logged trajectories (state, action, reward, next_state).
- **Example tasks**: Game playing, robotics control, recommendation systems.
- **Common datasets**: OpenAI Gym replay buffers, custom logs.

## Choosing the Right Task Type

- If you have a fixed set of categories → `classification`
- If you need to draw boxes around objects → `detection`
- If you need per‑pixel labeling → `segmentation`
- If you predict a numeric value → `regression`
- If you want to create new samples → `generation`
- If you transform from one modality/domain to another → `translation`
- If you have no labels and want to find groups → `clustering`
- If you learn from sequential decisions → `reinforcement_learning`

The `output_spec` should reflect the label format your dataset already uses (or will use after preprocessing). `torch-data` will generate Dataset/DataLoader code that respects this spec.