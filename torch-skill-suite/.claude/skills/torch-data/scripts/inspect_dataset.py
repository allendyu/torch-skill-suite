#!/usr/bin/env python3
"""
Inspect a dataset directory/file and attempt to infer its format.

Usage:
    python inspect_dataset.py --path /path/to/dataset --data_type image --task_type classification
    python inspect_dataset.py --path /path/to/dataset --data_type text --task_type classification

Outputs a YAML snippet for inferred_format_spec.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml

def inspect_image_classification(path):
    """Inspect image classification dataset, guess format."""
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    # Check for ImageFolder structure: class folders inside split folders
    # or flat directory with images
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if subdirs:
        # Possibly class folders
        class_names = [d.name for d in subdirs if not d.name.startswith('.')]
        # Check if subdirectories contain image files
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        for cls in class_names[:3]:  # sample first few
            cls_dir = path / cls
            if cls_dir.exists():
                images = [f for f in cls_dir.iterdir() if f.suffix.lower() in image_exts]
                if images:
                    return {
                        "format_type": "ImageFolder",
                        "details": {
                            "structure": "class folders inside root",
                            "extensions": list(image_exts),
                            "class_folders": class_names
                        }
                    }, None
    # Check for flat directory of images
    image_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    if image_files:
        return {
            "format_type": "FlatImageDirectory",
            "details": {
                "structure": "flat directory of images",
                "extensions": ['.jpg', '.jpeg', '.png']
            }
        }, None

    return None, "Could not infer image classification format"

def inspect_text_classification(path):
    """Inspect text classification dataset, guess format."""
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    # Check for CSV
    if path.suffix.lower() == '.csv':
        return {
            "format_type": "CSV",
            "details": {
                "delimiter": ",",
                "quotechar": '"',
                "encoding": "utf-8"
            }
        }, None
    # Check for JSONL
    if path.suffix.lower() == '.jsonl':
        return {
            "format_type": "JSONL",
            "details": {
                "encoding": "utf-8"
            }
        }, None
    # Check for directory of text files
    if path.is_dir():
        txt_files = [f for f in path.iterdir() if f.is_file() and f.suffix.lower() == '.txt']
        if txt_files:
            return {
                "format_type": "TextFiles",
                "details": {
                    "extension": ".txt",
                    "encoding": "utf-8"
                }
            }, None

    return None, "Could not infer text classification format"

def inspect_time_series(path):
    """Inspect time series dataset, guess format."""
    path = Path(path)
    # Check for NPZ
    if path.suffix.lower() == '.npz':
        return {
            "format_type": "NPZ",
            "details": {
                "keys": ["X", "y"]  # guess
            }
        }, None
    # Check for CSV
    if path.suffix.lower() == '.csv':
        return {
            "format_type": "CSV",
            "details": {
                "delimiter": ",",
                "has_header": True
            }
        }, None
    return None, "Could not infer time series format"

def inspect_detection(path):
    """Inspect object detection dataset, guess format."""
    path = Path(path)
    # Check for YOLO structure: images and labels subdirectories
    if path.is_dir():
        # Look for images folder
        image_dirs = [d for d in path.iterdir() if d.is_dir() and 'image' in d.name.lower()]
        label_dirs = [d for d in path.iterdir() if d.is_dir() and 'label' in d.name.lower()]
        if image_dirs and label_dirs:
            return {
                "format_type": "YOLO",
                "details": {
                    "image_dir": str(image_dirs[0]),
                    "label_dir": str(label_dirs[0]),
                    "label_suffix": ".txt"
                }
            }, None
        # Check for COCO style
        json_files = [f for f in path.iterdir() if f.suffix.lower() == '.json']
        if json_files:
            return {
                "format_type": "COCO",
                "details": {
                    "annotation_file": str(json_files[0])
                }
            }, None
    return None, "Could not infer detection format"

def main():
    parser = argparse.ArgumentParser(description="Inspect dataset and infer format.")
    parser.add_argument("--path", required=True, help="Path to dataset")
    parser.add_argument("--data_type", required=True, choices=["image", "text", "time_series", "tabular", "audio", "video", "detection"])
    parser.add_argument("--task_type", required=True, choices=["classification", "detection", "segmentation", "regression", "generation", "translation"])
    args = parser.parse_args()

    # Dispatch based on data_type and task_type
    if args.data_type == "image":
        if args.task_type == "classification":
            result, error = inspect_image_classification(args.path)
        elif args.task_type == "detection":
            result, error = inspect_detection(args.path)
        else:
            error = f"Inspection for {args.data_type} {args.task_type} not implemented"
            result = None
    elif args.data_type == "text" and args.task_type == "classification":
        result, error = inspect_text_classification(args.path)
    elif args.data_type == "time_series" and args.task_type in ["regression", "classification"]:
        result, error = inspect_time_series(args.path)
    else:
        error = f"Inspection for {args.data_type} {args.task_type} not implemented"
        result = None

    if error:
        print(f"Error: {error}")
        sys.exit(1)

    # Output YAML
    print("# Inferred format spec")
    print(yaml.dump({"inferred_format_spec": result}, default_flow_style=False))

if __name__ == "__main__":
    main()