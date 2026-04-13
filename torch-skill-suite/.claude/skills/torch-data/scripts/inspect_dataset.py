#!/usr/bin/env python3
"""
Inspect a dataset directory/file and attempt to infer its format.

Usage:
    python inspect_dataset.py --path /path/to/dataset --data_type image --task_type classification
    python inspect_dataset.py --path /path/to/dataset --data_type tabular --task_type regression

Outputs a YAML snippet containing inferred_format_spec plus lightweight
inspection metadata.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    yaml = None


class _NoAliasSafeDumper(yaml.SafeDumper if yaml is not None else object):
    def ignore_aliases(self, data):
        return True


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
TEXT_EXTS = {".txt", ".md", ".text"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
TABULAR_EXTS = {".csv", ".tsv", ".jsonl", ".json", ".parquet"}
MANIFEST_EXTS = {".csv", ".tsv", ".jsonl", ".json"}
TARGET_CANDIDATES = {"label", "labels", "target", "targets", "class", "class_id", "y"}
IMAGE_FIELD_CANDIDATES = {"image", "image_path", "img", "img_path", "filepath", "file_path"}
TEXT_FIELD_CANDIDATES = {"text", "caption", "description", "prompt", "question", "transcript"}
AUDIO_FIELD_CANDIDATES = {"audio", "audio_path", "wav_path", "sound_path"}
VIDEO_FIELD_CANDIDATES = {"video", "video_path", "clip_path"}


def dump_output(data):
    if yaml is not None:
        print("# Dataset inspection result")
        print(yaml.dump(data, Dumper=_NoAliasSafeDumper, sort_keys=False, default_flow_style=False, allow_unicode=True))
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


def make_result(format_type, details, confidence="medium", warnings=None, observed_fields=None, missing_information=None):
    return {
        "inferred_format_spec": {
            "format_type": format_type,
            "details": details,
        },
        "confidence": confidence,
        "warnings": warnings or [],
        "observed_fields": observed_fields or {},
        "missing_information": missing_information or [],
    }


def list_non_hidden_dirs(path):
    return [p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")]


def list_non_hidden_files(path):
    return [p for p in path.iterdir() if p.is_file() and not p.name.startswith(".")]


def files_with_suffixes(path, suffixes):
    return [p for p in list_non_hidden_files(path) if p.suffix.lower() in suffixes]


def dir_has_files_with_suffixes(path, suffixes):
    return any(p.is_file() and p.suffix.lower() in suffixes for p in path.iterdir())


def sample_sequence(items, limit=5):
    return list(items[:limit])


def maybe_split_dirs(path):
    dirs = list_non_hidden_dirs(path)
    split_names = [d.name for d in dirs if d.name.lower() in {"train", "val", "valid", "validation", "test"}]
    return dirs, split_names


def choose_manifest_file(path):
    if path.is_file() and path.suffix.lower() in MANIFEST_EXTS:
        return path

    if not path.is_dir():
        return None

    manifest_candidates = []
    for file_path in list_non_hidden_files(path):
        if file_path.suffix.lower() in MANIFEST_EXTS:
            score = 0
            lower = file_path.name.lower()
            if "manifest" in lower:
                score += 3
            if any(token in lower for token in ["train", "val", "test", "metadata", "labels", "samples"]):
                score += 2
            manifest_candidates.append((score, file_path))

    if not manifest_candidates:
        return None

    manifest_candidates.sort(key=lambda item: (-item[0], item[1].name))
    return manifest_candidates[0][1]


def sniff_delimiter(sample_text):
    delimiters = [",", "\t", ";", "|"]
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=delimiters)
        return dialect.delimiter
    except csv.Error:
        if "\t" in sample_text and sample_text.count("\t") >= sample_text.count(","):
            return "\t"
        return ","


def read_csv_preview(path, max_rows=5):
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample_text = f.read(4096)
        f.seek(0)
        delimiter = sniff_delimiter(sample_text)
        try:
            has_header = csv.Sniffer().has_header(sample_text)
        except csv.Error:
            has_header = True
        reader = csv.DictReader(f, delimiter=delimiter) if has_header else csv.reader(f, delimiter=delimiter)
        rows = []
        columns = None
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            if isinstance(row, dict):
                if columns is None:
                    columns = list(row.keys())
                rows.append(row)
            else:
                if columns is None:
                    columns = [f"column_{i}" for i in range(len(row))]
                rows.append({columns[i]: value for i, value in enumerate(row)})
    return {
        "delimiter": "\\t" if delimiter == "\t" else delimiter,
        "has_header": has_header,
        "columns": columns or [],
        "rows": rows,
    }


def read_jsonl_preview(path, max_rows=5):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    columns = []
    if rows and isinstance(rows[0], dict):
        columns = list(rows[0].keys())
    return {"columns": columns, "rows": rows}


def read_json_preview(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return {"kind": "list_of_objects", "columns": list(payload[0].keys()), "rows": payload[:5]}
    if isinstance(payload, dict):
        if payload.get("annotations") and payload.get("images"):
            return {"kind": "coco_like", "columns": ["images", "annotations", "categories"], "rows": []}
        return {"kind": "object", "keys": list(payload.keys()), "rows": []}
    return {"kind": type(payload).__name__, "columns": [], "rows": []}


def infer_target_columns(columns):
    candidates = [col for col in columns if col and col.lower() in TARGET_CANDIDATES]
    if candidates:
        return candidates
    fuzzy = [col for col in columns if col and any(token in col.lower() for token in ["label", "target", "class", "price", "score", "rating", "value"])]
    if fuzzy:
        return fuzzy[:3]
    if columns:
        return [columns[-1]]
    return []


def infer_field_types(rows, columns):
    numeric_columns = []
    textual_columns = []
    path_like_columns = []
    for column in columns:
        values = [row.get(column) for row in rows if isinstance(row, dict) and row.get(column) not in (None, "")]
        if not values:
            continue
        if all(_is_number_like(value) for value in values):
            numeric_columns.append(column)
        else:
            textual_columns.append(column)
        if any(_looks_like_path(value) for value in values):
            path_like_columns.append(column)
    return {
        "numeric_columns": numeric_columns,
        "textual_columns": textual_columns,
        "path_like_columns": path_like_columns,
    }


def _is_number_like(value):
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, str):
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _looks_like_path(value):
    if not isinstance(value, str):
        return False
    suffix = Path(value).suffix.lower()
    return "/" in value or "\\" in value or suffix in IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS


def _column_matches(columns, token):
    token = token.lower()
    return any(token in column.lower() for column in columns)


def detect_manifest_modalities(columns, rows):
    lowered = {col.lower(): col for col in columns}
    modalities = []

    if any(key in lowered for key in IMAGE_FIELD_CANDIDATES) or any(_column_matches(columns, token) for token in ["image", "img"]):
        modalities.append("image")
    if any(key in lowered for key in TEXT_FIELD_CANDIDATES) or any(_column_matches(columns, token) for token in ["text", "caption", "description", "prompt", "transcript"]):
        modalities.append("text")
    if any(key in lowered for key in AUDIO_FIELD_CANDIDATES) or any(_column_matches(columns, token) for token in ["audio", "wav", "sound"]):
        modalities.append("audio")
    if any(key in lowered for key in VIDEO_FIELD_CANDIDATES) or any(_column_matches(columns, token) for token in ["video", "clip"]):
        modalities.append("video")

    return sorted(set(modalities))


def inspect_manifest_for_modalities(path):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        preview = read_csv_preview(path)
        columns = preview["columns"]
        rows = preview["rows"]
        format_type = "CSV" if suffix == ".csv" else "TSV"
    elif suffix == ".jsonl":
        preview = read_jsonl_preview(path)
        columns = preview["columns"]
        rows = preview["rows"]
        format_type = "JSONL"
    else:
        preview = read_json_preview(path)
        columns = preview.get("columns", [])
        rows = preview.get("rows", [])
        format_type = "JSON"

    modalities = detect_manifest_modalities(columns, rows)
    label_candidates = infer_target_columns(columns)
    observed_extensions = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for value in row.values():
            if isinstance(value, str):
                suffix_value = Path(value).suffix.lower()
                if suffix_value:
                    observed_extensions.add(suffix_value)

    return {
        "columns": columns,
        "modalities": modalities,
        "label_candidates": label_candidates,
        "observed_extensions": observed_extensions,
    }, format_type


def inspect_image_classification(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    if path.is_dir():
        dirs, split_names = maybe_split_dirs(path)
        if split_names:
            for split_dir in dirs:
                class_dirs = [d.name for d in list_non_hidden_dirs(split_dir) if dir_has_files_with_suffixes(d, IMAGE_EXTS)]
                if class_dirs:
                    return make_result(
                        "ImageFolder",
                        {
                            "structure": "class folders inside split folders",
                            "split_dirs": split_names,
                            "class_folders": class_dirs,
                            "extensions": sorted(IMAGE_EXTS),
                        },
                        confidence="high",
                        observed_fields={"split_dirs": split_names, "sample_classes": sample_sequence(class_dirs)},
                    ), None

        class_dirs = [d.name for d in dirs if dir_has_files_with_suffixes(d, IMAGE_EXTS)]
        if class_dirs:
            return make_result(
                "ImageFolder",
                {
                    "structure": "class folders inside root",
                    "class_folders": class_dirs,
                    "extensions": sorted(IMAGE_EXTS),
                },
                confidence="high",
                observed_fields={"sample_classes": sample_sequence(class_dirs)},
            ), None

        image_files = files_with_suffixes(path, IMAGE_EXTS)
        if image_files:
            return make_result(
                "FlatImageDirectory",
                {
                    "structure": "flat directory of images",
                    "extensions": sorted({f.suffix.lower() for f in image_files}),
                },
                confidence="medium",
                warnings=["No explicit label source detected; labels may come from a separate manifest."],
            ), None

    return None, "Could not infer image classification format"


def inspect_image_segmentation(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    if path.is_dir():
        json_files = files_with_suffixes(path, {".json"})
        if json_files:
            return make_result(
                "COCO",
                {
                    "annotation_file": str(json_files[0]),
                    "task_hint": "segmentation",
                },
                confidence="medium",
                warnings=["COCO JSON detected; confirm whether segmentation polygons or masks are used."],
                missing_information=["Need confirmation of mask/polygon encoding and image root path."],
            ), None

        dirs = [d for d in path.rglob("*") if d.is_dir()]
        image_dirs = [d for d in dirs if "image" in d.name.lower()]
        mask_dirs = [d for d in dirs if any(token in d.name.lower() for token in ["mask", "label", "labels", "annotation", "annotations"])]
        for image_dir in image_dirs:
            for mask_dir in mask_dirs:
                if image_dir == mask_dir:
                    continue
                mask_files = [p for p in mask_dir.rglob("*") if p.is_file()]
                image_files = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
                if image_files and mask_files:
                    return make_result(
                        "ImageMaskPairs",
                        {
                            "image_dir": str(image_dir),
                            "mask_dir": str(mask_dir),
                            "pairing_rule": "Match image and mask by relative path and basename.",
                            "image_extensions": sorted({p.suffix.lower() for p in image_files}),
                            "mask_extensions": sorted({p.suffix.lower() for p in mask_files}),
                        },
                        confidence="high",
                        observed_fields={
                            "image_count_sampled": min(len(image_files), 20),
                            "mask_count_sampled": min(len(mask_files), 20),
                        },
                        missing_information=["Need confirmation of mask encoding (single-channel class indices vs RGB colors)."],
                    ), None

    return None, "Could not infer image segmentation format"


def inspect_detection(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    if path.is_dir():
        image_dirs = [d for d in path.iterdir() if d.is_dir() and "image" in d.name.lower()]
        label_dirs = [d for d in path.iterdir() if d.is_dir() and "label" in d.name.lower()]
        if image_dirs and label_dirs:
            return make_result(
                "YOLO",
                {
                    "image_dir": str(image_dirs[0]),
                    "label_dir": str(label_dirs[0]),
                    "label_suffix": ".txt",
                },
                confidence="high",
            ), None

        json_files = files_with_suffixes(path, {".json"})
        if json_files:
            return make_result(
                "COCO",
                {
                    "annotation_file": str(json_files[0]),
                },
                confidence="medium",
                missing_information=["Need confirmation that the JSON file contains detection boxes rather than segmentation polygons only."],
            ), None

    return None, "Could not infer detection format"


def inspect_text_classification(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    if path.suffix.lower() == ".csv":
        preview = read_csv_preview(path)
        return make_result(
            "CSV",
            {
                "delimiter": preview["delimiter"],
                "encoding": "utf-8",
                "columns": preview["columns"],
            },
            confidence="high",
            observed_fields={"target_candidates": infer_target_columns(preview["columns"])},
        ), None

    if path.suffix.lower() == ".jsonl":
        preview = read_jsonl_preview(path)
        return make_result(
            "JSONL",
            {
                "encoding": "utf-8",
                "columns": preview["columns"],
            },
            confidence="high",
            observed_fields={"target_candidates": infer_target_columns(preview["columns"])},
        ), None

    if path.is_dir():
        txt_files = files_with_suffixes(path, TEXT_EXTS)
        if txt_files:
            return make_result(
                "TextFiles",
                {
                    "extension": txt_files[0].suffix.lower(),
                    "encoding": "utf-8",
                },
                confidence="medium",
                warnings=["No explicit label mapping detected; labels may come from filenames or a separate manifest."],
            ), None

    return None, "Could not infer text classification format"


def inspect_time_series(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    if path.suffix.lower() == ".npz":
        return make_result(
            "NPZ",
            {
                "keys": ["X", "y"],
            },
            confidence="medium",
            warnings=["NPZ detected by suffix only; inspect array keys to confirm feature/target naming."],
        ), None

    if path.suffix.lower() in {".csv", ".tsv"}:
        preview = read_csv_preview(path)
        return make_result(
            "CSV" if path.suffix.lower() == ".csv" else "TSV",
            {
                "delimiter": preview["delimiter"],
                "has_header": preview["has_header"],
                "columns": preview["columns"],
            },
            confidence="medium",
            observed_fields=infer_field_types(preview["rows"], preview["columns"]),
            missing_information=["Need confirmation of timestamp column, grouping key, and forecast target if applicable."],
        ), None

    return None, "Could not infer time series format"


def inspect_tabular(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            preview = read_csv_preview(path)
            inferred = infer_field_types(preview["rows"], preview["columns"])
            return make_result(
                "CSV" if suffix == ".csv" else "TSV",
                {
                    "delimiter": preview["delimiter"],
                    "has_header": preview["has_header"],
                    "columns": preview["columns"],
                    "target_candidates": infer_target_columns(preview["columns"]),
                },
                confidence="high",
                observed_fields=inferred,
                missing_information=["Need confirmation of the target column and train/val/test split policy."],
            ), None
        if suffix == ".jsonl":
            preview = read_jsonl_preview(path)
            inferred = infer_field_types(preview["rows"], preview["columns"])
            return make_result(
                "JSONLManifest",
                {
                    "encoding": "utf-8",
                    "columns": preview["columns"],
                    "target_candidates": infer_target_columns(preview["columns"]),
                },
                confidence="medium",
                observed_fields=inferred,
                missing_information=["Need confirmation of target column and categorical feature handling."],
            ), None
        if suffix == ".json":
            preview = read_json_preview(path)
            columns = preview.get("columns", [])
            return make_result(
                "JSONManifest",
                {
                    "columns": columns,
                    "target_candidates": infer_target_columns(columns),
                },
                confidence="low",
                warnings=["JSON structure detected; verify whether rows represent tabular records or nested documents."],
            ), None
        if suffix == ".parquet":
            return make_result(
                "Parquet",
                {
                    "file": str(path),
                },
                confidence="low",
                warnings=["Parquet detected by suffix only; deeper schema inspection requires optional parquet tooling."],
                missing_information=["Need schema details, target column, and split policy."],
            ), None

    if path.is_dir():
        tabular_files = [p for p in list_non_hidden_files(path) if p.suffix.lower() in TABULAR_EXTS]
        if tabular_files:
            suffixes = sorted({p.suffix.lower() for p in tabular_files})
            return make_result(
                "TabularSplitFiles",
                {
                    "files": [p.name for p in sample_sequence(sorted(tabular_files, key=lambda p: p.name), 10)],
                    "extensions": suffixes,
                },
                confidence="medium",
                observed_fields={"file_count": len(tabular_files)},
                missing_information=["Need confirmation of which files correspond to train/val/test and which column is the target."],
            ), None

    return None, "Could not infer tabular format"


def inspect_audio_classification(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    if path.is_dir():
        manifest = choose_manifest_file(path)
        if manifest is not None:
            preview, format_type = inspect_manifest_for_modalities(manifest)
            observed_audio_exts = sorted(preview["observed_extensions"] & AUDIO_EXTS)
            if "audio" in preview["modalities"] or observed_audio_exts:
                return make_result(
                    "AudioFolderWithMetadata",
                    {
                        "manifest_path": str(manifest),
                        "manifest_format": format_type,
                        "audio_extensions": observed_audio_exts,
                        "label_source": "metadata manifest",
                        "fields": preview["columns"],
                    },
                    confidence="medium",
                    observed_fields={"modalities": preview["modalities"], "label_candidates": preview["label_candidates"]},
                    missing_information=["Need confirmation of the audio root path if manifest paths are relative."],
                ), None

        class_dirs = [d for d in list_non_hidden_dirs(path) if dir_has_files_with_suffixes(d, AUDIO_EXTS)]
        if class_dirs:
            class_names = [d.name for d in class_dirs]
            observed_exts = sorted({p.suffix.lower() for d in class_dirs for p in d.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS})
            return make_result(
                "AudioFolder",
                {
                    "structure": "class folders inside root",
                    "audio_extensions": observed_exts,
                    "class_folders": class_names,
                    "label_source": "directory names",
                },
                confidence="high",
                observed_fields={"sample_classes": sample_sequence(class_names)},
            ), None

    return None, "Could not infer audio classification format"


def inspect_video_classification(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    if path.is_dir():
        manifest = choose_manifest_file(path)
        if manifest is not None:
            preview, format_type = inspect_manifest_for_modalities(manifest)
            observed_video_exts = sorted(preview["observed_extensions"] & VIDEO_EXTS)
            if "video" in preview["modalities"] or observed_video_exts:
                return make_result(
                    "VideoManifest",
                    {
                        "manifest_path": str(manifest),
                        "manifest_format": format_type,
                        "video_extensions": observed_video_exts,
                        "label_source": "metadata manifest",
                        "fields": preview["columns"],
                    },
                    confidence="medium",
                    observed_fields={"modalities": preview["modalities"], "label_candidates": preview["label_candidates"]},
                    missing_information=["Need confirmation of clip sampling strategy and root path if paths are relative."],
                ), None

        class_dirs = [d for d in list_non_hidden_dirs(path) if dir_has_files_with_suffixes(d, VIDEO_EXTS)]
        if class_dirs:
            class_names = [d.name for d in class_dirs]
            observed_exts = sorted({p.suffix.lower() for d in class_dirs for p in d.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS})
            return make_result(
                "VideoFolder",
                {
                    "structure": "class folders inside root",
                    "video_extensions": observed_exts,
                    "class_folders": class_names,
                    "label_source": "directory names",
                },
                confidence="high",
                observed_fields={"sample_classes": sample_sequence(class_names)},
            ), None

        frame_dirs = [d for d in list_non_hidden_dirs(path) if dir_has_files_with_suffixes(d, IMAGE_EXTS)]
        if frame_dirs:
            return make_result(
                "FrameDirectory",
                {
                    "structure": "directories containing extracted frame images",
                    "image_extensions": sorted({p.suffix.lower() for d in frame_dirs for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}),
                },
                confidence="medium",
                warnings=["Frame directories detected; confirm whether labels come from folder names or a manifest."],
            ), None

    return None, "Could not infer video classification format"


def inspect_multimodal(path):
    path = Path(path)
    if not path.exists():
        return None, "Path does not exist"

    manifest = choose_manifest_file(path)
    if manifest is None:
        return None, "Could not locate a manifest file for multimodal inspection"

    preview, format_type = inspect_manifest_for_modalities(manifest)
    if len(preview["modalities"]) < 2:
        return None, "Manifest found but fewer than two modalities were detected"

    warnings = []
    if not preview["label_candidates"]:
        warnings.append("No obvious label column detected; downstream task may require manual confirmation.")

    return make_result(
        f"{format_type}Manifest" if not format_type.endswith("Manifest") else format_type,
        {
            "manifest_path": str(manifest),
            "modalities": preview["modalities"],
            "fields": preview["columns"],
            "label_candidates": preview["label_candidates"],
            "pairing_rule": "Manifest-driven alignment: each row defines one multimodal sample.",
        },
        confidence="high",
        warnings=warnings,
        observed_fields={"modalities": preview["modalities"], "observed_extensions": sorted(preview["observed_extensions"])},
        missing_information=["Need confirmation of relative root directories if manifest paths are not absolute."],
    ), None


INSPECTORS = {
    ("image", "classification"): inspect_image_classification,
    ("image", "detection"): inspect_detection,
    ("image", "segmentation"): inspect_image_segmentation,
    ("text", "classification"): inspect_text_classification,
    ("time_series", "classification"): inspect_time_series,
    ("time_series", "regression"): inspect_time_series,
    ("tabular", "classification"): inspect_tabular,
    ("tabular", "regression"): inspect_tabular,
    ("audio", "classification"): inspect_audio_classification,
    ("video", "classification"): inspect_video_classification,
    ("multimodal", "classification"): inspect_multimodal,
}


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset and infer format.")
    parser.add_argument("--path", required=True, help="Path to dataset")
    parser.add_argument("--data_type", required=True, choices=["image", "text", "time_series", "tabular", "audio", "video", "multimodal"])
    parser.add_argument(
        "--task_type",
        required=True,
        choices=["classification", "detection", "segmentation", "regression", "generation", "translation", "clustering", "reinforcement_learning"],
    )
    args = parser.parse_args()

    inspector = INSPECTORS.get((args.data_type, args.task_type))
    if inspector is None:
        print(f"Error: Inspection for {args.data_type} {args.task_type} not implemented")
        sys.exit(1)

    result, error = inspector(args.path)
    if error:
        print(f"Error: {error}")
        sys.exit(1)

    dump_output(result)


if __name__ == "__main__":
    main()
