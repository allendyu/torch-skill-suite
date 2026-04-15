import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "inspect_dataset.py"


def run_inspect(path, data_type, task_type):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--path", str(path), "--data_type", data_type, "--task_type", task_type],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in proc.stdout.splitlines() if not line.startswith("# ")]
    payload = "\n".join(lines).strip()
    return json.loads(payload) if payload.startswith("{") else None, payload


def test_yolo_detection_high_confidence(tmp_path):
    dataset = tmp_path / "detection"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    (images / "sample.jpg").write_text("img", encoding="utf-8")
    (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    _, payload = run_inspect(dataset, "image", "detection")
    assert "format_type: YOLO" in payload
    assert "confidence: high" in payload
    assert "valid_files: 1" in payload


def test_yolo_detection_downgrades_on_bad_labels(tmp_path):
    dataset = tmp_path / "detection_bad"
    images = dataset / "images"
    labels = dataset / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    (images / "sample.jpg").write_text("img", encoding="utf-8")
    (labels / "sample.txt").write_text("0 0.5 0.5\n", encoding="utf-8")

    _, payload = run_inspect(dataset, "image", "detection")
    assert "format_type: YOLO" in payload
    assert "confidence: low" in payload or "confidence: medium" in payload
    assert "malformed" in payload.lower()


def test_coco_detection_is_distinguished(tmp_path):
    dataset = tmp_path / "coco_detection"
    dataset.mkdir()
    annotation = {
        "images": [{"id": 1, "file_name": "sample.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
        "categories": [{"id": 1, "name": "object"}],
    }
    (dataset / "annotations.json").write_text(json.dumps(annotation), encoding="utf-8")

    _, payload = run_inspect(dataset, "image", "detection")
    assert "format_type: COCO_detection" in payload
    assert "task_hint: detection" in payload


def test_coco_segmentation_is_distinguished(tmp_path):
    dataset = tmp_path / "coco_segmentation"
    dataset.mkdir()
    annotation = {
        "images": [{"id": 1, "file_name": "sample.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, 1, 1, 2, 2]]}],
        "categories": [{"id": 1, "name": "road"}],
    }
    (dataset / "annotations.json").write_text(json.dumps(annotation), encoding="utf-8")

    _, payload = run_inspect(dataset, "image", "segmentation")
    assert "COCO_segmentation" in payload or "format_type: COCO" in payload
    assert "task_hint: segmentation" in payload


def test_image_mask_pairing_reports_incomplete_pairs(tmp_path):
    dataset = tmp_path / "segmentation"
    image_dir = dataset / "images"
    mask_dir = dataset / "masks"
    (image_dir / "train").mkdir(parents=True)
    (mask_dir / "train").mkdir(parents=True)
    (image_dir / "train" / "a.jpg").write_text("img", encoding="utf-8")
    (image_dir / "train" / "b.jpg").write_text("img", encoding="utf-8")
    (mask_dir / "train" / "a.png").write_text("mask", encoding="utf-8")

    _, payload = run_inspect(dataset, "image", "segmentation")
    assert "format_type: ImageMaskPairs" in payload
    assert "pairing_rate" in payload
    assert "incomplete" in payload.lower()
