"""Audit a YOLO detection dataset before training.

The report focuses on errors that can silently distort a small MVP dataset:
missing pairs, invalid normalized boxes, class counts, exact duplicates, and
likely Roboflow source-frame leakage across splits.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

from factoryguard.config import read_yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)
ENCODED_IMAGE_SUFFIX = re.compile(r"_(?:jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)


def display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def source_key(path):
    """Return a conservative source-image key for common Roboflow exports."""
    stem = ROBOFLOW_SUFFIX.sub("", Path(path).stem)
    return ENCODED_IMAGE_SUFFIX.sub("", stem).lower()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dataset(data_yaml):
    data_yaml = Path(data_yaml).resolve()
    data = read_yaml(data_yaml)
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("data.yaml must define non-empty class names")
    names = {int(key): str(value) for key, value in names.items()}
    root = (data_yaml.parent / data.get("path", ".")).resolve()
    splits = {}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if value:
            values = [value] if isinstance(value, str) else value
            if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
                raise ValueError(f"{split} must be a path or a list of paths")
            splits[split] = [(root / item).resolve() for item in values]
    if "train" not in splits or "val" not in splits:
        raise ValueError("data.yaml must define train and val splits")
    return data_yaml, names, splits


def parse_label(path, names):
    counts = Counter()
    issues = []
    text = Path(path).read_text(encoding="utf-8-sig")
    if not text.strip():
        return counts, issues, True
    for line_number, raw in enumerate(text.splitlines(), 1):
        fields = raw.split()
        detail = {"label": display_path(path), "line": line_number}
        if len(fields) != 5:
            issues.append({**detail, "code": "label_field_count", "message": "Expected 5 fields"})
            continue
        try:
            class_value = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            issues.append({**detail, "code": "label_not_numeric", "message": "All fields must be numeric"})
            continue
        if not class_value.is_integer() or int(class_value) not in names:
            issues.append({**detail, "code": "unknown_class", "message": f"Unknown class ID {fields[0]}"})
            continue
        if not all(math.isfinite(value) for value in values):
            issues.append({**detail, "code": "non_finite_box", "message": "Box values must be finite"})
            continue
        x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            issues.append({**detail, "code": "box_out_of_range", "message": "YOLO xywh must be normalized"})
            continue
        epsilon = 1e-6
        if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon or y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:
            issues.append({**detail, "code": "box_crosses_image", "message": "Box extends beyond the image"})
            continue
        counts[int(class_value)] += 1
    return counts, issues, False


def audit_dataset(data_yaml):
    data_yaml, names, splits = resolve_dataset(data_yaml)
    issues = []
    split_reports = {}
    hashes = defaultdict(list)
    sources = defaultdict(list)
    total_counts = Counter()

    for split, image_dirs in splits.items():
        label_dirs = [image_dir.parent / "labels" for image_dir in image_dirs]
        images = []
        labels = []
        image_keys = {}
        label_keys = {}
        for directory_index, (image_dir, label_dir) in enumerate(zip(image_dirs, label_dirs)):
            directory_images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
            directory_labels = sorted(label_dir.rglob("*.txt")) if label_dir.is_dir() else []
            images.extend(directory_images)
            labels.extend(directory_labels)
            image_keys.update({(directory_index, path.relative_to(image_dir).with_suffix("")): path
                               for path in directory_images})
            label_keys.update({(directory_index, path.relative_to(label_dir).with_suffix("")): path
                               for path in directory_labels})
        missing = sorted(image_keys.keys() - label_keys.keys(), key=str)
        orphan = sorted(label_keys.keys() - image_keys.keys(), key=str)
        for key in missing:
            issues.append({"severity": "error", "code": "missing_label", "split": split,
                           "image": display_path(image_keys[key])})
        for key in orphan:
            issues.append({"severity": "error", "code": "orphan_label", "split": split,
                           "label": display_path(label_keys[key])})

        class_counts = Counter()
        invalid = 0
        empty = 0
        for key in image_keys.keys() & label_keys.keys():
            counts, label_issues, is_empty = parse_label(label_keys[key], names)
            class_counts.update(counts)
            empty += int(is_empty)
            invalid += len(label_issues)
            issues.extend({"severity": "error", "split": split, **item} for item in label_issues)
        for image in images:
            hashes[sha256(image)].append((split, display_path(image)))
            sources[source_key(image)].append((split, display_path(image)))
        total_counts.update(class_counts)
        split_reports[split] = {
            "image_dirs": [display_path(path) for path in image_dirs],
            "label_dirs": [display_path(path) for path in label_dirs],
            "images": len(images),
            "labels": len(labels),
            "paired": len(image_keys.keys() & label_keys.keys()),
            "missing_labels": len(missing),
            "orphan_labels": len(orphan),
            "empty_labels": empty,
            "invalid_annotations": invalid,
            "class_instances": {str(class_id): class_counts[class_id] for class_id in sorted(names)},
        }

    def cross_split_groups(groups):
        output = []
        for key, entries in groups.items():
            if len({split for split, _ in entries}) > 1:
                output.append({"key": key, "files": [{"split": split, "path": path} for split, path in entries]})
        return sorted(output, key=lambda item: item["key"])

    exact_duplicates = cross_split_groups(hashes)
    source_overlaps = cross_split_groups(sources)
    for group in exact_duplicates:
        issues.append({"severity": "error", "code": "exact_duplicate_across_splits", "key": group["key"]})
    for group in source_overlaps:
        issues.append({"severity": "warning", "code": "possible_source_overlap", "key": group["key"]})

    return {
        "status": "fail" if any(item["severity"] == "error" for item in issues) else "pass",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": display_path(data_yaml),
        "class_names": {str(key): names[key] for key in sorted(names)},
        "splits": split_reports,
        "totals": {
            "images": sum(item["images"] for item in split_reports.values()),
            "class_instances": {str(class_id): total_counts[class_id] for class_id in sorted(names)},
            "errors": sum(item["severity"] == "error" for item in issues),
            "warnings": sum(item["severity"] == "warning" for item in issues),
        },
        "exact_duplicate_groups_across_splits": exact_duplicates,
        "possible_source_overlap_groups_across_splits": source_overlaps,
        "issues": issues,
    }


def print_summary(report):
    print(f"Dataset audit: {report['status'].upper()}")
    print("split       images  paired  empty  invalid  class instances")
    for split, item in report["splits"].items():
        counts = ", ".join(f"{report['class_names'][key]}={value}" for key, value in item["class_instances"].items())
        print(f"{split:<11} {item['images']:>6} {item['paired']:>7} {item['empty_labels']:>6} "
              f"{item['invalid_annotations']:>8}  {counts}")
    totals = report["totals"]
    print(f"Errors: {totals['errors']} | Warnings: {totals['warnings']}")


def main():
    parser = argparse.ArgumentParser(description="Audit YOLO image/label pairs, boxes, classes, and split leakage")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--output", default="reports/dataset-audit.json")
    args = parser.parse_args()
    report = audit_dataset(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)
    print(f"Report: {output.resolve()}")
    raise SystemExit(1 if report["status"] == "fail" else 0)


if __name__ == "__main__":
    main()
