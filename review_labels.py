"""Create a human-review queue for suspicious YOLO detection labels.

The command is intentionally read-only with respect to the dataset. It writes
JSON/CSV findings and annotated contact sheets, but never changes labels.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import cv2
import numpy as np

from audit_dataset import IMAGE_SUFFIXES, display_path, resolve_dataset


COLORS = {
    0: (255, 120, 20),   # forklift: blue/orange in BGR
    1: (40, 220, 220),   # person: yellow
}
SUSPICIOUS_COLOR = (40, 40, 255)


def box_iou(first, second):
    ax1, ay1, ax2, ay2 = first["xyxy"]
    bx1, by1, bx2, by2 = second["xyxy"]
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    union = first["area"] + second["area"] - intersection
    return intersection / union if union > 0 else 0.0


def parse_boxes(label_path, names):
    boxes = []
    for line_number, raw in enumerate(Path(label_path).read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = raw.split()
        if len(fields) != 5:
            continue
        try:
            class_value = float(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError:
            continue
        if (
            not class_value.is_integer()
            or int(class_value) not in names
            or not all(math.isfinite(value) for value in (x, y, width, height))
            or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1)
        ):
            continue
        boxes.append({
            "line": line_number,
            "class_id": int(class_value),
            "class_name": names[int(class_value)],
            "xywh": [x, y, width, height],
            "xyxy": [x - width / 2, y - height / 2, x + width / 2, y + height / 2],
            "area": width * height,
            "issue_codes": [],
        })
    return boxes


def inspect_boxes(boxes, large_area=0.7, duplicate_iou=0.85, cross_class_iou=0.9):
    findings = []
    for box in boxes:
        if box["area"] >= large_area:
            box["issue_codes"].append("large_box")
            findings.append({"code": "large_box", "lines": [box["line"]], "value": box["area"]})

    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            overlap = box_iou(first, second)
            if first["class_id"] == second["class_id"] and overlap >= duplicate_iou:
                code = "duplicate_same_class"
            elif first["class_id"] != second["class_id"] and overlap >= cross_class_iou:
                code = "cross_class_overlap"
            else:
                continue
            first["issue_codes"].append(code)
            second["issue_codes"].append(code)
            findings.append({"code": code, "lines": [first["line"], second["line"]], "value": overlap})

    for outer in boxes:
        if outer["area"] < large_area:
            continue
        for inner in boxes:
            if inner is outer or inner["class_id"] != outer["class_id"] or inner["area"] >= large_area:
                continue
            ix1, iy1, ix2, iy2 = inner["xyxy"]
            ox1, oy1, ox2, oy2 = outer["xyxy"]
            if ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2:
                code = "large_box_with_inner_same_class"
                outer["issue_codes"].append(code)
                inner["issue_codes"].append(code)
                findings.append({"code": code, "lines": [outer["line"], inner["line"]],
                                 "value": outer["area"]})

    for box in boxes:
        box["issue_codes"] = sorted(set(box["issue_codes"]))
    unique = {}
    for finding in findings:
        key = (finding["code"], tuple(finding["lines"]))
        unique[key] = finding
    return sorted(unique.values(), key=lambda item: (item["code"], item["lines"]))


def iter_image_label_pairs(splits):
    for split, image_dirs in splits.items():
        for image_dir in image_dirs:
            label_dir = image_dir.parent / "labels"
            for image in sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
                relative = image.relative_to(image_dir).with_suffix(".txt")
                label = label_dir / relative
                if label.is_file():
                    yield split, image, label


def build_review(data_yaml, large_area=0.7, duplicate_iou=0.85, cross_class_iou=0.9):
    data_yaml, names, splits = resolve_dataset(data_yaml)
    items = []
    code_counts = Counter()
    class_counts = Counter()
    scanned = 0
    for split, image, label in iter_image_label_pairs(splits):
        scanned += 1
        boxes = parse_boxes(label, names)
        findings = inspect_boxes(boxes, large_area, duplicate_iou, cross_class_iou)
        if not findings:
            continue
        for finding in findings:
            code_counts[finding["code"]] += 1
        for box in boxes:
            if box["issue_codes"]:
                class_counts[box["class_name"]] += 1
        items.append({
            "split": split,
            "image": display_path(image),
            "label": display_path(label),
            "issue_codes": sorted({item["code"] for item in findings}),
            "max_box_area": max((box["area"] for box in boxes), default=0.0),
            "findings": findings,
            "boxes": boxes,
        })

    priority = {
        "large_box_with_inner_same_class": 0,
        "cross_class_overlap": 1,
        "duplicate_same_class": 2,
        "large_box": 3,
    }
    items.sort(key=lambda item: (
        min(priority.get(code, 99) for code in item["issue_codes"]),
        -item["max_box_area"],
        item["image"],
    ))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_yaml": display_path(data_yaml),
        "thresholds": {
            "large_box_area": large_area,
            "duplicate_same_class_iou": duplicate_iou,
            "cross_class_iou": cross_class_iou,
        },
        "summary": {
            "images_scanned": scanned,
            "images_flagged": len(items),
            "findings": sum(code_counts.values()),
            "findings_by_code": dict(sorted(code_counts.items())),
            "flagged_boxes_by_class": dict(sorted(class_counts.items())),
        },
        "items": items,
    }


def write_csv(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image", "label", "issue_codes", "suspicious_lines", "max_box_area"])
        for item in report["items"]:
            lines = sorted({line for finding in item["findings"] for line in finding["lines"]})
            writer.writerow([
                item["split"], item["image"], item["label"],
                "|".join(item["issue_codes"]), "|".join(map(str, lines)),
                f"{item['max_box_area']:.6f}",
            ])


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def fit_image(image, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                 max(1, round(image.shape[0] * scale))))
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas, scale, x, y


def draw_item(item, tile_width=440, tile_height=330):
    header = 64
    image = read_image(Path(item["image"]))
    if image is None:
        tile = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
        cv2.putText(tile, "IMAGE READ FAILED", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, SUSPICIOUS_COLOR, 2)
        return tile
    source_height, source_width = image.shape[:2]
    body, scale, offset_x, offset_y = fit_image(image, tile_width, tile_height - header)
    for box in item["boxes"]:
        x1, y1, x2, y2 = box["xyxy"]
        first = (round(x1 * source_width * scale + offset_x), round(y1 * source_height * scale + offset_y))
        second = (round(x2 * source_width * scale + offset_x), round(y2 * source_height * scale + offset_y))
        suspicious = bool(box["issue_codes"])
        color = SUSPICIOUS_COLOR if suspicious else COLORS.get(box["class_id"], (50, 180, 50))
        cv2.rectangle(body, first, second, color, 3 if suspicious else 2)
        label = f"L{box['line']} {box['class_name']} {box['area']:.2f}"
        cv2.putText(body, label, (first[0], max(18, first[1] + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    tile = np.full((tile_height, tile_width, 3), 250, dtype=np.uint8)
    title = f"{item['split']} | {Path(item['image']).name}"
    codes = ", ".join(item["issue_codes"])
    cv2.putText(tile, title[:62], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(tile, codes[:62], (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .46, SUSPICIOUS_COLOR, 1, cv2.LINE_AA)
    tile[header:] = body
    return tile


def write_contact_sheets(report, output_dir, columns=4, rows=2, max_images=0):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = report["items"][:max_images or None]
    per_page = columns * rows
    pages = []
    for page_index in range(0, len(items), per_page):
        batch = items[page_index:page_index + per_page]
        tiles = [draw_item(item) for item in batch]
        blank = np.full_like(tiles[0], 245) if tiles else None
        while len(tiles) < per_page:
            tiles.append(blank.copy())
        page_rows = [np.hstack(tiles[index:index + columns]) for index in range(0, per_page, columns)]
        sheet = np.vstack(page_rows)
        page = output_dir / f"label-review-{len(pages) + 1:03d}.jpg"
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(page)
        pages.append(display_path(page))
    return pages


def main():
    parser = argparse.ArgumentParser(description="Find suspicious YOLO labels and render review sheets")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--report", default="reports/label-review.json")
    parser.add_argument("--csv", default="reports/label-review.csv")
    parser.add_argument("--sheets", default="runs/label-review")
    parser.add_argument("--large-area", type=float, default=0.7)
    parser.add_argument("--duplicate-iou", type=float, default=0.85)
    parser.add_argument("--cross-class-iou", type=float, default=0.9)
    parser.add_argument("--max-sheet-images", type=int, default=0,
                        help="Limit rendered images; 0 renders every flagged image")
    args = parser.parse_args()
    for name, value in (("large-area", args.large_area), ("duplicate-iou", args.duplicate_iou),
                        ("cross-class-iou", args.cross_class_iou)):
        if not 0 < value <= 1:
            parser.error(f"--{name} must be in (0, 1]")
    if args.max_sheet_images < 0:
        parser.error("--max-sheet-images must be >= 0")

    report = build_review(args.data, args.large_area, args.duplicate_iou, args.cross_class_iou)
    write_csv(report, args.csv)
    pages = write_contact_sheets(report, args.sheets, max_images=args.max_sheet_images)
    report["contact_sheets"] = pages
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = report["summary"]
    print(f"Scanned {summary['images_scanned']} images; flagged {summary['images_flagged']}")
    print("Findings: " + ", ".join(f"{key}={value}" for key, value in summary["findings_by_code"].items()))
    print(f"Report: {report_path.resolve()}")
    print(f"CSV: {Path(args.csv).resolve()}")
    print(f"Contact sheets: {len(pages)} page(s) in {Path(args.sheets).resolve()}")


if __name__ == "__main__":
    main()
