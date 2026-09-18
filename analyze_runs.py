"""Summarize detection and tracking behavior from FactoryGuard run folders."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics

import cv2


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def dimensions(run_dir, summary):
    for candidate in (Path(run_dir) / "annotated.mp4", Path(summary.get("source", ""))):
        if not candidate.is_file():
            continue
        capture = cv2.VideoCapture(str(candidate))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if width > 0 and height > 0:
            return width, height
    raise ValueError(f"Cannot determine video dimensions for {run_dir}")


def iou(first, second):
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def analyze_run(run_dir, short_track_frames=10, overlap_iou=0.5):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    width, height = dimensions(run_dir, summary)
    by_class = defaultdict(list)
    with (run_dir / "tracks.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            row["frame"] = int(row["frame"])
            row["track_id"] = int(row["track_id"])
            row["confidence"] = float(row["confidence"])
            row["bbox"] = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
            by_class[row["class"]].append(row)

    classes = {}
    for class_name, rows in sorted(by_class.items()):
        id_counts = Counter(row["track_id"] for row in rows)
        frame_rows = defaultdict(list)
        for row in rows:
            frame_rows[row["frame"]].append(row)
        overlap_frames = 0
        for current in frame_rows.values():
            if any(iou(current[a]["bbox"], current[b]["bbox"]) >= overlap_iou
                   for a in range(len(current)) for b in range(a + 1, len(current))):
                overlap_frames += 1
        areas = [((row["bbox"][2] - row["bbox"][0]) * (row["bbox"][3] - row["bbox"][1]))
                 / (width * height) for row in rows]
        confidences = [row["confidence"] for row in rows]
        classes[class_name] = {
            "rows": len(rows),
            "frames_with_track": len(frame_rows),
            "unique_ids": len(id_counts),
            "short_ids": sum(count < short_track_frames for count in id_counts.values()),
            "max_tracks_per_frame": max(map(len, frame_rows.values()), default=0),
            "overlap_frames": overlap_frames,
            "overlap_frame_ratio": overlap_frames / len(frame_rows) if frame_rows else 0.0,
            "confidence_mean": statistics.fmean(confidences) if confidences else None,
            "confidence_below_0_25_ratio": sum(value < .25 for value in confidences) / len(confidences)
            if confidences else 0.0,
            "box_area_median": percentile(areas, .5),
            "box_area_p90": percentile(areas, .9),
            "box_area_max": max(areas, default=None),
            "box_area_over_0_5_ratio": sum(value > .5 for value in areas) / len(areas) if areas else 0.0,
        }
    return {
        "run": run_dir.as_posix(),
        "frames": summary.get("processed_frames"),
        "fps": summary.get("fps"),
        "events": summary.get("event_count"),
        "video_width": width,
        "video_height": height,
        "short_track_frames": short_track_frames,
        "overlap_iou": overlap_iou,
        "classes": classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare FactoryGuard tracking run diagnostics")
    parser.add_argument("--run", action="append", required=True, help="Run directory; repeat to compare")
    parser.add_argument("--output", default="reports/run-diagnostics.json")
    parser.add_argument("--short-track-frames", type=int, default=10)
    parser.add_argument("--overlap-iou", type=float, default=.5)
    args = parser.parse_args()
    report = {"runs": [analyze_run(path, args.short_track_frames, args.overlap_iou) for path in args.run]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for run in report["runs"]:
        print(run["run"])
        for class_name, values in run["classes"].items():
            print(f"  {class_name}: rows={values['rows']} ids={values['unique_ids']} "
                  f"short={values['short_ids']} max/frame={values['max_tracks_per_frame']} "
                  f"area-p90={values['box_area_p90']:.3f}")
    print(f"Report: {output.resolve()}")


if __name__ == "__main__":
    main()
